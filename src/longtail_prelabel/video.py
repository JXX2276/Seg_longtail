from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from .core import Candidate, read_jsonl, write_jsonl


def propagate_video(
    config: dict[str, Any],
    index_path: str,
    seeds_path: str,
    output_path: str,
    device: str,
    shard_index: int,
    num_shards: int,
) -> None:
    import cv2
    import torch
    from transformers import Sam2VideoModel, Sam2VideoProcessor

    sequences: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(index_path):
        sequences[row["sample_token"]].append(row)
    for frames in sequences.values():
        frames.sort(key=lambda item: item["timestamp"])

    seeds_by_image: dict[str, list[Candidate]] = defaultdict(list)
    for row in read_jsonl(seeds_path):
        for item in row.get("accepted", []):
            seeds_by_image[str(Path(row["image_path"]).resolve())].append(Candidate.from_dict(item))

    model_path = config["models"]["segment_teacher"]
    processor = Sam2VideoProcessor.from_pretrained(model_path, local_files_only=True)
    model = Sam2VideoModel.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16
    ).to(device)
    model.eval()

    output_rows = []
    sequence_items = sorted(sequences.items())
    for seq_index, (_, frames) in enumerate(tqdm(sequence_items, desc="propagate")):
        if seq_index % num_shards != shard_index:
            continue
        seed_frame_index = None
        seed_candidates = []
        for frame_index, frame in enumerate(frames):
            matches = seeds_by_image.get(str(Path(frame["image_path"]).resolve()), [])
            if matches:
                seed_frame_index = frame_index
                seed_candidates = matches
                break
        if seed_frame_index is None:
            continue

        images = [Image.open(frame["image_path"]).convert("RGB") for frame in frames]
        session = processor.init_video_session(
            video=images, inference_device=device, dtype=torch.bfloat16
        )
        obj_ids = list(range(1, len(seed_candidates) + 1))
        points = []
        labels = []
        for candidate in seed_candidates:
            x1, y1, x2, y2 = candidate.bbox
            points.append([[(x1 + x2) / 2, (y1 + y2) / 2]])
            labels.append([1])
        processor.add_inputs_to_inference_session(
            inference_session=session,
            frame_idx=seed_frame_index,
            obj_ids=obj_ids,
            input_points=[points],
            input_labels=[labels],
        )
        with torch.inference_mode():
            model(inference_session=session, frame_idx=seed_frame_index)
            sequence_output = {}
            for reverse in (False, True):
                iterator = model.propagate_in_video_iterator(
                    session, start_frame_idx=seed_frame_index, reverse=reverse
                )
                for result in iterator:
                    masks = processor.post_process_masks(
                        [result.pred_masks],
                        original_sizes=[[session.video_height, session.video_width]],
                        binarize=False,
                    )[0]
                    frame = frames[result.frame_idx]
                    accepted = []
                    for object_index, _ in enumerate(session.obj_ids):
                        mask = (masks[object_index, 0].float().cpu().numpy() > 0).astype(np.uint8)
                        ys, xs = np.where(mask > 0)
                        if len(xs) == 0:
                            continue
                        contours, _ = cv2.findContours(
                            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        contour = max(contours, key=cv2.contourArea)
                        polygon = contour.reshape(-1, 2).astype(float).flatten().tolist()
                        template = seed_candidates[object_index]
                        item = template.to_dict()
                        item.update(
                            {
                                "bbox": [
                                    float(xs.min()),
                                    float(ys.min()),
                                    float(xs.max() + 1),
                                    float(ys.max() + 1),
                                ],
                                "segmentation": polygon,
                                "source": "sam2_video",
                                "sources": sorted(set(template.sources + ["sam2_video"])),
                            }
                        )
                        accepted.append(item)
                    sequence_output[result.frame_idx] = {
                        **frame,
                        "accepted": accepted,
                        "review": [],
                    }
            output_rows.extend(sequence_output[index] for index in sorted(sequence_output))
        session.reset_inference_session()
    write_jsonl(output_path, output_rows)
