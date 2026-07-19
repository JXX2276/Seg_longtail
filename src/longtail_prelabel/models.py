from __future__ import annotations

import gc
import json
import os
import re
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .core import Candidate


def _device_dtype(device: str):
    import torch

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    return torch, dtype


def release_model(model: Any) -> None:
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class YoloProposer:
    def __init__(
        self,
        model_path: str,
        device: str,
        confidence: float,
        iou: float,
        image_size: int = 1280,
        tile_grid: int = 1,
        tile_overlap: float = 0.2,
    ):
        os.environ.setdefault("YOLO_OFFLINE", "true")
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.device = device
        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size
        self.tile_grid = tile_grid
        self.tile_overlap = tile_overlap

    @staticmethod
    def _candidates(results, offset_x: int = 0, offset_y: int = 0, tiled: bool = False):
        candidates = []
        if results.boxes is None:
            return candidates
        names = results.names
        for box, score, class_id in zip(
            results.boxes.xyxy.cpu().tolist(),
            results.boxes.conf.cpu().tolist(),
            results.boxes.cls.cpu().tolist(),
        ):
            mapped = [
                box[0] + offset_x,
                box[1] + offset_y,
                box[2] + offset_x,
                box[3] + offset_y,
            ]
            source = "student_tile" if tiled else "student"
            sources = ["student", "student_tile"] if tiled else ["student"]
            candidates.append(
                Candidate(
                    mapped,
                    float(score),
                    source,
                    str(names.get(int(class_id), class_id)),
                    sources=sources,
                )
            )
        return candidates

    def predict(self, image_path: str, class_ids: list[int] | None = None) -> list[Candidate]:
        results = self.model.predict(
            source=image_path,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            device=self.device,
            classes=class_ids or None,
            verbose=False,
        )[0]
        candidates = self._candidates(results)
        if self.tile_grid <= 1:
            return candidates
        if not 0 <= self.tile_overlap < 1:
            raise ValueError("student_tile_overlap must be in [0, 1)")
        image = Image.open(image_path).convert("RGB")
        denominator = self.tile_grid - self.tile_overlap * (self.tile_grid - 1)
        tile_width = min(image.width, round(image.width / denominator))
        tile_height = min(image.height, round(image.height / denominator))
        x_starts = [
            round(index * (image.width - tile_width) / (self.tile_grid - 1))
            for index in range(self.tile_grid)
        ]
        y_starts = [
            round(index * (image.height - tile_height) / (self.tile_grid - 1))
            for index in range(self.tile_grid)
        ]
        offsets = [(x, y) for y in y_starts for x in x_starts]
        tiles = [
            image.crop((x, y, x + tile_width, y + tile_height)) for x, y in offsets
        ]
        tile_results = self.model.predict(
            source=tiles,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            device=self.device,
            classes=class_ids or None,
            verbose=False,
        )
        for tile_result, (offset_x, offset_y) in zip(tile_results, offsets):
            candidates.extend(self._candidates(tile_result, offset_x, offset_y, tiled=True))
        return candidates


class GroundingDinoProposer:
    def __init__(
        self,
        model_path: str,
        device: str,
        box_threshold: float,
        text_threshold: float,
        tile_grid: int = 1,
        tile_overlap: float = 0.2,
        tile_box_threshold: float | None = None,
    ):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        torch, _ = _device_dtype(device)
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_path, local_files_only=True, dtype=torch.float32
        ).to(device)
        self.model.eval()
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.tile_grid = tile_grid
        self.tile_overlap = tile_overlap
        self.tile_box_threshold = (
            tile_box_threshold if tile_box_threshold is not None else box_threshold
        )

    def _predict_image(
        self,
        image: Image.Image,
        prompts: list[str],
        offset_x: int = 0,
        offset_y: int = 0,
        tiled: bool = False,
    ) -> list[Candidate]:
        text = ". ".join(prompt.strip().lower().rstrip(".") for prompt in prompts) + "."
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.tile_box_threshold if tiled else self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        labels = result["text_labels"] if "text_labels" in result else result["labels"]
        source = "grounding_dino"
        # A tile is another view from the same detector, not independent model agreement.
        sources = ["grounding_dino"]
        return [
            Candidate(
                [
                    float(box[0]) + offset_x,
                    float(box[1]) + offset_y,
                    float(box[2]) + offset_x,
                    float(box[3]) + offset_y,
                ],
                float(score),
                source,
                str(label),
                sources=sources,
                reason="grounding_tile" if tiled else "",
                view="tile" if tiled else "full",
            )
            for box, score, label in zip(result["boxes"].cpu(), result["scores"].cpu(), labels)
        ]

    def predict(self, image_path: str, prompts: list[str]) -> list[Candidate]:
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
        candidates = self._predict_image(image, prompts)
        if self.tile_grid <= 1:
            return candidates
        if not 0 <= self.tile_overlap < 1:
            raise ValueError("grounding_tile_overlap must be in [0, 1)")
        denominator = self.tile_grid - self.tile_overlap * (self.tile_grid - 1)
        tile_width = min(image.width, round(image.width / denominator))
        tile_height = min(image.height, round(image.height / denominator))
        x_starts = [
            round(index * (image.width - tile_width) / (self.tile_grid - 1))
            for index in range(self.tile_grid)
        ]
        y_starts = [
            round(index * (image.height - tile_height) / (self.tile_grid - 1))
            for index in range(self.tile_grid)
        ]
        for offset_y in y_starts:
            for offset_x in x_starts:
                tile = image.crop(
                    (offset_x, offset_y, offset_x + tile_width, offset_y + tile_height)
                )
                candidates.extend(
                    self._predict_image(tile, prompts, offset_x, offset_y, tiled=True)
                )
        return candidates


class QwenVerifier:
    def __init__(
        self,
        model_path: str,
        device: str,
        max_new_tokens: int,
        gpu_memory: str = "13GiB",
        cpu_memory: str = "48GiB",
        offload_folder: str = "workspace/qwen_offload",
    ):
        from transformers import (
            AutoProcessor,
            Qwen2_5_VLForConditionalGeneration,
        )

        torch, dtype = _device_dtype(device)
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        model_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "dtype": dtype,
            "attn_implementation": "sdpa",
        }
        if device.startswith("cuda"):
            device_index = int(device.split(":", 1)[1]) if ":" in device else 0
            offload = os.path.abspath(offload_folder)
            os.makedirs(offload, exist_ok=True)
            model_kwargs.update(
                {
                    "device_map": "auto",
                    "max_memory": {device_index: gpu_memory, "cpu": cpu_memory},
                    "offload_folder": offload,
                    "offload_state_dict": True,
                }
            )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
        if not device.startswith("cuda"):
            self.model = self.model.to(device)
        self.model.eval()
        # The bundled generation config contains sampling-only flags. We use deterministic decoding.
        self.model.generation_config.temperature = None
        self.device = self.model.get_input_embeddings().weight.device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.last_response = ""

    @staticmethod
    def _parse_many(text: str, expected_ids: list[int]) -> dict[int, tuple[str, float, str]]:
        """Parse a strict multi-box response; malformed or missing entries become uncertain."""
        payload: Any = None
        decoder = json.JSONDecoder()
        for match in re.finditer(r"[\[{]", text):
            try:
                payload, _ = decoder.raw_decode(text[match.start() :])
                break
            except json.JSONDecodeError:
                continue

        if isinstance(payload, dict):
            if isinstance(payload.get("results"), list):
                payload = payload["results"]
            else:
                payload = [
                    {"id": candidate_id, "decision": decision}
                    for candidate_id, decision in payload.items()
                ]
        elif isinstance(payload, list) and all(isinstance(item, str) for item in payload):
            payload = [
                {"id": candidate_id, "decision": decision}
                for candidate_id, decision in enumerate(payload, start=1)
            ]
        if not isinstance(payload, list):
            payload = []

        allowed = set(expected_ids)
        parsed: dict[int, tuple[str, float, str]] = {}
        for item in payload:
            if not isinstance(item, dict):
                continue
            try:
                candidate_id = int(item.get("id"))
            except (TypeError, ValueError):
                continue
            if candidate_id not in allowed or candidate_id in parsed:
                continue
            decision = str(item.get("decision", "uncertain")).strip().lower()
            decision = {
                "t": "target",
                "n": "not_target",
                "u": "uncertain",
            }.get(decision, decision)
            if decision not in {"target", "not_target", "uncertain"}:
                decision = "uncertain"
            confidence = {"target": 0.9, "not_target": 0.95, "uncertain": 0.5}[decision]
            parsed[candidate_id] = (decision, confidence, "qwen_multi_box")

        invalid_reason = "qwen_missing_or_invalid_response"
        for candidate_id in expected_ids:
            parsed.setdefault(candidate_id, ("uncertain", 0.5, invalid_reason))
        return parsed

    @staticmethod
    def _annotate(
        image: Image.Image,
        candidates: list[Candidate],
        crop_expansion: float,
        tile_size: int,
    ) -> Image.Image:
        """Build a compact grid of numbered, context-expanded candidate crops."""
        columns = min(4, max(1, len(candidates)))
        rows = (len(candidates) + columns - 1) // columns
        annotated = Image.new("RGB", (columns * tile_size, rows * tile_size), "black")
        font_size = max(18, tile_size // 10)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()
        line_width = max(3, tile_size // 80)
        for candidate_id, candidate in enumerate(candidates, start=1):
            x1, y1, x2, y2 = candidate.bbox
            dx = max(8.0, (x2 - x1) * crop_expansion)
            dy = max(8.0, (y2 - y1) * crop_expansion)
            crop_x1, crop_y1 = max(0, int(x1 - dx)), max(0, int(y1 - dy))
            crop_x2 = min(image.width, int(x2 + dx + 1))
            crop_y2 = min(image.height, int(y2 + dy + 1))
            crop = image.crop((crop_x1, crop_y1, crop_x2, crop_y2)).resize(
                (tile_size, tile_size), Image.Resampling.BILINEAR
            )
            draw = ImageDraw.Draw(crop)
            scale_x = tile_size / max(1, crop_x2 - crop_x1)
            scale_y = tile_size / max(1, crop_y2 - crop_y1)
            local_box = (
                (x1 - crop_x1) * scale_x,
                (y1 - crop_y1) * scale_y,
                (x2 - crop_x1) * scale_x,
                (y2 - crop_y1) * scale_y,
            )
            draw.rectangle(local_box, outline="red", width=line_width)
            label = str(candidate_id)
            left, top, right, bottom = draw.textbbox((4, 4), label, font=font)
            label_height = bottom - top + 8
            label_width = right - left + 10
            draw.rectangle((0, 0, label_width, label_height), fill="red")
            draw.text((5, 2), label, fill="white", font=font)
            column = (candidate_id - 1) % columns
            row = (candidate_id - 1) // columns
            annotated.paste(crop, (column * tile_size, row * tile_size))
        return annotated

    def verify_many(
        self,
        image_path: str,
        candidates: list[Candidate],
        class_name: str,
        description: str,
        negative_examples: list[str],
        crop_expansion: float,
        tile_size: int,
    ) -> dict[int, tuple[str, float, str]]:
        """Verify all numbered candidates in one model call."""
        from qwen_vl_utils import process_vision_info

        image = Image.open(image_path).convert("RGB")
        annotated = self._annotate(image, candidates, crop_expansion, tile_size)
        negative_text = (
            "、".join(negative_examples) if negative_examples else "正常道路参与者和设施"
        )
        prompt = (
            "Do not infer an object from the red box itself. First identify a clearly visible, "
            "discrete physical object inside each box. Empty road, road texture, lane markings, "
            "shadows, vegetation, grass, buildings, fences and blur are always N. "
            "A complete car, truck, bus, "
            "motorcycle, bicycle, person, traffic sign, or traffic light is ALWAYS N, whether it "
            "is moving, parked, or apparently abandoned. T is only for loose debris, fragments, "
            "discarded material, or detached objects with no normal traffic function. "
            f"你是自动驾驶数据标注审核员。拼图每格包含一个红色编号候选框。判断主体是否属于 {class_name}。"
            f"类别定义：{description}\n"
            "只有同时满足以下条件才输出 target：主体是独立、松散、遗弃或已经脱落的物体，"
            "并且位于道路、轨行区或其附近。"
            f"这些通常不是目标：{negative_text}。完整路面和大面积背景也不是目标。"
            "红框过大、没有清晰独立物体、看不清主体或证据不足时必须输出 uncertain，不要猜测。"
            "宁可输出U进入人工复核，也不能把背景或正常物体猜成T。"
            f"只输出一个长度严格等于{len(candidates)}的JSON字符串数组，顺序对应编号1到"
            f"{len(candidates)}。T表示target，N表示not_target，U表示uncertain。"
            '禁止解释或Markdown，例如：["N","T","U"]'
        )
        negative_examples_text = ", ".join(negative_examples) or "normal scene content"
        # Keep the semantic question class-specific so one verifier can serve a shared
        # multi-class student without turning every candidate into debris.
        prompt = (
            "You are reviewing autonomous-driving object candidates. Each numbered tile "
            f"contains one red candidate box. The requested class is '{class_name}', defined "
            f"as: {description}. Output T only when a clearly visible object in the box matches "
            "that class. Output N for background or a different class, including these common "
            f"negatives: {negative_examples_text}. Output U only when image evidence is "
            "insufficient. Judge the pixels, not the red box or detector label. "
            f"Return only one JSON string array of exactly {len(candidates)} entries in number "
            f"order 1 through {len(candidates)}, using only T, N, or U. "
            'Example: ["N","T","U"]'
        )
        if any(candidate.view == "region" for candidate in candidates):
            prompt = (
                "You are reviewing autonomous-driving candidate REGIONS. Each numbered tile "
                "contains one red region box. A large box is expected: inspect everything "
                f"inside it. The requested class is '{class_name}', defined as: {description}. "
                "Output T if the region contains at least one clearly visible instance of that "
                f"class. Output N for background, another class, or: {negative_examples_text}. "
                "Output U only when evidence is truly insufficient. The target need not fill "
                "the red box. "
                f"Return only one JSON string array of exactly {len(candidates)} entries in "
                f"number order 1 through {len(candidates)}, using only T, N, or U. "
                'Example: ["N","T","U"]'
            )
        elif any(candidate.view == "mask_crop" for candidate in candidates):
            prompt = (
                "You are reviewing SAM-segmented autonomous-driving object candidates. Each "
                "numbered tile contains one red box tightly enclosing a segmented physical "
                f"object. The requested class is '{class_name}', defined as: {description}. "
                "Judge the visible object, not box size or segmentation quality. Output T when "
                f"the object matches that class, N for another class or {negative_examples_text}, "
                "and U only when it cannot be identified from the pixels. Small and partially "
                "occluded targets may still be T. "
                f"Return only one JSON string array of exactly {len(candidates)} entries in "
                f"number order 1 through {len(candidates)}, using only T, N, or U. "
                'Example: ["N","T","U"]'
            )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": annotated},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        chat = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = process_vision_info(messages)
        inputs = self.processor(
            text=[chat], images=images, videos=videos, padding=True, return_tensors="pt"
        ).to(self.device, dtype=self.dtype)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        generated = generated[:, inputs.input_ids.shape[1] :]
        text = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        self.last_response = text
        return self._parse_many(text, list(range(1, len(candidates) + 1)))


class Sam2Refiner:
    def __init__(self, model_path: str, device: str, mask_threshold: float = 0.0):
        from transformers import Sam2Model, Sam2Processor

        torch, dtype = _device_dtype(device)
        self.torch = torch
        self.processor = Sam2Processor.from_pretrained(model_path, local_files_only=True)
        self.model = Sam2Model.from_pretrained(model_path, local_files_only=True, dtype=dtype).to(
            device
        )
        self.model.eval()
        self.device = device
        self.dtype = dtype
        self.mask_threshold = mask_threshold

    def refine(self, image_path: str, boxes: list[list[float]]) -> list[dict[str, Any]]:
        import cv2

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, input_boxes=[boxes], return_tensors="pt").to(
            self.device, dtype=self.dtype
        )
        with self.torch.inference_mode():
            outputs = self.model(**inputs, multimask_output=True)
        masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"]
        )[0]
        scores = outputs.iou_scores.detach().float().cpu()[0]
        records = []
        for object_index in range(len(boxes)):
            best = int(scores[object_index].argmax().item())
            mask = (masks[object_index, best].numpy() > self.mask_threshold).astype(np.uint8)
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                records.append({"bbox": boxes[object_index], "score": 0.0, "segmentation": []})
                continue
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = max(contours, key=cv2.contourArea)
            contour = cv2.approxPolyDP(
                contour, max(1.0, 0.001 * cv2.arcLength(contour, True)), True
            )
            polygon = contour.reshape(-1, 2).astype(float).flatten().tolist()
            records.append(
                {
                    "bbox": [
                        float(xs.min()),
                        float(ys.min()),
                        float(xs.max() + 1),
                        float(ys.max() + 1),
                    ],
                    "score": float(scores[object_index, best].item()),
                    "segmentation": polygon,
                }
            )
        return records

    def refine_points(
        self, image_path: str, points: list[list[float]]
    ) -> list[dict[str, Any]]:
        """Create one mask per positive human click."""
        import cv2

        if not points:
            return []
        image = Image.open(image_path).convert("RGB")
        input_points = [[[point] for point in points]]
        input_labels = [[[1] for _ in points]]
        inputs = self.processor(
            images=image,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        ).to(self.device, dtype=self.dtype)
        with self.torch.inference_mode():
            outputs = self.model(**inputs, multimask_output=True)
        masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"]
        )[0]
        scores = outputs.iou_scores.detach().float().cpu()[0]
        records = []
        for point_index in range(len(points)):
            best = int(scores[point_index].argmax().item())
            mask = (masks[point_index, best].numpy() > self.mask_threshold).astype(np.uint8)
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                records.append({"bbox": points[point_index] * 2, "score": 0.0, "segmentation": []})
                continue
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = max(contours, key=cv2.contourArea)
            contour = cv2.approxPolyDP(
                contour, max(1.0, 0.001 * cv2.arcLength(contour, True)), True
            )
            records.append(
                {
                    "bbox": [
                        float(xs.min()),
                        float(ys.min()),
                        float(xs.max() + 1),
                        float(ys.max() + 1),
                    ],
                    "score": float(scores[point_index, best].item()),
                    "segmentation": contour.reshape(-1, 2).astype(float).flatten().tolist(),
                }
            )
        return records
