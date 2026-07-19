from __future__ import annotations

import base64
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from pycocotools import mask as mask_utils

from .core import write_jsonl


def _load(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def mask_to_polygon(mask: np.ndarray) -> list[float]:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    epsilon = max(1.0, 0.001 * cv2.arcLength(contour, True))
    contour = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    return contour.astype(float).flatten().tolist() if len(contour) >= 3 else []


def decode_nuimages_mask(encoded: dict[str, Any]) -> np.ndarray:
    """Decode nuImages' base64-wrapped COCO RLE mask."""
    rle = dict(encoded)
    counts = rle.get("counts")
    if isinstance(counts, str):
        rle["counts"] = base64.b64decode(counts)
    return mask_utils.decode(rle)


def _place_image(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return
    if mode == "symlink":
        os.symlink(source.resolve(), target)
    elif mode == "hardlink":
        os.link(source, target)
    else:
        shutil.copy2(source, target)


def convert_nuimages(config: dict[str, Any]) -> dict[str, Any]:
    dataset = config["dataset"]
    root = Path(dataset["root"])
    metadata_root = Path(dataset.get("metadata_root", root))
    meta = metadata_root / dataset["version"]
    workspace = Path(dataset["workspace"])
    yolo_root = workspace / "base_dataset"

    categories = _load(meta / "category.json")
    all_sample_data = _load(meta / "sample_data.json")

    excluded_sample_tokens: set[str] = set()
    if dataset.get("exclude_index"):
        exclude_indexes = dataset["exclude_index"]
        if not isinstance(exclude_indexes, list):
            exclude_indexes = [exclude_indexes]
        for index_path in exclude_indexes:
            with Path(index_path).open(encoding="utf-8") as handle:
                excluded_sample_tokens.update(
                    json.loads(line)["sample_token"] for line in handle if line.strip()
                )
    excluded_dataset_images = 0
    if dataset.get("exclude_datasets"):
        exclude_datasets = dataset["exclude_datasets"]
        if not isinstance(exclude_datasets, list):
            exclude_datasets = [exclude_datasets]
        excluded_image_tokens: set[str] = set()
        for dataset_root in exclude_datasets:
            image_root = Path(dataset_root) / "images"
            excluded_image_tokens.update(
                path.stem
                for split in ("train", "val")
                for path in (image_root / split).glob("*")
                if path.is_file()
            )
        known_image_tokens = {item["token"] for item in all_sample_data}
        missing_image_tokens = excluded_image_tokens - known_image_tokens
        if missing_image_tokens:
            preview = ", ".join(sorted(missing_image_tokens)[:5])
            raise ValueError(
                f"exclude_datasets contains {len(missing_image_tokens)} unknown image tokens: {preview}"
            )
        excluded_sample_tokens.update(
            item["sample_token"]
            for item in all_sample_data
            if item["token"] in excluded_image_tokens
        )
        excluded_dataset_images = len(excluded_image_tokens)
    all_sample_tokens = sorted(
        {item["sample_token"] for item in all_sample_data} - excluded_sample_tokens
    )
    random.Random(dataset["split_seed"]).shuffle(all_sample_tokens)
    max_samples = dataset.get("max_samples")
    if max_samples is not None:
        max_samples = int(max_samples)
        if max_samples <= 0:
            raise ValueError("dataset.max_samples must be a positive integer or null")
        selected_sample_tokens = set(all_sample_tokens[:max_samples])
    else:
        selected_sample_tokens = set(all_sample_tokens)

    sample_data = [
        item for item in all_sample_data if item["sample_token"] in selected_sample_tokens
    ]
    selected_image_tokens = {item["token"] for item in sample_data}
    del all_sample_data
    annotations = [
        ann
        for ann in _load(meta / "object_ann.json")
        if ann["sample_data_token"] in selected_image_tokens
    ]

    present_tokens = {ann["category_token"] for ann in annotations}
    present_categories = [cat for cat in categories if cat["token"] in present_tokens]
    class_id = {cat["token"]: index for index, cat in enumerate(present_categories)}
    names = [cat["name"] for cat in present_categories]

    ann_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ann in annotations:
        ann_by_image[ann["sample_data_token"]].append(ann)

    sample_tokens = [token for token in all_sample_tokens if token in selected_sample_tokens]
    val_count = max(1, round(len(sample_tokens) * dataset["val_fraction"]))
    val_tokens = set(sample_tokens[:val_count])

    index_rows = []
    written = {"train": 0, "val": 0}
    for item in sorted(sample_data, key=lambda row: (row["sample_token"], row["timestamp"])):
        image_path = root / item["filename"]
        index_rows.append(
            {
                "image_path": str(image_path.resolve()),
                "sample_data_token": item["token"],
                "sample_token": item["sample_token"],
                "timestamp": item["timestamp"],
                "is_key_frame": item["is_key_frame"],
                "width": item["width"],
                "height": item["height"],
            }
        )
        if not item["is_key_frame"]:
            continue

        split = "val" if item["sample_token"] in val_tokens else "train"
        stem = item["token"]
        target_image = yolo_root / "images" / split / f"{stem}{image_path.suffix.lower()}"
        target_label = yolo_root / "labels" / split / f"{stem}.txt"
        _place_image(image_path, target_image, dataset.get("image_mode", "copy"))
        target_label.parent.mkdir(parents=True, exist_ok=True)

        lines = []
        for ann in ann_by_image.get(item["token"], []):
            polygon = []
            if ann.get("mask"):
                decoded = decode_nuimages_mask(ann["mask"])
                polygon = mask_to_polygon(decoded)
            if not polygon:
                x1, y1, x2, y2 = ann["bbox"]
                polygon = [x1, y1, x2, y1, x2, y2, x1, y2]
            normalized = []
            for index, value in enumerate(polygon):
                normalized.append(value / (item["width"] if index % 2 == 0 else item["height"]))
            lines.append(
                str(class_id[ann["category_token"]])
                + " "
                + " ".join(f"{v:.6f}" for v in normalized)
            )
        target_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written[split] += 1

    workspace.mkdir(parents=True, exist_ok=True)
    write_jsonl(workspace / "index.jsonl", index_rows)
    with (yolo_root / "data.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "path": str(yolo_root.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": dict(enumerate(names)),
            },
            handle,
            allow_unicode=True,
            sort_keys=False,
        )
    summary = {
        "available_samples": len(all_sample_tokens),
        "selected_samples": len(sample_tokens),
        "excluded_samples": len(excluded_sample_tokens),
        "excluded_dataset_images": excluded_dataset_images,
        "frames": len(sample_data),
        "key_frames": sum(bool(item["is_key_frame"]) for item in sample_data),
        "annotations": len(annotations),
        "classes": names,
        "split": written,
        "index": str(workspace / "index.jsonl"),
        "data_yaml": str(yolo_root / "data.yaml"),
    }
    (workspace / "conversion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
