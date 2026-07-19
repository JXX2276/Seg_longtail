from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TargetClass:
    id: int
    name: str
    description: str
    prompts: tuple[str, ...]
    negative_examples: tuple[str, ...]


def _normalized(value: str) -> str:
    return " ".join(re.sub(r"[._/-]+", " ", value.lower()).split())


def target_classes(config: dict[str, Any]) -> list[TargetClass]:
    """Return a validated, contiguous YOLO taxonomy with legacy single-class support."""
    target = config["target"]
    raw_classes = target.get("classes")
    if not raw_classes:
        return [
            TargetClass(
                id=0,
                name=str(target["name"]),
                description=str(target.get("description", target["name"])),
                prompts=tuple(str(item) for item in target.get("prompts", [target["name"]])),
                negative_examples=tuple(
                    str(item) for item in target.get("negative_examples", [])
                ),
            )
        ]
    if not isinstance(raw_classes, list):
        raise ValueError("target.classes must be a list")

    classes: list[TargetClass] = []
    for index, item in enumerate(raw_classes):
        if not isinstance(item, dict) or not item.get("name"):
            raise ValueError(f"target.classes[{index}] requires a name")
        class_id = int(item.get("id", index))
        name = str(item["name"])
        classes.append(
            TargetClass(
                id=class_id,
                name=name,
                description=str(item.get("description", name)),
                prompts=tuple(str(value) for value in item.get("prompts", [name])),
                negative_examples=tuple(
                    str(value)
                    for value in item.get(
                        "negative_examples", target.get("negative_examples", [])
                    )
                ),
            )
        )
    ids = [item.id for item in classes]
    if ids != list(range(len(classes))):
        raise ValueError(
            "target.classes ids must be unique and contiguous in list order: 0, 1, ..., N-1"
        )
    names = [item.name for item in classes]
    if len(set(names)) != len(names):
        raise ValueError("target.classes names must be unique")
    return classes


def target_names(config: dict[str, Any]) -> dict[int, str]:
    return {item.id: item.name for item in target_classes(config)}


def target_prompts(config: dict[str, Any]) -> list[str]:
    return [prompt for item in target_classes(config) for prompt in item.prompts]


def class_for_label(label: str, classes: list[TargetClass]) -> TargetClass | None:
    """Map detector text or a student class name to the canonical target class."""
    if len(classes) == 1:
        return classes[0]
    label_key = _normalized(label)
    exact: dict[str, TargetClass] = {}
    for item in classes:
        aliases = (item.name, item.name.rsplit(".", 1)[-1], *item.prompts)
        for alias in aliases:
            key = _normalized(alias)
            previous = exact.get(key)
            if previous is not None and previous.id != item.id:
                raise ValueError(f"Ambiguous target class alias: {alias}")
            exact[key] = item
    return exact.get(label_key)


def dataset_names(dataset_dir: str | Path, fallback: dict[int, str]) -> dict[int, str]:
    """Read YOLO names from a dataset, allowing legacy one-class folders without data.yaml."""
    data_yaml = Path(dataset_dir) / "data.yaml"
    if not data_yaml.exists():
        if len(fallback) == 1:
            return dict(fallback)
        raise FileNotFoundError(f"Multi-class source dataset requires data.yaml: {data_yaml}")
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    raw_names = data.get("names")
    if isinstance(raw_names, list):
        names = {index: str(name) for index, name in enumerate(raw_names)}
    elif isinstance(raw_names, dict):
        names = {int(index): str(name) for index, name in raw_names.items()}
    else:
        raise ValueError(f"Dataset has invalid names mapping: {data_yaml}")
    return names


def class_remap(source_names: dict[int, str], canonical_names: dict[int, str]) -> dict[int, int]:
    """Build source-id -> canonical-id mapping by class name, never by numeric coincidence."""
    canonical_by_name = {name: class_id for class_id, name in canonical_names.items()}
    return {
        source_id: canonical_by_name[name]
        for source_id, name in source_names.items()
        if name in canonical_by_name
    }


def remap_yolo_label(text: str, remap: dict[int, int]) -> tuple[list[str], int]:
    """Remap a YOLO detection/segmentation label and drop classes outside the taxonomy."""
    lines: list[str] = []
    dropped = 0
    for raw_line in text.splitlines():
        parts = raw_line.strip().split()
        if not parts:
            continue
        try:
            source_id = int(parts[0])
        except ValueError as error:
            raise ValueError(f"Invalid YOLO class id in label line: {raw_line}") from error
        if source_id not in remap:
            dropped += 1
            continue
        lines.append(" ".join([str(remap[source_id]), *parts[1:]]))
    return lines, dropped
