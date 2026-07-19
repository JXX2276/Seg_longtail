from __future__ import annotations

import os
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "workspace" / "teacher" / "generated_masks"
GROUND_TRUTH = ROOT / "workspace" / "teacher" / "base_dataset"
OUTPUT = ROOT / "workspace" / "dataset"
NAMES = {
    0: "movable_object.barrier",
    1: "movable_object.debris",
    2: "movable_object.pushable_pullable",
    3: "movable_object.trafficcone",
}


def link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        os.link(source, target)


def copy_train() -> int:
    count = 0
    for image in sorted((GENERATED / "images" / "train").glob("*")):
        if not image.is_file():
            continue
        label = GENERATED / "labels" / "train" / f"{image.stem}.txt"
        if not label.exists():
            raise FileNotFoundError(f"Missing label: {label}")
        link(image, OUTPUT / "images" / "train" / image.name)
        link(label, OUTPUT / "labels" / "train" / label.name)
        count += 1
    return count


def copy_validation() -> int:
    source_yaml = yaml.safe_load((GROUND_TRUTH / "data.yaml").read_text(encoding="utf-8"))
    raw_names = source_yaml["names"]
    source_names = (
        {index: name for index, name in enumerate(raw_names)}
        if isinstance(raw_names, list)
        else {int(index): name for index, name in raw_names.items()}
    )
    canonical = {name: index for index, name in NAMES.items()}
    remap = {
        source_id: canonical[name]
        for source_id, name in source_names.items()
        if name in canonical
    }

    count = 0
    for image in sorted((GROUND_TRUTH / "images" / "val").glob("*")):
        if not image.is_file():
            continue
        source_label = GROUND_TRUTH / "labels" / "val" / f"{image.stem}.txt"
        if not source_label.exists():
            raise FileNotFoundError(f"Missing label: {source_label}")
        lines = []
        for line in source_label.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if parts and int(parts[0]) in remap:
                lines.append(" ".join([str(remap[int(parts[0])]), *parts[1:]]))
        link(image, OUTPUT / "images" / "val" / image.name)
        label = OUTPUT / "labels" / "val" / source_label.name
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
        count += 1
    return count


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(
            "workspace/dataset already exists; remove it before rebuilding from teacher output"
        )
    train = copy_train()
    val = copy_validation()
    if train == 0:
        raise RuntimeError("Teacher produced no accepted masks; inspect segmented.jsonl")
    data = {"path": ".", "train": "images/train", "val": "images/val", "names": NAMES}
    (OUTPUT / "data.yaml").write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    print(f"train pseudo images: {train}")
    print(f"val ground-truth images: {val}")


if __name__ == "__main__":
    main()
