from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import yaml


def _class_ids(label: Path) -> set[int]:
    return {
        int(line.split()[0])
        for line in label.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--target-per-rare-class", type=int, default=800)
    parser.add_argument("--max-repeat", type=int, default=8)
    parser.add_argument("--dominant-images", type=int, default=1200)
    parser.add_argument("--background-images", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=339)
    parser.add_argument("--output-prefix", default="balanced")
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    data = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))
    names = {int(key): str(value) for key, value in data["names"].items()}
    dominant_id = 2
    rare_ids = sorted(set(names) - {dominant_id})
    images = {
        image.stem: image.resolve()
        for image in (dataset / "images" / "train").glob("*")
        if image.is_file()
    }
    records = []
    class_image_counts: Counter[int] = Counter()
    for stem, image in images.items():
        label = dataset / "labels" / "train" / f"{stem}.txt"
        classes = _class_ids(label) if label.exists() else set()
        records.append((image, classes))
        class_image_counts.update(classes)

    repeats = {
        class_id: min(
            args.max_repeat,
            max(1, math.ceil(args.target_per_rare_class / max(1, class_image_counts[class_id]))),
        )
        for class_id in rare_ids
    }
    rare_records = [(image, classes) for image, classes in records if classes & set(rare_ids)]
    dominant_records = [
        (image, classes) for image, classes in records if classes == {dominant_id}
    ]
    background_records = [(image, classes) for image, classes in records if not classes]
    rng = random.Random(args.seed)
    rng.shuffle(dominant_records)
    rng.shuffle(background_records)

    scheduled: list[Path] = []
    scheduled_class_appearances: Counter[int] = Counter()
    for image, classes in rare_records:
        repeat = max(repeats[class_id] for class_id in classes if class_id in repeats)
        scheduled.extend([image] * repeat)
        for class_id in classes:
            scheduled_class_appearances[class_id] += repeat
    for image, classes in dominant_records[: args.dominant_images]:
        scheduled.append(image)
        scheduled_class_appearances.update(classes)
    scheduled.extend(image for image, _ in background_records[: args.background_images])
    rng.shuffle(scheduled)

    train_list = dataset / f"{args.output_prefix}_train.txt"
    portable_paths = ["./" + path.relative_to(dataset).as_posix() for path in scheduled]
    train_list.write_text("\n".join(portable_paths) + "\n", encoding="utf-8")
    balanced_yaml = dataset / f"{args.output_prefix}_data.yaml"
    balanced_yaml.write_text(
        yaml.safe_dump(
            {
                "path": ".",
                "train": train_list.name,
                "val": "images/val",
                "names": names,
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    summary = {
        "unique_train_images": len(records),
        "scheduled_image_occurrences": len(scheduled),
        "class_image_counts": {names[key]: class_image_counts[key] for key in names},
        "rare_repeat_factors": {names[key]: repeats[key] for key in rare_ids},
        "scheduled_class_image_appearances": {
            names[key]: scheduled_class_appearances[key] for key in names
        },
        "dominant_only_images_used": min(args.dominant_images, len(dominant_records)),
        "background_images_used": min(args.background_images, len(background_records)),
        "seed": args.seed,
        "output_prefix": args.output_prefix,
    }
    (dataset / f"{args.output_prefix}_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
