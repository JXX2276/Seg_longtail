from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import yaml


RARE_CLASSES = (
    "movable_object.barrier",
    "movable_object.debris",
    "movable_object.trafficcone",
)

ROOT = Path(__file__).resolve().parents[1]


def _order(path: Path) -> tuple[int, str]:
    match = re.search(r"epoch(\d+)", path.stem)
    return (int(match.group(1)), path.name) if match else (10**9, path.name)


def _evaluate(config: dict, data: Path, checkpoint: Path) -> dict:
    from ultralytics import YOLO

    train = config["training"]
    result = YOLO(str(checkpoint)).val(
        data=data.resolve().relative_to(ROOT).as_posix(),
        split="val",
        imgsz=train["image_size"],
        batch=train["batch"],
        device=train["device"],
        workers=train["workers"],
        plots=False,
        verbose=False,
    )
    values = result.results_dict
    metrics = {
        "box_precision": float(values.get("metrics/precision(B)", 0.0)),
        "box_recall": float(values.get("metrics/recall(B)", 0.0)),
        "box_map50": float(values.get("metrics/mAP50(B)", 0.0)),
        "mask_precision": float(values.get("metrics/precision(M)", 0.0)),
        "mask_recall": float(values.get("metrics/recall(M)", 0.0)),
        "mask_map50": float(values.get("metrics/mAP50(M)", 0.0)),
    }
    names = result.names if isinstance(result.names, dict) else dict(enumerate(result.names))
    per_class = {str(name): {} for name in names.values()}
    for prefix, metric in (("box", result.box), ("mask", result.seg)):
        for class_id, name in names.items():
            class_id = int(class_id)
            per_class[str(name)].update(
                {
                    f"{prefix}_precision": float(metric.p[class_id]),
                    f"{prefix}_recall": float(metric.r[class_id]),
                    f"{prefix}_map50": float(metric.ap50[class_id]),
                }
            )
    metrics["per_class"] = per_class
    return metrics


def _rare(metrics: dict) -> dict[str, float]:
    per_class = metrics["per_class"]
    recalls = [
        per_class[name][metric]
        for name in RARE_CLASSES
        for metric in ("box_recall", "mask_recall")
    ]
    precisions = [
        per_class[name][metric]
        for name in RARE_CLASSES
        for metric in ("box_precision", "mask_precision")
    ]
    return {
        "macro_rare_recall": sum(recalls) / len(recalls),
        "macro_rare_precision": sum(precisions) / len(precisions),
        "minimum_rare_recall": min(recalls),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--recall-tolerance", type=float, default=0.02)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    checkpoints = sorted(args.weights_dir.glob("epoch*.pt"), key=_order)
    if not checkpoints:
        checkpoints = sorted(args.weights_dir.glob("*.pt"), key=_order)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints in {args.weights_dir}")

    baseline = None
    floor = 0.0
    if args.baseline:
        baseline_metrics = _evaluate(config, args.data, args.baseline)
        baseline = {**baseline_metrics, **_rare(baseline_metrics)}
        floor = max(0.0, baseline["macro_rare_recall"] - args.recall_tolerance)

    evaluations = []
    for checkpoint in checkpoints:
        metrics = _evaluate(config, args.data, checkpoint)
        row = {
            "checkpoint": checkpoint.resolve().relative_to(ROOT).as_posix(),
            **metrics,
            **_rare(metrics),
        }
        row["recall_guard_passed"] = row["macro_rare_recall"] >= floor
        evaluations.append(row)
    eligible = [row for row in evaluations if row["recall_guard_passed"]]
    if not eligible:
        raise RuntimeError(f"No checkpoint preserved macro rare recall >= {floor:.4f}")
    if baseline:
        selected = max(
            eligible, key=lambda row: (row["macro_rare_precision"], row["macro_rare_recall"])
        )
    else:
        selected = max(
            eligible, key=lambda row: (row["macro_rare_recall"], row["macro_rare_precision"])
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    shutil.copy2(selected["checkpoint"], temporary)
    temporary.replace(args.output)
    report = {
        "rare_classes": list(RARE_CLASSES),
        "recall_floor": floor,
        "baseline": baseline,
        "evaluations": evaluations,
        "selected": selected,
        "output": args.output.resolve().relative_to(ROOT).as_posix(),
        "output_sha256": _sha256(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
