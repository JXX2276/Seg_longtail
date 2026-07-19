from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
DATASET = ROOT / "workspace" / "dataset"
PIPELINE = ROOT / "workspace" / "pipeline"
LOG = PIPELINE / "pipeline.log"


def _log(message: str) -> None:
    PIPELINE.mkdir(parents=True, exist_ok=True)
    line = f"{dt.datetime.now().astimezone().isoformat()} {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _run(*arguments: str | Path) -> None:
    command = [str(PYTHON), *(str(item) for item in arguments)]
    _log("RUN python " + " ".join(str(item) for item in arguments))
    subprocess.run(command, cwd=ROOT, check=True)


def _copy_best(result_path: Path, output: Path) -> None:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    source = Path(result["best_checkpoint"])
    if not source.exists():
        raise FileNotFoundError(f"Best checkpoint does not exist: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(output)
    _log(f"SELECT {source} -> {output.relative_to(ROOT)}")


def _weights_dir(result_path: Path) -> Path:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    return Path(result["save_dir"]) / "weights"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1 (10 epochs) -> Stage 2 balanced -> Stage 3 rare recall"
    )
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()

    data_yaml = DATASET / "data.yaml"
    pretrained = ROOT / "models" / "yolo11s-seg.pt"
    if not data_yaml.exists():
        raise FileNotFoundError(
            "workspace/dataset/data.yaml is missing; run scripts/run_teacher_500.ps1 first"
        )
    if not args.build_only and not pretrained.exists():
        raise FileNotFoundError(
            "models/yolo11s-seg.pt is missing; run scripts/download_models.py first"
        )

    _log("Build deterministic Stage-2 and Stage-3 sampling schedules")
    _run(
        "scripts/build_balanced_train_list.py",
        "--dataset",
        DATASET.relative_to(ROOT),
        "--target-per-rare-class",
        "800",
        "--max-repeat",
        "8",
        "--dominant-images",
        "1200",
        "--background-images",
        "1200",
        "--seed",
        "339",
        "--output-prefix",
        "stage2_balanced",
    )
    _run(
        "scripts/build_balanced_train_list.py",
        "--dataset",
        DATASET.relative_to(ROOT),
        "--target-per-rare-class",
        "3000",
        "--max-repeat",
        "48",
        "--dominant-images",
        "400",
        "--background-images",
        "600",
        "--seed",
        "341",
        "--output-prefix",
        "stage3_rare",
    )
    if args.build_only:
        _log("Build-only completed; no training started")
        return

    stage1_result = PIPELINE / "stage1" / "training_result.json"
    _log("Stage 1 start: downloaded pretrained weights, exactly 10 epochs")
    _run(
        "scripts/train_yolo.py",
        "--config",
        "configs/stage1_pretrained_epoch10.yaml",
        "--data",
        data_yaml.relative_to(ROOT),
        "--result",
        stage1_result.relative_to(ROOT),
    )
    stage1_selected = PIPELINE / "stage1_selected.pt"
    _copy_best(stage1_result, stage1_selected)

    stage2_result = PIPELINE / "stage2" / "training_result.json"
    _log("Stage 2 start: balanced sampling with rare-recall guard")
    _run(
        "scripts/train_yolo.py",
        "--config",
        "configs/stage2_balanced.yaml",
        "--data",
        (DATASET / "stage2_balanced_data.yaml").relative_to(ROOT),
        "--result",
        stage2_result.relative_to(ROOT),
    )
    stage2_selected = PIPELINE / "stage2_selected.pt"
    _run(
        "scripts/select_checkpoint.py",
        "--config",
        "configs/stage2_balanced.yaml",
        "--data",
        data_yaml.relative_to(ROOT),
        "--weights-dir",
        _weights_dir(stage2_result),
        "--baseline",
        stage1_selected.relative_to(ROOT),
        "--recall-tolerance",
        "0.02",
        "--output",
        stage2_selected.relative_to(ROOT),
        "--report",
        (PIPELINE / "stage2" / "selection_report.json").relative_to(ROOT),
    )

    stage3_result = PIPELINE / "stage3" / "training_result.json"
    _log("Stage 3 start: aggressive rare-class recall training")
    _run(
        "scripts/train_yolo.py",
        "--config",
        "configs/stage3_rare_recall.yaml",
        "--data",
        (DATASET / "stage3_rare_data.yaml").relative_to(ROOT),
        "--result",
        stage3_result.relative_to(ROOT),
    )
    _run(
        "scripts/select_checkpoint.py",
        "--config",
        "configs/stage3_rare_recall.yaml",
        "--data",
        data_yaml.relative_to(ROOT),
        "--weights-dir",
        _weights_dir(stage3_result),
        "--output",
        (PIPELINE / "final_selected.pt").relative_to(ROOT),
        "--report",
        (PIPELINE / "stage3" / "selection_report.json").relative_to(ROOT),
    )
    _log("Pipeline completed: workspace/pipeline/final_selected.pt")


if __name__ == "__main__":
    main()
