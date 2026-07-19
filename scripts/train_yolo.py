from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


TRAINING_KEYS = (
    "optimizer",
    "lr0",
    "lrf",
    "freeze",
    "mosaic",
    "close_mosaic",
    "warmup_epochs",
    "warmup_bias_lr",
    "weight_decay",
    "seed",
    "box",
    "cls",
    "dfl",
    "translate",
    "scale",
    "fliplr",
    "mixup",
    "copy_paste",
    "save_period",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    train = config["training"]
    workspace = ROOT / config["workspace"]
    workspace.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_OFFLINE", "true")
    os.environ.setdefault("WANDB_PROJECT", str(config.get("wandb_project", "Seg_longtail")))
    os.environ.setdefault("WANDB_RUN_GROUP", str(config.get("wandb_group", "pipeline")))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(workspace / "ultralytics-config"))
    os.environ.setdefault("MPLCONFIGDIR", str(workspace / "matplotlib-config"))
    wandb_root = workspace / "wandb"
    for name, path in {
        "WANDB_DIR": wandb_root / "runs",
        "WANDB_CACHE_DIR": wandb_root / "cache",
        "WANDB_CONFIG_DIR": wandb_root / "config",
        "WANDB_DATA_DIR": wandb_root / "data",
        "WANDB_ARTIFACT_DIR": wandb_root / "artifacts",
    }.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(name, str(path.resolve()))

    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS

    SETTINGS["wandb"] = True
    training_args = {
        "data": args.data.resolve().relative_to(ROOT).as_posix(),
        "imgsz": train["image_size"],
        "batch": train["batch"],
        "epochs": train["epochs"],
        "patience": train["patience"],
        "device": train["device"],
        "workers": train["workers"],
        "amp": True,
        "cache": False,
        "plots": False,
        "project": (workspace / "runs").relative_to(ROOT).as_posix(),
        "name": config["run_name"],
    }
    for key in TRAINING_KEYS:
        if key in train:
            training_args[key] = train[key]
    model = YOLO(config["model"])
    model.train(**training_args)
    result = {
        "config": args.config.resolve().relative_to(ROOT).as_posix(),
        "data": args.data.resolve().relative_to(ROOT).as_posix(),
        "best_checkpoint": Path(model.trainer.best).resolve().relative_to(ROOT).as_posix(),
        "last_checkpoint": Path(model.trainer.last).resolve().relative_to(ROOT).as_posix(),
        "save_dir": Path(model.trainer.save_dir).resolve().relative_to(ROOT).as_posix(),
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
