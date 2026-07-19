from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_raw(path: Path, seen: set[Path]) -> dict[str, Any]:
    if path in seen:
        raise ValueError(f"Circular config inheritance: {path}")
    seen.add(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    parent = config.pop("extends", None)
    if parent:
        base = _load_raw((path.parent / parent).resolve(), seen)
        config = _merge(base, config)
    seen.remove(path)
    return config


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = _load_raw(config_path, set())
    project_root = config_path.parent.parent
    for key in ("root", "metadata_root", "workspace"):
        if key not in config["dataset"]:
            continue
        value = Path(config["dataset"][key])
        if not value.is_absolute():
            config["dataset"][key] = str((project_root / value).resolve())
    if config["dataset"].get("exclude_index"):
        values = config["dataset"]["exclude_index"]
        is_list = isinstance(values, list)
        if not is_list:
            values = [values]
        resolved = []
        for value in values:
            path_value = Path(value)
            if not path_value.is_absolute():
                path_value = (project_root / path_value).resolve()
            resolved.append(str(path_value))
        config["dataset"]["exclude_index"] = resolved if is_list else resolved[0]
    if config["dataset"].get("exclude_datasets"):
        values = config["dataset"]["exclude_datasets"]
        is_list = isinstance(values, list)
        if not is_list:
            values = [values]
        resolved = []
        for value in values:
            path_value = Path(value)
            if not path_value.is_absolute():
                path_value = (project_root / path_value).resolve()
            resolved.append(str(path_value))
        config["dataset"]["exclude_datasets"] = resolved if is_list else resolved[0]
    for key, value in config["models"].items():
        path_value = Path(value)
        if not path_value.is_absolute():
            config["models"][key] = str((project_root / path_value).resolve())
    self_training = config.get("self_training", {})
    for key in (
        "initial_student",
        "current_model",
        "seed_dataset",
        "pseudo_dataset",
        "train_dataset",
        "training_result",
        "promotion_report",
        "validation_dataset",
    ):
        if not self_training.get(key):
            continue
        path_value = Path(self_training[key])
        if not path_value.is_absolute():
            self_training[key] = str((project_root / path_value).resolve())
    for dataset_group in ("history_datasets", "validation_datasets"):
        entries = self_training.get(dataset_group, [])
        resolved_entries = []
        for entry in entries:
            item = {"path": entry} if isinstance(entry, str) else dict(entry)
            path_value = Path(item["path"])
            if not path_value.is_absolute():
                item["path"] = str((project_root / path_value).resolve())
            resolved_entries.append(item)
        if entries:
            self_training[dataset_group] = resolved_entries
    config["project_root"] = str(project_root)
    return config
