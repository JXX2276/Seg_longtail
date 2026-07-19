from __future__ import annotations

import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "huggingface"
MODELS = ROOT / "models"

# Set this before importing huggingface_hub so no cache escapes the project.
os.environ.setdefault("HF_HOME", str(CACHE))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(CACHE / "hub"))


SNAPSHOTS = {
    "grounding-dino-tiny": (
        "IDEA-Research/grounding-dino-tiny",
        ["*.json", "*.txt", "*.safetensors"],
    ),
    "Qwen2.5-VL-7B-Instruct": (
        "Qwen/Qwen2.5-VL-7B-Instruct",
        ["*.json", "*.txt", "*.jinja", "*.model", "*.safetensors"],
    ),
    "sam2.1-hiera-base-plus": (
        "facebook/sam2.1-hiera-base-plus",
        ["*.json", "*.yaml", "*.safetensors"],
    ),
}


def main() -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    MODELS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    for local_name, (repo_id, patterns) in SNAPSHOTS.items():
        target = MODELS / local_name
        print(f"Downloading {repo_id} -> {target.relative_to(ROOT)}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=target,
            allow_patterns=patterns,
            cache_dir=CACHE / "hub",
            max_workers=1,
        )

    print("Downloading Ultralytics/YOLO11 yolo11s-seg.pt")
    cached = hf_hub_download(
        repo_id="Ultralytics/YOLO11",
        filename="yolo11s-seg.pt",
        cache_dir=CACHE / "hub",
    )
    shutil.copy2(cached, MODELS / "yolo11s-seg.pt")
    print("All four pretrained models are ready in models/.")


if __name__ == "__main__":
    main()
