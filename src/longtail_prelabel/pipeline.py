from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import yaml
from PIL import Image
from tqdm import tqdm

from .core import (
    Candidate,
    apply_candidate_budget,
    auto_verify_candidates,
    box_iou,
    filter_candidate_geometry,
    merge_candidates,
    read_jsonl,
    shard_rows,
    should_promote_metrics,
    write_jsonl,
)
from .models import GroundingDinoProposer, QwenVerifier, Sam2Refiner, YoloProposer
from .taxonomy import (
    class_for_label,
    class_remap,
    dataset_names,
    remap_yolo_label,
    target_classes,
    target_names,
    target_prompts,
)


def propose(
    config: dict[str, Any],
    input_path: str,
    output_path: str,
    device: str,
    shard_index: int,
    num_shards: int,
) -> None:
    inf, target = config["inference"], config["target"]
    classes = target_classes(config)
    use_student = bool(inf.get("use_student", False))
    student_class_ids = target.get("student_class_ids") or (
        [item.id for item in classes] if target.get("classes") else []
    )
    if use_student and not student_class_ids:
        raise ValueError("use_student=true requires non-empty target.student_class_ids")
    student = None
    if use_student:
        student = YoloProposer(
            config["models"]["student"],
            device,
            inf["student_confidence"],
            inf["student_iou"],
            int(inf.get("student_image_size", 1280)),
            int(inf.get("student_tile_grid", 1)),
            float(inf.get("student_tile_overlap", 0.2)),
        )
    grounding = GroundingDinoProposer(
        config["models"]["grounding_dino"],
        device,
        inf["grounding_box_threshold"],
        inf["grounding_text_threshold"],
        int(inf.get("grounding_tile_grid", 1)),
        float(inf.get("grounding_tile_overlap", 0.2)),
        (
            float(inf["grounding_tile_box_threshold"])
            if "grounding_tile_box_threshold" in inf
            else None
        ),
    )
    source_rows = read_jsonl(input_path)
    if inf.get("keyframes_only", False):
        source_rows = (row for row in source_rows if row.get("is_key_frame", True))
    rows = shard_rows(source_rows, shard_index, num_shards)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in tqdm(rows, desc=f"propose[{shard_index}/{num_shards}]"):
            candidates = []
            if student is not None:
                candidates.extend(student.predict(row["image_path"], student_class_ids))
            candidates += grounding.predict(row["image_path"], target_prompts(config))
            for candidate in candidates:
                matched_class = class_for_label(candidate.label, classes)
                if matched_class is None:
                    candidate.decision = "uncertain"
                    candidate.semantic_confidence = 0.0
                    candidate.reason = "unknown_target_class"
                    continue
                candidate.class_id = matched_class.id
                candidate.label = matched_class.name
            raw_count = len(candidates)
            with Image.open(row["image_path"]) as image:
                candidates, geometry_rejected = filter_candidate_geometry(
                    candidates,
                    image.width,
                    image.height,
                    inf["candidate_max_area_ratio"],
                )
            merged = merge_candidates(candidates, inf["merge_iou"])
            active, deferred = apply_candidate_budget(merged, inf["max_candidates_per_image"])
            row["candidates"] = [candidate.to_dict() for candidate in active + deferred]
            row["proposal_stats"] = {
                "raw": raw_count,
                "geometry_rejected": len(geometry_rejected),
                "merged": len(merged),
                "auto_verify": len(active),
                "deferred_review": len(deferred),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def verify(
    config: dict[str, Any],
    input_path: str,
    output_path: str,
    device: str,
    shard_index: int,
    num_shards: int,
    resume: bool = False,
) -> None:
    inf = config["inference"]
    classes = target_classes(config)
    teacher = QwenVerifier(
        config["models"]["semantic_teacher"],
        device,
        inf["max_new_tokens"],
        inf.get("qwen_gpu_memory", "13GiB"),
        inf.get("qwen_cpu_memory", "48GiB"),
        inf.get("qwen_offload_folder", "workspace/qwen_offload"),
    )
    rows = shard_rows(read_jsonl(input_path), shard_index, num_shards)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed: set[str] = set()
    if resume and output.exists():
        completed = {
            str(row.get("sample_data_token", row.get("image_path"))) for row in read_jsonl(output)
        }
    mode = "a" if resume and output.exists() else "w"
    processed_decisions = 0
    raw_targets = 0
    with output.open(mode, encoding="utf-8") as handle:
        for row in tqdm(rows, desc=f"verify[{shard_index}/{num_shards}]"):
            row_key = str(row.get("sample_data_token", row.get("image_path")))
            if row_key in completed:
                continue
            candidates = [Candidate.from_dict(raw) for raw in row.get("candidates", [])]
            for candidate in candidates:
                if candidate.class_id is None:
                    matched_class = class_for_label(candidate.label, classes)
                    if matched_class is not None:
                        candidate.class_id = matched_class.id
                        candidate.label = matched_class.name
                if candidate.class_id is None:
                    candidate.decision = "uncertain"
                    candidate.semantic_confidence = 0.0
                    candidate.reason = "unknown_target_class"
                    continue
                class_spec = classes[candidate.class_id]
                hard_negative_labels = {
                    label.lower().strip() for label in class_spec.negative_examples
                }
                normalized_label = (
                    candidate.label.lower().replace(".", " ").replace("_", " ").strip()
                )
                if any(
                    normalized_label == negative or normalized_label.endswith(" " + negative)
                    for negative in hard_negative_labels
                ):
                    candidate.decision = "not_target"
                    candidate.semantic_confidence = 1.0
                    candidate.reason = "detector_hard_negative"
            batch_size = inf["semantic_batch_size"]
            qwen_responses = []
            pending = [candidate for candidate in candidates if candidate.decision is None]
            for class_spec in classes:
                class_pending = [
                    candidate for candidate in pending if candidate.class_id == class_spec.id
                ]
                for start in range(0, len(class_pending), batch_size):
                    batch = class_pending[start : start + batch_size]
                    decisions = teacher.verify_many(
                        row["image_path"],
                        batch,
                        class_spec.name,
                        class_spec.description,
                        list(class_spec.negative_examples),
                        inf["crop_expansion"],
                        inf["semantic_crop_size"],
                    )
                    qwen_responses.append(
                        {"class": class_spec.name, "response": teacher.last_response[:2000]}
                    )
                    for candidate_id, candidate in enumerate(batch, start=1):
                        decision, confidence, reason = decisions[candidate_id]
                        candidate.decision = decision
                        candidate.semantic_confidence = confidence
                        candidate.reason = reason

            row_target_count = sum(candidate.decision == "target" for candidate in pending)
            raw_targets += row_target_count
            processed_decisions += len(pending)
            if row_target_count > inf["max_targets_per_image"]:
                for candidate in pending:
                    if candidate.decision == "target":
                        candidate.decision = "uncertain"
                        candidate.semantic_confidence = 0.0
                        candidate.reason = "safety_max_targets_per_image"

            row["candidates"] = [candidate.to_dict() for candidate in candidates]
            row["verification_stats"] = {
                "processed": len(pending),
                "raw_targets": row_target_count,
                "final_targets": sum(candidate.decision == "target" for candidate in candidates),
            }
            row["qwen_responses"] = qwen_responses
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()

            if (
                processed_decisions >= inf["safety_warmup_candidates"]
                and raw_targets / processed_decisions > inf["max_target_acceptance_rate"]
            ):
                rate = raw_targets / processed_decisions
                raise RuntimeError(
                    f"Qwen safety gate triggered: target acceptance rate {rate:.1%} exceeds "
                    f"{inf['max_target_acceptance_rate']:.1%}. Review prompt and candidates."
                )


def auto_verify(
    config: dict[str, Any],
    input_path: str,
    output_path: str,
    shard_index: int,
    num_shards: int,
) -> None:
    """Semantic gate that never calls Qwen and never creates automatic negatives."""
    inf = config["inference"]
    rows = shard_rows(read_jsonl(input_path), shard_index, num_shards)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in tqdm(rows, desc=f"auto-verify[{shard_index}/{num_shards}]"):
            candidates = [Candidate.from_dict(item) for item in row.get("candidates", [])]
            stats = auto_verify_candidates(
                candidates,
                list(inf.get("auto_required_sources", ["student", "grounding_dino"])),
                float(inf.get("auto_min_proposal_score", 0.15)),
                int(inf.get("auto_max_targets_per_image", 12)),
                (
                    float(inf["auto_teacher_only_min_score"])
                    if inf.get("auto_teacher_only_min_score") is not None
                    else None
                ),
            )
            row["candidates"] = [candidate.to_dict() for candidate in candidates]
            row["automatic_verification_stats"] = stats
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def segment(
    config: dict[str, Any],
    input_path: str,
    output_path: str,
    device: str,
    shard_index: int,
    num_shards: int,
) -> None:
    inf = config["inference"]
    teacher = Sam2Refiner(
        config["models"]["segment_teacher"], device, inf.get("sam_mask_threshold", 0.0)
    )
    rows = shard_rows(read_jsonl(input_path), shard_index, num_shards)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in tqdm(rows, desc=f"segment[{shard_index}/{num_shards}]"):
            candidates = [Candidate.from_dict(item) for item in row.get("candidates", [])]
            selected = [
                candidate
                for candidate in candidates
                if candidate.decision == "target"
                and (candidate.semantic_confidence or 0.0) >= inf["semantic_accept"]
            ]
            if selected:
                refinements = teacher.refine(row["image_path"], [item.bbox for item in selected])
                for candidate, refined in zip(selected, refinements):
                    candidate.proposal_bbox = list(candidate.bbox)
                    candidate.bbox = refined["bbox"]
                    candidate.mask_score = refined["score"]
                    candidate.segmentation = refined["segmentation"]
            selected_ids = {id(item) for item in selected}
            row["accepted"] = [item.to_dict() for item in selected]
            row["review"] = [
                item.to_dict()
                for item in candidates
                if id(item) not in selected_ids and item.decision != "not_target"
            ]
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def auto_select_masks(
    config: dict[str, Any],
    input_path: str,
    output_path: str,
    shard_index: int,
    num_shards: int,
) -> None:
    """Keep high-quality SAM masks from detector-agreed proposals without human review."""
    inf = config["inference"]
    mask_score = float(inf.get("auto_min_mask_score", 0.75))
    proposal_iou = float(inf.get("auto_min_proposal_mask_iou", 0.30))
    dedup_iou = float(inf.get("auto_mask_dedup_iou", 0.70))
    rows = shard_rows(read_jsonl(input_path), shard_index, num_shards)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in tqdm(rows, desc=f"auto-select[{shard_index}/{num_shards}]"):
            source = [Candidate.from_dict(item) for item in row.get("accepted", [])]
            ranked = sorted(
                source, key=lambda item: (item.mask_score or 0.0, item.score), reverse=True
            )
            accepted: list[Candidate] = []
            rejected: list[Candidate] = []
            for candidate in ranked:
                valid_polygon = len(candidate.segmentation or []) >= 6
                aligned = (
                    candidate.proposal_bbox is not None
                    and box_iou(candidate.proposal_bbox, candidate.bbox) >= proposal_iou
                )
                duplicate = any(
                    candidate.class_id == other.class_id
                    and box_iou(candidate.bbox, other.bbox) >= dedup_iou
                    for other in accepted
                )
                if (
                    valid_polygon
                    and (candidate.mask_score or 0.0) >= mask_score
                    and aligned
                    and not duplicate
                ):
                    candidate.decision = "target"
                    candidate.semantic_confidence = 1.0
                    candidate.reason = "automatic_agreement_sam"
                    accepted.append(candidate)
                else:
                    candidate.decision = "uncertain"
                    candidate.semantic_confidence = 0.0
                    if duplicate:
                        candidate.reason = "automatic_mask_duplicate"
                    elif not aligned:
                        candidate.reason = "automatic_mask_box_mismatch"
                    else:
                        candidate.reason = "automatic_mask_quality"
                    rejected.append(candidate)
            existing_review = [Candidate.from_dict(item) for item in row.get("review", [])]
            row["accepted"] = [item.to_dict() for item in accepted]
            row["review"] = [item.to_dict() for item in existing_review + rejected]
            row["automatic_mask_stats"] = {
                "input": len(source),
                "accepted": len(accepted),
                "uncertain": len(existing_review) + len(rejected),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def export_pseudo(config: dict[str, Any], input_path: str, output_dir: str) -> dict[str, Any]:
    target_dir = Path(output_dir).resolve()
    build_dir = target_dir.with_name(f".{target_dir.name}.building")
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    mode = config["dataset"].get("image_mode", "copy")
    accepted_count = 0
    class_counts = {name: 0 for name in target_names(config).values()}
    classes = target_classes(config)
    image_count = 0
    review_rows = []
    for row in tqdm(read_jsonl(input_path), desc="export"):
        accepted = [Candidate.from_dict(item) for item in row.get("accepted", [])]
        review_rows.extend(
            {"image_path": row["image_path"], "candidate": item} for item in row.get("review", [])
        )
        valid_accepted = []
        for candidate in accepted:
            matched_class = (
                classes[candidate.class_id]
                if candidate.class_id is not None and 0 <= candidate.class_id < len(classes)
                else class_for_label(candidate.label, classes)
            )
            if matched_class is None:
                candidate.decision = "uncertain"
                candidate.reason = "unknown_target_class"
                review_rows.append(
                    {"image_path": row["image_path"], "candidate": candidate.to_dict()}
                )
                continue
            candidate.class_id = matched_class.id
            candidate.label = matched_class.name
            valid_accepted.append(candidate)
        accepted = valid_accepted
        if not accepted:
            continue
        source = Path(row["image_path"])
        stem = row.get("sample_data_token", source.stem)
        image_target = build_dir / "images" / "train" / f"{stem}{source.suffix.lower()}"
        label_target = build_dir / "labels" / "train" / f"{stem}.txt"
        _place_image(source, image_target, mode)
        with Image.open(source) as image:
            width, height = image.size
        lines = []
        for candidate in accepted:
            polygon = candidate.segmentation or []
            if len(polygon) < 6:
                x1, y1, x2, y2 = candidate.bbox
                polygon = [x1, y1, x2, y1, x2, y2, x1, y2]
            values = [
                value / (width if index % 2 == 0 else height) for index, value in enumerate(polygon)
            ]
            lines.append(
                f"{candidate.class_id} " + " ".join(f"{value:.6f}" for value in values)
            )
            accepted_count += 1
            class_counts[classes[candidate.class_id].name] += 1
        label_target.parent.mkdir(parents=True, exist_ok=True)
        label_target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        image_count += 1

    data = {
        "path": str(target_dir),
        "train": "images/train",
        "val": "images/train",
        "names": target_names(config),
    }
    with (build_dir / "data.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)
    write_jsonl(build_dir / "review.jsonl", review_rows)
    summary = {
        "images": image_count,
        "accepted": accepted_count,
        "review": len(review_rows),
        "class_instances": class_counts,
    }
    (build_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if target_dir.exists():
        shutil.rmtree(target_dir)
    build_dir.replace(target_dir)
    return summary


def _merge_remapped_label(
    source: Path | None,
    destination: Path,
    remap: dict[int, int],
) -> tuple[int, int]:
    lines, dropped = remap_yolo_label(
        source.read_text(encoding="utf-8") if source and source.exists() else "", remap
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = destination.read_text(encoding="utf-8").splitlines() if destination.exists() else []
    merged = list(dict.fromkeys([*existing, *lines]))
    destination.write_text(("\n".join(merged) + "\n") if merged else "", encoding="utf-8")
    return len(lines), dropped


def _add_dataset_split(
    source: Path,
    source_split: str,
    output_split: str,
    build: Path,
    canonical_names: dict[int, str],
    mode: str,
    excluded_tokens: set[str] | None = None,
) -> dict[str, int]:
    source_names = dataset_names(source, canonical_names)
    remap = class_remap(source_names, canonical_names)
    excluded = excluded_tokens or set()
    stats = {"images": 0, "new_images": 0, "instances": 0, "dropped_instances": 0}
    existing_tokens = {
        image.stem for image in (build / "images" / output_split).glob("*") if image.is_file()
    }
    for image in sorted((source / "images" / source_split).glob("*")):
        if not image.is_file() or image.stem in excluded:
            continue
        stats["images"] += 1
        if image.stem not in existing_tokens:
            _place_image(image, build / "images" / output_split / image.name, mode)
            existing_tokens.add(image.stem)
            stats["new_images"] += 1
        label = source / "labels" / source_split / f"{image.stem}.txt"
        kept, dropped = _merge_remapped_label(
            label if label.exists() else None,
            build / "labels" / output_split / f"{image.stem}.txt",
            remap,
        )
        stats["instances"] += kept
        stats["dropped_instances"] += dropped
    return stats


def _class_instance_counts(
    dataset: Path, names: dict[int, str], split: str = "train"
) -> dict[str, int]:
    counts = {name: 0 for name in names.values()}
    for label in (dataset / "labels" / split).glob("*.txt"):
        for line in label.read_text(encoding="utf-8").splitlines():
            if line.strip():
                counts[names[int(line.split()[0])]] += 1
    return counts


def _build_unified_training_dataset(
    config: dict[str, Any], output_dir: str, pseudo_dir: str | None
) -> dict[str, Any]:
    """Build one canonical taxonomy from seed, historical, and optional pseudo datasets."""
    settings = config["self_training"]
    seed = Path(settings["seed_dataset"])
    target = Path(output_dir).resolve()
    build = target.with_name(f".{target.name}.building")
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)
    mode = config["dataset"].get("image_mode", "copy")
    names = target_names(config)

    fixed_val_tokens = {
        image.stem for image in (seed / "images" / "val").glob("*") if image.is_file()
    }
    validation_entries = []
    for entry in settings.get("validation_datasets", []):
        item = {"path": entry} if isinstance(entry, str) else entry
        source = Path(item["path"])
        source_split = str(item.get("split", "val"))
        validation_entries.append((source, source_split))
        fixed_val_tokens.update(
            image.stem
            for image in (source / "images" / source_split).glob("*")
            if image.is_file()
        )

    seed_train = _add_dataset_split(
        seed, "train", "train", build, names, mode, fixed_val_tokens
    )
    seed_val = _add_dataset_split(seed, "val", "val", build, names, mode)
    validation_summaries = []
    for source, source_split in validation_entries:
        stats = _add_dataset_split(source, source_split, "val", build, names, mode)
        validation_summaries.append(
            {"dataset": str(source), "split": source_split, **stats}
        )

    history_summaries = []
    for entry in settings.get("history_datasets", []):
        item = {"path": entry} if isinstance(entry, str) else entry
        source = Path(item["path"])
        source_split = str(item.get("split", "train"))
        stats = _add_dataset_split(
            source,
            source_split,
            "train",
            build,
            names,
            mode,
            fixed_val_tokens,
        )
        history_summaries.append({"dataset": str(source), "split": source_split, **stats})

    added_pseudo: list[str] = []
    pseudo = Path(pseudo_dir).resolve() if pseudo_dir else None
    if pseudo is not None:
        pseudo_names = dataset_names(pseudo, names)
        pseudo_remap = class_remap(pseudo_names, names)
        supervised_train_count = len(list((build / "images" / "train").glob("*")))
        max_ratio = float(settings.get("max_pseudo_to_seed_ratio", 1.0))
        max_pseudo = max(1, int(supervised_train_count * max_ratio))
        minimum = int(settings.get("min_pseudo_images", 1))
        existing_train_tokens = {
            image.stem for image in (build / "images" / "train").glob("*") if image.is_file()
        }
        for label in sorted((pseudo / "labels" / "train").glob("*.txt")):
            remapped_lines, _ = remap_yolo_label(label.read_text(encoding="utf-8"), pseudo_remap)
            if not remapped_lines or label.stem in fixed_val_tokens:
                continue
            images = list((pseudo / "images" / "train").glob(f"{label.stem}.*"))
            if not images:
                continue
            existed = label.stem in existing_train_tokens
            _place_image(images[0], build / "images" / "train" / images[0].name, mode)
            existing_train_tokens.add(label.stem)
            _merge_remapped_label(
                label,
                build / "labels" / "train" / f"{label.stem}.txt",
                pseudo_remap,
            )
            if not existed:
                added_pseudo.append(label.stem)
            if len(added_pseudo) >= max_pseudo:
                break
        if len(added_pseudo) < minimum:
            shutil.rmtree(build)
            raise RuntimeError(
                f"Automatic update stopped: {len(added_pseudo)} non-overlapping pseudo images, "
                f"need at least {minimum}."
            )

    data = {
        "path": str(target),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    with (build / "data.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)
    summary = {
        "classes": names,
        "seed_dataset": str(seed),
        "pseudo_dataset": str(pseudo) if pseudo else None,
        "seed_train_images": seed_train["images"],
        "fixed_val_images": seed_val["images"],
        "validation_datasets": validation_summaries,
        "history_datasets": history_summaries,
        "train_images": len(list((build / "images" / "train").glob("*"))),
        "validation_images": len(list((build / "images" / "val").glob("*"))),
        "pseudo_images": len(added_pseudo),
        "pseudo_tokens": added_pseudo,
        "class_instances": _class_instance_counts(build, names, "train"),
        "validation_class_instances": _class_instance_counts(build, names, "val"),
    }
    (build / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if target.exists():
        shutil.rmtree(target)
    build.replace(target)
    return summary


def build_joint_training_dataset(config: dict[str, Any], output_dir: str) -> dict[str, Any]:
    """Combine seed and historical datasets under one canonical multi-class data.yaml."""
    return _build_unified_training_dataset(config, output_dir, None)


def build_self_training_dataset(
    config: dict[str, Any], pseudo_dir: str, output_dir: str
) -> dict[str, Any]:
    """Add conservative pseudo labels to the unified historical training dataset."""
    return _build_unified_training_dataset(config, output_dir, pseudo_dir)


def bootstrap_student(config: dict[str, Any]) -> dict[str, Any]:
    settings = config["self_training"]
    initial = Path(settings["initial_student"])
    current = Path(settings["current_model"])
    if current.exists():
        return {"created": False, "current_model": str(current)}
    if not initial.exists():
        raise FileNotFoundError(f"Initial student checkpoint does not exist: {initial}")
    current.parent.mkdir(parents=True, exist_ok=True)
    temporary = current.with_suffix(current.suffix + ".tmp")
    shutil.copy2(initial, temporary)
    temporary.replace(current)
    return {"created": True, "current_model": str(current), "initial_student": str(initial)}


def evaluate_student(config: dict[str, Any], model_path: str | Path) -> dict[str, Any]:
    os.environ.setdefault("YOLO_OFFLINE", "true")
    from ultralytics import YOLO

    train = config["training"]
    settings = config["self_training"]
    validation_dataset = Path(
        settings.get("validation_dataset", settings.get("train_dataset", settings["seed_dataset"]))
    )
    if not (validation_dataset / "data.yaml").exists():
        validation_dataset = Path(settings["seed_dataset"])
    data_yaml = validation_dataset / "data.yaml"
    result = YOLO(str(model_path)).val(
        data=str(data_yaml),
        split="val",
        imgsz=train["image_size"],
        batch=train["batch"],
        device=train["device"],
        workers=train["workers"],
        plots=False,
        verbose=False,
    )
    values = result.results_dict
    metrics: dict[str, Any] = {
        "box_precision": float(values.get("metrics/precision(B)", 0.0)),
        "box_recall": float(values.get("metrics/recall(B)", 0.0)),
        "box_map50": float(values.get("metrics/mAP50(B)", 0.0)),
        "mask_precision": float(values.get("metrics/precision(M)", 0.0)),
        "mask_recall": float(values.get("metrics/recall(M)", 0.0)),
        "mask_map50": float(values.get("metrics/mAP50(M)", 0.0)),
    }
    result_names = result.names if isinstance(result.names, dict) else dict(enumerate(result.names))
    per_class: dict[str, dict[str, float]] = {
        str(name): {} for name in result_names.values()
    }
    for prefix, metric in (("box", getattr(result, "box", None)), ("mask", getattr(result, "seg", None))):
        if metric is None:
            continue
        for class_id, name in result_names.items():
            if int(class_id) >= len(metric.p):
                continue
            per_class[str(name)].update(
                {
                    f"{prefix}_precision": float(metric.p[int(class_id)]),
                    f"{prefix}_recall": float(metric.r[int(class_id)]),
                    f"{prefix}_map50": float(metric.ap50[int(class_id)]),
                }
            )
    metrics["per_class"] = per_class
    return metrics


def promote_student(config: dict[str, Any]) -> dict[str, Any]:
    settings = config["self_training"]
    current = Path(settings["current_model"])
    result_path = Path(settings["training_result"])
    if not current.exists():
        raise FileNotFoundError(f"Current student checkpoint does not exist: {current}")
    if not result_path.exists():
        raise FileNotFoundError(f"Training result does not exist: {result_path}")
    training_result = json.loads(result_path.read_text(encoding="utf-8"))
    candidate = Path(training_result["best_checkpoint"])
    if not candidate.exists():
        raise FileNotFoundError(f"Candidate checkpoint does not exist: {candidate}")

    baseline_metrics = evaluate_student(config, current)
    candidate_metrics = evaluate_student(config, candidate)
    promoted, reasons = should_promote_metrics(
        baseline_metrics, candidate_metrics, settings.get("promotion", {})
    )
    if promoted:
        temporary = current.with_suffix(current.suffix + ".tmp")
        shutil.copy2(candidate, temporary)
        temporary.replace(current)
    report = {
        "promoted": promoted,
        "reasons": reasons,
        "baseline_checkpoint": str(current),
        "candidate_checkpoint": str(candidate),
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
    }
    report_path = Path(settings["promotion_report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def train_student(config: dict[str, Any], data_yaml: str) -> dict[str, Any]:
    os.environ.setdefault("YOLO_OFFLINE", "true")
    workspace = Path(config["dataset"]["workspace"])
    train = config["training"]
    monitoring = config.get("monitoring", {}).get("wandb", {})
    wandb_enabled = bool(monitoring.get("enabled", False))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(workspace / "ultralytics-config"))
    os.environ.setdefault("MPLCONFIGDIR", str(workspace / "matplotlib-config"))

    wandb_project = str(monitoring.get("project", "longtail-prelabel"))
    if wandb_enabled:
        configured_mode = str(monitoring.get("mode", "auto")).lower()
        if configured_mode == "auto":
            configured_mode = os.environ.get("WANDB_MODE") or (
                "online" if os.environ.get("WANDB_API_KEY") else "offline"
            )
        if configured_mode not in {"online", "offline", "disabled"}:
            raise ValueError("monitoring.wandb.mode must be auto, online, offline, or disabled")

        wandb_root = (workspace / "wandb").resolve()
        wandb_paths = {
            "WANDB_DIR": wandb_root / "runs",
            "WANDB_CACHE_DIR": wandb_root / "cache",
            "WANDB_CONFIG_DIR": wandb_root / "config",
            "WANDB_DATA_DIR": wandb_root / "data",
            "WANDB_ARTIFACT_DIR": wandb_root / "artifacts",
        }
        temp_dir = wandb_root / "tmp"
        for path in [*wandb_paths.values(), temp_dir]:
            path.mkdir(parents=True, exist_ok=True)
        for key, path in wandb_paths.items():
            os.environ[key] = str(path)
        for key in ("TEMP", "TMP", "TMPDIR"):
            os.environ[key] = str(temp_dir)
        os.environ["WANDB_MODE"] = configured_mode
        os.environ["WANDB_DISABLE_GIT"] = "true"
        if monitoring.get("entity"):
            os.environ["WANDB_ENTITY"] = str(monitoring["entity"])
        if monitoring.get("group"):
            os.environ["WANDB_RUN_GROUP"] = str(monitoring["group"])
        if monitoring.get("tags"):
            os.environ["WANDB_TAGS"] = ",".join(str(tag) for tag in monitoring["tags"])

    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS

    SETTINGS["wandb"] = wandb_enabled
    SETTINGS["runs_dir"] = str((workspace / "runs").resolve())

    target_name = config["target"]["name"].rsplit(".", 1)[-1]
    dataset_name = Path(data_yaml).resolve().parent.name
    run_name = str(monitoring.get("run_name") or f"student_{target_name}_{dataset_name}")
    model = YOLO(config["models"]["student"])
    training_args: dict[str, Any] = dict(
        data=data_yaml,
        imgsz=train["image_size"],
        batch=train["batch"],
        epochs=train["epochs"],
        patience=train["patience"],
        device=train["device"],
        workers=train["workers"],
        amp=True,
        cache=False,
        plots=bool(monitoring.get("log_plots", False)),
        project=wandb_project,
        name=run_name,
    )
    for key in (
        "optimizer",
        "lr0",
        "lrf",
        "freeze",
        "mosaic",
        "close_mosaic",
        "warmup_epochs",
        "warmup_momentum",
        "warmup_bias_lr",
        "weight_decay",
        "seed",
        "box",
        "cls",
        "dfl",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "flipud",
        "fliplr",
        "mixup",
        "copy_paste",
        "copy_paste_mode",
        "save_period",
    ):
        if key in train:
            training_args[key] = train[key]
    model.train(**training_args)
    best_checkpoint = Path(model.trainer.best).resolve()
    result = {"best_checkpoint": str(best_checkpoint), "save_dir": str(model.trainer.save_dir)}
    result_path = config.get("self_training", {}).get("training_result")
    if result_path:
        target = Path(result_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
