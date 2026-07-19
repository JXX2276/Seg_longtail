from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


@dataclass
class Candidate:
    bbox: list[float]
    score: float
    source: str
    label: str = ""
    class_id: int | None = None
    sources: list[str] = field(default_factory=list)
    decision: str | None = None
    semantic_confidence: float | None = None
    reason: str = ""
    mask_score: float | None = None
    segmentation: list[float] | None = None
    view: str = ""
    proposal_bbox: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if not data["sources"]:
            data["sources"] = [self.source]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Candidate":
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in data.items() if key in fields})


def box_iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def box_area(box: list[float]) -> float:
    """Return the non-negative area of an xyxy box."""
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def filter_candidate_geometry(
    candidates: list[Candidate],
    width: int,
    height: int,
    max_area_ratio: float,
) -> tuple[list[Candidate], list[Candidate]]:
    """Split candidates into usable and rejected boxes using conservative geometry checks."""
    image_area = max(1.0, float(width * height))
    kept: list[Candidate] = []
    rejected: list[Candidate] = []
    for candidate in candidates:
        x1, y1, x2, y2 = candidate.bbox
        valid = x2 > x1 and y2 > y1 and x2 > 0 and y2 > 0 and x1 < width and y1 < height
        too_large = box_area(candidate.bbox) / image_area > max_area_ratio
        if valid and not too_large:
            kept.append(candidate)
        else:
            candidate.decision = "not_target"
            candidate.semantic_confidence = 1.0
            candidate.reason = "geometry_filter"
            rejected.append(candidate)
    return kept, rejected


def apply_candidate_budget(
    candidates: list[Candidate], max_candidates: int
) -> tuple[list[Candidate], list[Candidate]]:
    """Keep the strongest candidates for automatic verification and defer the rest to review."""
    ranked = sorted(
        candidates,
        key=lambda item: (len(set(item.sources or [item.source])), item.score),
        reverse=True,
    )
    active = ranked[:max_candidates]
    deferred = ranked[max_candidates:]
    for candidate in deferred:
        candidate.decision = "uncertain"
        candidate.semantic_confidence = 0.0
        candidate.reason = "candidate_budget"
    return active, deferred


def merge_candidates(candidates: list[Candidate], iou_threshold: float) -> list[Candidate]:
    merged: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        match = next(
            (
                item
                for item in merged
                if _same_candidate_class(item, candidate)
                and box_iou(item.bbox, candidate.bbox) >= iou_threshold
            ),
            None,
        )
        if match is None:
            candidate.sources = sorted(set(candidate.sources or [candidate.source]))
            merged.append(candidate)
            continue
        match.sources = sorted(set(match.sources + candidate.sources + [candidate.source]))
        if candidate.score > match.score:
            match.bbox = candidate.bbox
            match.score = candidate.score
            match.label = candidate.label
            match.source = candidate.source
            match.reason = candidate.reason
            match.view = candidate.view
    return merged


def _same_candidate_class(left: Candidate, right: Candidate) -> bool:
    if left.class_id is not None or right.class_id is not None:
        return left.class_id is not None and left.class_id == right.class_id
    if left.label and right.label:
        return left.label.lower().strip() == right.label.lower().strip()
    return True


def auto_verify_candidates(
    candidates: list[Candidate],
    required_sources: list[str],
    min_score: float,
    max_targets: int,
    teacher_only_min_score: float | None = None,
) -> dict[str, int]:
    """Accept only independent detector agreement; everything else stays uncertain."""
    required = set(required_sources)
    accepted: list[Candidate] = []
    for candidate in candidates:
        if candidate.decision is not None:
            continue
        sources = set(candidate.sources or [candidate.source])
        agreement = required.issubset(sources) and candidate.score >= min_score
        teacher_only = (
            teacher_only_min_score is not None
            and "grounding_dino" in sources
            and candidate.score >= teacher_only_min_score
        )
        if agreement or teacher_only:
            candidate.decision = "target"
            candidate.semantic_confidence = 1.0
            candidate.reason = (
                "automatic_detector_agreement" if agreement else "automatic_high_confidence_teacher"
            )
            accepted.append(candidate)
        else:
            candidate.decision = "uncertain"
            candidate.semantic_confidence = 0.0
            candidate.reason = "automatic_gate_disagreement"

    if len(accepted) > max_targets:
        accepted.sort(key=lambda item: item.score, reverse=True)
        for candidate in accepted[max_targets:]:
            candidate.decision = "uncertain"
            candidate.semantic_confidence = 0.0
            candidate.reason = "automatic_target_budget"

    return {
        "processed": sum(candidate.reason.startswith("automatic_") for candidate in candidates),
        "accepted": sum(candidate.decision == "target" for candidate in candidates),
        "uncertain": sum(candidate.decision == "uncertain" for candidate in candidates),
    }


def should_promote_metrics(
    baseline: dict[str, float],
    candidate: dict[str, float],
    settings: dict[str, float],
) -> tuple[bool, list[str]]:
    """Apply a conservative fixed-validation promotion gate."""
    checks = {
        "box_recall": float(settings.get("max_box_recall_drop", 0.02)),
        "mask_recall": float(settings.get("max_mask_recall_drop", 0.0)),
        "mask_map50": float(settings.get("max_mask_map50_drop", 0.0)),
    }
    reasons = [
        f"{name}_drop"
        for name, tolerance in checks.items()
        if candidate.get(name, 0.0) < baseline.get(name, 0.0) - tolerance
    ]
    for name in (
        "box_precision",
        "box_recall",
        "box_map50",
        "mask_precision",
        "mask_recall",
        "mask_map50",
    ):
        minimum = settings.get(f"min_{name}")
        if minimum is not None and candidate.get(name, 0.0) < float(minimum):
            reasons.append(f"{name}_below_minimum")
    recall_gain = candidate.get("mask_recall", 0.0) - baseline.get("mask_recall", 0.0)
    map_gain = candidate.get("mask_map50", 0.0) - baseline.get("mask_map50", 0.0)
    improved = recall_gain >= float(
        settings.get("min_mask_recall_gain", 0.01)
    ) or map_gain >= float(settings.get("min_mask_map50_gain", 0.01))
    if not improved:
        reasons.append("no_material_mask_gain")
    return not reasons, reasons


def expand_box(box: list[float], width: int, height: int, ratio: float) -> list[int]:
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * ratio, (y2 - y1) * ratio
    return [
        max(0, math.floor(x1 - dx)),
        max(0, math.floor(y1 - dy)),
        min(width, math.ceil(x2 + dx)),
        min(height, math.ceil(y2 + dy)),
    ]


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def shard_rows(rows: Iterable[dict[str, Any]], shard_index: int, num_shards: int):
    for index, row in enumerate(rows):
        if index % num_shards == shard_index:
            yield row
