"""Detector-consistency guard for safety-oriented restoration selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from saferestore.metrics.detection import box_iou

DEFAULT_PERCEPTION_THRESHOLD = 0.90
ADAPTIVE_GUARD_MAX_INTERVAL = 3
ADAPTIVE_GUARD_SAFETY_MARGIN = 0.05


@dataclass(frozen=True)
class DetectionRecord:
    """Minimal detector output needed by the perception guard."""

    class_id: int
    confidence: float
    box: np.ndarray


@dataclass(frozen=True)
class PerceptionDecision:
    """Auditable decision between the restored candidate and input frame."""

    accept_candidate: bool
    safety_score: float
    reference_retention: float
    matched_reference: int
    reference_count: int
    unmatched_candidate: int
    reason: str


def schedule_perception_guard(
    frame_index: int,
    frames_since_evaluation: int,
    last_safety_score: float,
    last_selection: str,
    scene_cut: bool,
    *,
    threshold: float = DEFAULT_PERCEPTION_THRESHOLD,
    safety_margin: float = ADAPTIVE_GUARD_SAFETY_MARGIN,
    max_interval: int = ADAPTIVE_GUARD_MAX_INTERVAL,
) -> tuple[bool, str]:
    """Schedule detector-consistency evaluation from observable runtime risk.

    The scheduler never waits after a scene cut or a risky/non-temporal
    selection. It only reuses a decision when the temporal candidate had a
    clear safety margin, and even then forces a refresh at ``max_interval``.
    """
    if frame_index <= 0:
        return True, "initial_frame"
    if scene_cut:
        return True, "scene_cut"
    if last_selection != "temporal_candidate":
        return True, "candidate_recovery"
    if last_safety_score < threshold + safety_margin:
        return True, "low_safety_margin"
    if frames_since_evaluation >= max(1, int(max_interval)):
        return True, "periodic_refresh"
    return False, "high_margin_reuse"


def decide_perception_safe(
    reference: list[DetectionRecord],
    candidate: list[DetectionRecord],
    iou_threshold: float = 0.30,
    minimum_score: float = DEFAULT_PERCEPTION_THRESHOLD,
) -> PerceptionDecision:
    """Use frozen-detector consistency as a conservative restoration guard.

    Input detections are treated as references rather than ground truth. A
    candidate is rejected only when it loses or strongly shifts sufficiently
    confident reference objects. New candidate detections are reported but
    lightly penalized because adverse-weather input can hide genuine objects.
    """
    if not reference:
        return PerceptionDecision(
            accept_candidate=True,
            safety_score=1.0,
            reference_retention=1.0,
            matched_reference=0,
            reference_count=0,
            unmatched_candidate=len(candidate),
            reason="输入图未提供可用于一致性约束的道路目标，保留复原候选。",
        )

    used_candidates: set[int] = set()
    retained_weight = 0.0
    reference_weight = sum(max(item.confidence, 0.05) for item in reference)
    matched = 0
    for reference_item in sorted(reference, key=lambda item: item.confidence, reverse=True):
        options = [
            (box_iou(reference_item.box, candidate_item.box), index, candidate_item)
            for index, candidate_item in enumerate(candidate)
            if index not in used_candidates and candidate_item.class_id == reference_item.class_id
        ]
        if not options:
            continue
        overlap, index, candidate_item = max(options, key=lambda item: item[0])
        if overlap < iou_threshold:
            continue
        used_candidates.add(index)
        matched += 1
        confidence_ratio = min(1.0, candidate_item.confidence / max(reference_item.confidence, 1e-6))
        geometry_quality = min(1.0, overlap / 0.70)
        retained_weight += max(reference_item.confidence, 0.05) * (
            0.65 * confidence_ratio + 0.35 * geometry_quality
        )

    retention = float(np.clip(retained_weight / reference_weight, 0.0, 1.0))
    unmatched = [item for index, item in enumerate(candidate) if index not in used_candidates]
    unmatched_risk = sum(item.confidence**2 for item in unmatched) / max(
        1.0, sum(item.confidence**2 for item in candidate)
    )
    safety_score = float(np.clip(retention - 0.08 * unmatched_risk, 0.0, 1.0))
    accept = safety_score >= minimum_score
    reason = (
        "参考目标保持充分，采用保真复原候选。"
        if accept
        else "参考目标出现丢失、偏移或置信度下降，回退到输入帧。"
    )
    return PerceptionDecision(
        accept_candidate=accept,
        safety_score=safety_score,
        reference_retention=retention,
        matched_reference=matched,
        reference_count=len(reference),
        unmatched_candidate=len(unmatched),
        reason=reason,
    )
