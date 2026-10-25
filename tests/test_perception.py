import numpy as np

from saferestore.perception import (
    DetectionRecord,
    decide_perception_safe,
    schedule_perception_guard,
)


def detection(class_id: int, confidence: float, box: list[float]) -> DetectionRecord:
    return DetectionRecord(class_id, confidence, np.asarray(box, dtype=np.float32))


def test_guard_accepts_well_preserved_reference_detection() -> None:
    reference = [detection(2, 0.80, [10, 10, 40, 40])]
    candidate = [detection(2, 0.77, [11, 10, 41, 40])]

    decision = decide_perception_safe(reference, candidate)

    assert decision.accept_candidate
    assert decision.matched_reference == 1
    assert decision.reference_retention > 0.90


def test_guard_falls_back_when_reference_object_disappears() -> None:
    reference = [detection(0, 0.85, [10, 10, 30, 50])]
    candidate: list[DetectionRecord] = []

    decision = decide_perception_safe(reference, candidate)

    assert not decision.accept_candidate
    assert decision.safety_score == 0.0
    assert "回退" in decision.reason


def test_guard_does_not_reject_when_input_has_no_reference_objects() -> None:
    candidate = [detection(2, 0.60, [5, 5, 25, 25])]

    decision = decide_perception_safe([], candidate)

    assert decision.accept_candidate
    assert decision.unmatched_candidate == 1


def test_adaptive_schedule_rechecks_risky_events_immediately() -> None:
    assert schedule_perception_guard(0, 0, 1.0, "temporal_candidate", False) == (
        True,
        "initial_frame",
    )
    assert schedule_perception_guard(4, 1, 1.0, "temporal_candidate", True) == (
        True,
        "scene_cut",
    )
    assert schedule_perception_guard(4, 1, 0.2, "input_fallback", False) == (
        True,
        "candidate_recovery",
    )
    assert schedule_perception_guard(4, 1, 0.93, "temporal_candidate", False) == (
        True,
        "low_safety_margin",
    )


def test_adaptive_schedule_reuses_only_high_margin_decision() -> None:
    assert schedule_perception_guard(1, 1, 0.98, "temporal_candidate", False) == (
        False,
        "high_margin_reuse",
    )
    assert schedule_perception_guard(3, 3, 0.98, "temporal_candidate", False) == (
        True,
        "periodic_refresh",
    )
