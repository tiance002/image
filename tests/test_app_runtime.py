import json
import os
from pathlib import Path

import numpy as np

from app import (
    DemoModels,
    configure_local_network,
    load_guard_evidence,
    load_performance_evidence,
    load_precision_evidence,
)
from saferestore.perception import DetectionRecord


def test_configure_local_network_preserves_and_adds_hosts(monkeypatch) -> None:
    monkeypatch.setenv("NO_PROXY", "example.test")

    configure_local_network()

    assert set(os.environ["NO_PROXY"].split(",")) == {"127.0.0.1", "localhost", "example.test"}
    assert set(os.environ["no_proxy"].split(",")) == {"127.0.0.1", "localhost", "example.test"}


def test_load_guard_evidence_states_statistical_limit(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "runtime_threshold": 0.9,
                "reference": {"map50": 0.2, "recall": 0.4, "precision": 0.7},
                "candidate": {"map50": 0.1, "recall": 0.3, "precision": 0.6},
                "thresholds": [
                    {
                        "threshold": 0.9,
                        "map50": 0.201,
                        "recall": 0.41,
                        "precision": 0.69,
                        "accept_count": 20,
                        "map50_difference_vs_input": {
                            "observed_difference": 0.001,
                            "ci_lower": -0.002,
                            "ci_upper": 0.003,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    evidence = load_guard_evidence(path)

    assert "0.2010" in evidence
    assert "区间跨越 0" in evidence


def test_load_performance_evidence_renders_stage_timings(tmp_path: Path) -> None:
    path = tmp_path / "performance.json"
    path.write_text(
        json.dumps(
            {
                "frames": 30,
                "max_side": 640,
                "modes": [
                    {
                        "mode": "safety_every_frame",
                        "achieved_fps": 12.3,
                        "average_latency_ms": 81.3,
                        "restoration_ms": 21.0,
                        "temporal_ms": 38.0,
                        "detection_ms": 22.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    evidence = load_performance_evidence(path)

    assert "30 帧" in evidence
    assert "每帧安全门控" in evidence
    assert "12.30" in evidence
    assert "0/0/0" in evidence


def test_load_performance_evidence_renders_repeated_uncertainty(tmp_path: Path) -> None:
    path = tmp_path / "performance_repeated.json"
    path.write_text(
        json.dumps(
            {
                "frames": 30,
                "max_side": 640,
                "repeats": 3,
                "modes": [
                    {
                        "mode": "risk_adaptive",
                        "achieved_fps": 9.83,
                        "achieved_fps_std": 0.50,
                        "average_latency_ms": 102.0,
                        "average_latency_ms_std": 5.0,
                        "restoration_ms": 29.0,
                        "temporal_ms": 42.3,
                        "detection_ms": 30.6,
                        "temporal_accepts": 12.0,
                        "frame_candidate_accepts": 0.0,
                        "input_fallbacks": 18.0,
                        "candidate_on_baseline_fallback": 3.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    evidence = load_performance_evidence(path)

    assert "3 轮" in evidence
    assert "9.83 ± 0.50" in evidence
    assert "12.0/0.0/18.0" in evidence
    assert "| 3.0 |" in evidence


def test_load_precision_evidence_states_speed_and_safety_limit(tmp_path: Path) -> None:
    path = tmp_path / "precision.json"
    path.write_text(
        json.dumps(
            {
                "frames": 30,
                "max_side": 640,
                "speedup": 1.01,
                "fp32": {
                    "achieved_fps": 9.41,
                    "average_latency_ms": 106.3,
                    "restoration_ms": 23.7,
                },
                "fp16": {
                    "achieved_fps": 9.50,
                    "average_latency_ms": 105.2,
                    "restoration_ms": 23.4,
                },
                "selection_comparison": {
                    "selection_disagreements": 0,
                    "compared_frames": 30,
                },
                "candidate_comparison": {"candidate_psnr_db": 63.28},
            }
        ),
        encoding="utf-8",
    )

    evidence = load_precision_evidence(path)

    assert "63.28 dB" in evidence
    assert "0/30" in evidence
    assert "1.010x" in evidence
    assert "默认仍采用 FP32" in evidence


def _guard_test_model(monkeypatch, records_by_marker: dict[int, list[DetectionRecord]]):
    model = DemoModels.__new__(DemoModels)
    monkeypatch.setattr(
        DemoModels,
        "_predict_rgb",
        lambda self, image, use_half=False: int(image[0, 0, 0]),
    )
    monkeypatch.setattr(
        DemoModels,
        "_detection_records",
        staticmethod(lambda marker: records_by_marker[marker]),
    )
    monkeypatch.setattr(
        DemoModels,
        "_summarize_detection",
        staticmethod(lambda marker: ({"car": len(records_by_marker[marker])}, [])),
    )
    monkeypatch.setattr(
        DemoModels,
        "_render_detection",
        staticmethod(lambda image, _result: image.copy()),
    )
    return model


def _image(marker: int) -> np.ndarray:
    return np.full((2, 2, 3), marker, dtype=np.uint8)


def _car() -> DetectionRecord:
    return DetectionRecord(
        class_id=2,
        confidence=0.9,
        box=np.asarray([0.0, 0.0, 10.0, 10.0]),
    )


def test_video_guard_prefers_safe_temporal_candidate(monkeypatch) -> None:
    model = _guard_test_model(monkeypatch, {1: [_car()], 2: [_car()], 3: [_car()]})

    selected, _, _, _, decision, selection, temporal_score = (
        model._apply_video_perception_guard(_image(1), _image(2), _image(3))
    )

    assert selection == "temporal_candidate"
    assert np.array_equal(selected, _image(3))
    assert decision.accept_candidate
    assert temporal_score >= 0.9


def test_video_guard_uses_frame_candidate_when_temporal_is_unsafe(monkeypatch) -> None:
    model = _guard_test_model(monkeypatch, {1: [_car()], 2: [_car()], 3: []})

    selected, _, _, _, decision, selection, temporal_score = (
        model._apply_video_perception_guard(_image(1), _image(2), _image(3))
    )

    assert selection == "frame_candidate"
    assert np.array_equal(selected, _image(2))
    assert decision.accept_candidate
    assert temporal_score < 0.9


def test_video_guard_falls_back_when_both_candidates_are_unsafe(monkeypatch) -> None:
    model = _guard_test_model(monkeypatch, {1: [_car()], 2: [], 3: []})

    selected, _, _, _, decision, selection, temporal_score = (
        model._apply_video_perception_guard(_image(1), _image(2), _image(3))
    )

    assert selection == "input_fallback"
    assert np.array_equal(selected, _image(1))
    assert not decision.accept_candidate
    assert temporal_score < 0.9
