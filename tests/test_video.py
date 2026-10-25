import json
import zipfile
from pathlib import Path

import cv2
import numpy as np

from saferestore.video import (
    build_sequence_video,
    create_video_result_bundle,
    render_safety_timeline,
    resize_to_max_side,
    safe_video_fps,
    simulate_realtime_schedule,
    summarize_video_traces,
    temporal_stabilize,
    write_video_trace,
)


def test_resize_to_max_side_preserves_aspect_ratio() -> None:
    image = np.zeros((600, 1200, 3), dtype=np.uint8)

    resized = resize_to_max_side(image, 300)

    assert resized.shape == (150, 300, 3)
    assert resize_to_max_side(image, 1600).shape == image.shape


def test_safe_video_fps_rejects_invalid_values() -> None:
    assert safe_video_fps(25.0) == 25.0
    assert safe_video_fps(0.0) == 10.0
    assert safe_video_fps(float("nan")) == 10.0
    assert safe_video_fps(120.0) == 60.0


def test_realtime_schedule_keeps_frames_within_budget() -> None:
    result = simulate_realtime_schedule([60.0] * 10, target_fps=10.0)

    assert result.frame_budget_ms == 100.0
    assert result.deadline_miss_count == 0
    assert result.simulated_keep_count == 10
    assert result.simulated_drop_count == 0
    assert result.simulated_effective_fps == 10.0


def test_realtime_schedule_drops_arrivals_while_worker_is_busy() -> None:
    result = simulate_realtime_schedule([60.0] * 10, target_fps=25.0)

    assert result.frame_budget_ms == 40.0
    assert result.deadline_miss_count == 10
    assert result.deadline_miss_ratio == 1.0
    assert result.simulated_keep_count == 5
    assert result.simulated_drop_count == 5
    assert result.simulated_effective_fps == 12.5


def test_temporal_stabilize_reduces_static_region_flicker() -> None:
    input_frame = np.full((48, 64, 3), 100, dtype=np.uint8)
    previous_output = np.full_like(input_frame, 100)
    current_output = np.full_like(input_frame, 130)

    result = temporal_stabilize(
        input_frame,
        current_output,
        input_frame,
        previous_output,
        strength=0.25,
    )

    assert not result.scene_cut
    assert result.mean_blend_weight > 0.20
    assert result.stabilized_temporal_error < result.raw_temporal_error
    assert 100 < float(result.image.mean()) < 130


def test_temporal_stabilize_resets_on_scene_cut() -> None:
    previous_input = np.zeros((48, 64, 3), dtype=np.uint8)
    current_input = np.full_like(previous_input, 255)
    current_output = np.full_like(previous_input, 220)

    result = temporal_stabilize(
        current_input,
        current_output,
        previous_input,
        previous_input,
    )

    assert result.scene_cut
    assert result.mean_blend_weight == 0.0
    assert np.array_equal(result.image, current_output)


def test_build_sequence_video(tmp_path: Path) -> None:
    images: list[Path] = []
    for index in range(3):
        path = tmp_path / f"frame_{index}.png"
        ok, encoded = cv2.imencode(
            ".png", np.full((40, 80, 3), index * 40, dtype=np.uint8)
        )
        assert ok
        encoded.tofile(path)
        images.append(path)

    output = build_sequence_video(images, tmp_path / "sample.mp4", fps=5.0, max_side=80)

    capture = cv2.VideoCapture(str(output))
    assert output.is_file()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    assert capture.get(cv2.CAP_PROP_FPS) == 5.0
    capture.release()


def test_write_video_trace_creates_json_and_csv(tmp_path: Path) -> None:
    video_path = tmp_path / "result.mp4"
    records = [{"frame": 1, "candidate_accepted": True, "score": 0.91}]

    json_path, csv_path = write_video_trace(
        video_path,
        {"schema_version": 1, "summary": {"processed_frames": 1}},
        records,
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["frames"] == records
    assert "candidate_accepted" in csv_path.read_text(encoding="utf-8-sig")


def test_create_video_result_bundle_has_manifest_and_all_artifacts(tmp_path: Path) -> None:
    video = tmp_path / "saferestore_test.mp4"
    trace_json = tmp_path / "saferestore_test.json"
    trace_csv = tmp_path / "saferestore_test.csv"
    timeline = tmp_path / "saferestore_test_safety_timeline.png"
    for path, content in (
        (video, b"video"),
        (trace_json, b"{}"),
        (trace_csv, b"frame,score\n1,0.9\n"),
        (timeline, b"png"),
    ):
        path.write_bytes(content)

    bundle = create_video_result_bundle(video, trace_json, trace_csv, timeline)

    with zipfile.ZipFile(bundle) as archive:
        assert set(archive.namelist()) == {
            video.name,
            trace_json.name,
            trace_csv.name,
            timeline.name,
            "manifest.json",
        }
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["experiment_id"] == "saferestore_test"
    assert {item["role"] for item in manifest["files"]} == {
        "annotated_video",
        "machine_readable_trace",
        "frame_table",
        "safety_timeline",
    }
    assert all(len(item["sha256"]) == 64 for item in manifest["files"])


def test_render_safety_timeline_creates_readable_png(tmp_path: Path) -> None:
    records = [
        {
            "frame": 1,
            "perception_safety_score": 0.95,
            "guard_selection": "temporal_candidate",
            "guard_evaluated": True,
            "scene_cut": False,
        },
        {
            "frame": 2,
            "perception_safety_score": 0.40,
            "guard_selection": "input_fallback",
            "guard_evaluated": True,
            "scene_cut": True,
        },
    ]

    output = render_safety_timeline(records, tmp_path / "timeline.png")
    image = cv2.imdecode(np.fromfile(output, dtype=np.uint8), cv2.IMREAD_COLOR)

    assert output.is_file()
    assert image is not None
    assert image.shape == (560, 1400, 3)


def test_summarize_video_traces_skips_invalid_files(tmp_path: Path) -> None:
    output = tmp_path / "saferestore_valid.mp4"
    write_video_trace(
        output,
        {
            "schema_version": 1,
            "settings": {"max_side": 640},
            "summary": {
                "processed_frames": 4,
                "achieved_fps": 2.5,
                "average_latency_ms": 400.0,
                "raw_temporal_error": 0.1,
                "stabilized_temporal_error": 0.08,
                "guard_accepts": 3,
                "guard_fallbacks": 1,
            },
        },
        [{"frame": 1}],
    )
    (tmp_path / "saferestore_broken.json").write_text("not json", encoding="utf-8")

    rows = summarize_video_traces(tmp_path)

    assert rows == [
        [
            "saferestore_valid",
            4,
            640,
            "FP32",
            2.5,
            400.0,
            0.0,
            0,
            "20.0%",
            3,
            1,
        ]
    ]
