"""Small, testable helpers for the offline video simulation pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class TemporalStabilizationResult:
    """Outputs and diagnostics from one motion-compensated temporal update."""

    image: np.ndarray
    mean_blend_weight: float
    scene_cut: bool
    raw_temporal_error: float
    stabilized_temporal_error: float


@dataclass(frozen=True)
class RealtimeSimulationResult:
    """Single-worker camera-stream feasibility estimated from measured latency."""

    target_fps: float
    frame_budget_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    deadline_miss_count: int
    deadline_miss_ratio: float
    simulated_keep_count: int
    simulated_drop_count: int
    simulated_effective_fps: float


def simulate_realtime_schedule(
    frame_latencies_ms: list[float],
    target_fps: float,
) -> RealtimeSimulationResult:
    """Simulate a drop-while-busy stream scheduler from measured frame costs.

    Frames arrive at a fixed sensor cadence. A frame arriving while the single
    processing worker is busy is dropped, matching a low-latency latest-frame
    policy rather than allowing an ever-growing queue.
    """
    if not frame_latencies_ms:
        raise ValueError("At least one frame latency is required.")
    latencies = np.asarray(frame_latencies_ms, dtype=np.float64)
    if not np.all(np.isfinite(latencies)) or np.any(latencies < 0):
        raise ValueError("Frame latencies must be finite and non-negative.")
    target_fps = safe_video_fps(float(target_fps))
    frame_budget_ms = 1000.0 / target_fps
    deadline_misses = int(np.count_nonzero(latencies > frame_budget_ms))
    available_at_ms = 0.0
    kept = 0
    dropped = 0
    for index, latency_ms in enumerate(latencies):
        arrival_ms = index * frame_budget_ms
        if arrival_ms + 1e-9 < available_at_ms:
            dropped += 1
            continue
        kept += 1
        available_at_ms = arrival_ms + float(latency_ms)
    return RealtimeSimulationResult(
        target_fps=target_fps,
        frame_budget_ms=frame_budget_ms,
        latency_p50_ms=float(np.percentile(latencies, 50)),
        latency_p95_ms=float(np.percentile(latencies, 95)),
        deadline_miss_count=deadline_misses,
        deadline_miss_ratio=deadline_misses / len(latencies),
        simulated_keep_count=kept,
        simulated_drop_count=dropped,
        simulated_effective_fps=kept * target_fps / len(latencies),
    )


def resize_to_max_side(frame: np.ndarray, max_side: int) -> np.ndarray:
    """Resize a BGR/RGB frame while preserving its aspect ratio."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected an HxWx3 colour frame.")
    if max_side <= 0:
        raise ValueError("max_side must be positive.")
    height, width = frame.shape[:2]
    scale = min(1.0, float(max_side) / max(height, width))
    if scale == 1.0:
        return frame.copy()
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(frame, target, interpolation=cv2.INTER_AREA)


def safe_video_fps(value: float, fallback: float = 10.0) -> float:
    """Return a finite FPS accepted by OpenCV video writers."""
    if not np.isfinite(value) or value <= 0:
        return float(fallback)
    return float(np.clip(value, 1.0, 60.0))


def temporal_stabilize(
    current_input: np.ndarray,
    current_output: np.ndarray,
    previous_input: np.ndarray,
    previous_output: np.ndarray,
    strength: float = 0.18,
    flow_max_side: int = 320,
) -> TemporalStabilizationResult:
    """Reduce frame flicker using guarded motion-compensated blending.

    Optical flow warps the previous frame into the current view. The previous
    output is reused only where the warped inputs still agree and the current
    image is not on a strong edge. This avoids the obvious ghosting caused by
    unconditional exponential averaging.
    """
    frames = (current_input, current_output, previous_input, previous_output)
    if any(frame.ndim != 3 or frame.shape[2] != 3 for frame in frames):
        raise ValueError("Temporal stabilization expects four HxWx3 frames.")
    if any(frame.shape != current_input.shape for frame in frames[1:]):
        raise ValueError("Temporal stabilization frames must have matching shapes.")
    strength = float(np.clip(strength, 0.0, 0.45))

    current_gray = cv2.cvtColor(current_input, cv2.COLOR_RGB2GRAY)
    previous_gray = cv2.cvtColor(previous_input, cv2.COLOR_RGB2GRAY)
    current_hist = cv2.calcHist([current_gray], [0], None, [32], [0, 256])
    previous_hist = cv2.calcHist([previous_gray], [0], None, [32], [0, 256])
    cv2.normalize(current_hist, current_hist)
    cv2.normalize(previous_hist, previous_hist)
    histogram_distance = cv2.compareHist(
        current_hist, previous_hist, cv2.HISTCMP_BHATTACHARYYA
    )
    if histogram_distance > 0.62:
        return TemporalStabilizationResult(
            image=current_output.copy(),
            mean_blend_weight=0.0,
            scene_cut=True,
            raw_temporal_error=float("nan"),
            stabilized_temporal_error=float("nan"),
        )

    height, width = current_gray.shape
    flow_scale = min(1.0, float(flow_max_side) / max(height, width))
    if flow_scale < 1.0:
        flow_size = (max(1, round(width * flow_scale)), max(1, round(height * flow_scale)))
        flow_current = cv2.resize(current_gray, flow_size, interpolation=cv2.INTER_AREA)
        flow_previous = cv2.resize(previous_gray, flow_size, interpolation=cv2.INTER_AREA)
    else:
        flow_current, flow_previous = current_gray, previous_gray
    flow = cv2.calcOpticalFlowFarneback(
        flow_current,
        flow_previous,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    if flow_scale < 1.0:
        small_height, small_width = flow.shape[:2]
        flow = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
        flow[..., 0] *= width / small_width
        flow[..., 1] *= height / small_height
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]
    warped_previous_input = cv2.remap(
        previous_input,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    warped_previous_output = cv2.remap(
        previous_output,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )

    input_difference = np.mean(
        np.abs(current_input.astype(np.float32) - warped_previous_input.astype(np.float32)),
        axis=2,
    )
    gradient_x = cv2.Sobel(current_gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(current_gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_magnitude = cv2.magnitude(gradient_x, gradient_y)
    reliability = np.exp(-np.square(input_difference / 24.0))
    reliability *= np.exp(-edge_magnitude / 180.0)
    blend_weight = strength * reliability

    current_float = current_output.astype(np.float32)
    previous_float = warped_previous_output.astype(np.float32)
    raw_error = float(np.mean(np.abs(current_float - previous_float)) / 255.0)
    alpha = blend_weight[..., None]
    stabilized_float = current_float * (1.0 - alpha) + previous_float * alpha
    stabilized = np.clip(stabilized_float, 0, 255).astype(np.uint8)
    stabilized_error = float(np.mean(np.abs(stabilized_float - previous_float)) / 255.0)
    return TemporalStabilizationResult(
        image=stabilized,
        mean_blend_weight=float(blend_weight.mean()),
        scene_cut=False,
        raw_temporal_error=raw_error,
        stabilized_temporal_error=stabilized_error,
    )


def build_sequence_video(
    image_paths: list[Path],
    output_path: Path,
    fps: float = 10.0,
    max_side: int = 960,
) -> Path:
    """Build a deterministic MP4 from an ordered list of image frames."""
    if not image_paths:
        raise ValueError("At least one image is required to build a video.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(suffix=".mp4")
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    writer: cv2.VideoWriter | None = None
    completed = False
    try:
        for path in image_paths:
            encoded = np.fromfile(path, dtype=np.uint8)
            frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"Could not read video frame: {path}")
            frame = resize_to_max_side(frame, max_side)
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(temporary_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    safe_video_fps(fps),
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not create video: {output_path}")
            writer.write(frame)
        completed = True
    finally:
        if writer is not None:
            writer.release()
    try:
        if completed:
            shutil.copy2(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path


def write_video_trace(
    output_video: Path,
    metadata: dict[str, object],
    frame_records: list[dict[str, object]],
) -> tuple[Path, Path]:
    """Save a machine-readable JSON trace and spreadsheet-friendly CSV."""
    if not frame_records:
        raise ValueError("At least one frame record is required.")
    json_path = output_video.with_suffix(".json")
    csv_path = output_video.with_suffix(".csv")
    payload = {**metadata, "frames": frame_records}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(frame_records[0]))
        writer.writeheader()
        writer.writerows(frame_records)
    return json_path, csv_path


def create_video_result_bundle(
    output_video: Path,
    trace_json: Path,
    trace_csv: Path,
    safety_timeline: Path,
) -> Path:
    """Package one video experiment with a checksummed artifact manifest."""
    artifacts = [
        (output_video, "annotated_video"),
        (trace_json, "machine_readable_trace"),
        (trace_csv, "frame_table"),
        (safety_timeline, "safety_timeline"),
    ]
    missing = [str(path) for path, _ in artifacts if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot create result bundle; missing: {', '.join(missing)}")

    bundle_path = output_video.with_name(f"{output_video.stem}_result_bundle.zip")
    manifest_files = []
    for path, role in artifacts:
        content = path.read_bytes()
        manifest_files.append(
            {
                "name": path.name,
                "role": role,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    manifest = {
        "schema_version": 1,
        "experiment_id": output_video.stem,
        "files": manifest_files,
    }
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, _ in artifacts:
            archive.write(path, arcname=path.name)
        archive.writestr(
            "manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2),
        )
    return bundle_path


def render_safety_timeline(
    frame_records: list[dict[str, object]],
    output_path: Path,
    threshold: float = 0.90,
) -> Path:
    """Render an auditable per-frame safety timeline without GUI dependencies."""
    if not frame_records:
        raise ValueError("At least one frame record is required.")
    width, height = 1400, 560
    left, right, top, plot_bottom = 92, 42, 54, 390
    canvas = np.full((height, width, 3), (247, 249, 252), dtype=np.uint8)
    plot_width = width - left - right
    plot_height = plot_bottom - top
    count = len(frame_records)

    def x_position(index: int) -> int:
        return left + round(index * plot_width / max(count - 1, 1))

    def y_position(score: float) -> int:
        return plot_bottom - round(float(np.clip(score, 0.0, 1.0)) * plot_height)

    cv2.rectangle(canvas, (left, top), (width - right, plot_bottom), (255, 255, 255), -1)
    for tick in range(6):
        score = tick / 5
        y = y_position(score)
        cv2.line(canvas, (left, y), (width - right, y), (224, 228, 234), 1)
        cv2.putText(
            canvas,
            f"{score:.1f}",
            (34, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (75, 85, 99),
            1,
            cv2.LINE_AA,
        )

    threshold_y = y_position(threshold)
    for x in range(left, width - right, 18):
        cv2.line(
            canvas,
            (x, threshold_y),
            (min(x + 10, width - right), threshold_y),
            (60, 72, 220),
            2,
        )
    cv2.putText(
        canvas,
        f"safety threshold {threshold:.2f}",
        (left + 10, threshold_y - 9),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (60, 72, 220),
        1,
        cv2.LINE_AA,
    )

    points: list[tuple[int, int]] = []
    for index, record in enumerate(frame_records):
        value = record.get("perception_safety_score")
        if value is not None:
            points.append((x_position(index), y_position(float(value))))
    if len(points) >= 2:
        cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, (181, 104, 25), 3)

    selection_colors = {
        "temporal_candidate": (116, 156, 22),
        "frame_candidate": (23, 156, 220),
        "input_fallback": (113, 113, 122),
    }
    band_top, band_bottom = 424, 466
    for index, record in enumerate(frame_records):
        start_x = left + round(index * plot_width / count)
        end_x = left + round((index + 1) * plot_width / count)
        color = selection_colors.get(str(record.get("guard_selection")), (180, 180, 180))
        cv2.rectangle(canvas, (start_x, band_top), (max(start_x + 1, end_x), band_bottom), color, -1)
        if bool(record.get("guard_evaluated")):
            cv2.circle(canvas, (x_position(index), plot_bottom + 10), 5, (32, 32, 32), -1)
        if bool(record.get("scene_cut")):
            x = x_position(index)
            cv2.line(canvas, (x, top), (x, band_bottom), (170, 40, 150), 2)

    cv2.putText(canvas, "Detection safety score", (left, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (35, 42, 52), 2, cv2.LINE_AA)
    cv2.putText(canvas, "frame", (width - 92, plot_bottom + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (75, 85, 99), 1, cv2.LINE_AA)
    cv2.putText(canvas, "final selection", (left, band_top - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (55, 65, 78), 1, cv2.LINE_AA)
    legends = [
        ("temporal candidate", selection_colors["temporal_candidate"]),
        ("frame candidate", selection_colors["frame_candidate"]),
        ("input fallback", selection_colors["input_fallback"]),
        ("guard evaluated", (32, 32, 32)),
        ("scene cut", (170, 40, 150)),
    ]
    cursor_x = left
    for label, color in legends:
        cv2.rectangle(canvas, (cursor_x, 505), (cursor_x + 20, 521), color, -1)
        cv2.putText(canvas, label, (cursor_x + 27, 519), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (55, 65, 78), 1, cv2.LINE_AA)
        cursor_x += 238

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError(f"Could not encode safety timeline: {output_path}")
    encoded.tofile(output_path)
    return output_path


def summarize_video_traces(trace_directory: Path, limit: int = 10) -> list[list[object]]:
    """Return recent valid video runs as rows suitable for a UI table."""
    rows: list[list[object]] = []
    if not trace_directory.is_dir():
        return rows
    paths = sorted(
        trace_directory.glob("saferestore_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            summary = payload["summary"]
            settings = payload["settings"]
            raw_error = float(summary.get("raw_temporal_error", 0.0))
            stable_error = float(summary.get("stabilized_temporal_error", 0.0))
            reduction = 1.0 - stable_error / raw_error if raw_error > 1e-9 else 0.0
            rows.append(
                [
                    path.stem,
                    int(summary["processed_frames"]),
                    int(settings["max_side"]),
                    str(settings.get("effective_precision", "fp32")).upper(),
                    round(float(summary["achieved_fps"]), 2),
                    round(float(summary["average_latency_ms"]), 1),
                    round(float(summary.get("latency_p95_ms", 0.0)), 1),
                    int(summary.get("simulated_drop_count", 0)),
                    f"{reduction:.1%}",
                    int(summary.get("guard_accepts", 0)),
                    int(summary.get("guard_fallbacks", 0)),
                ]
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
            continue
        if len(rows) >= limit:
            break
    return rows
