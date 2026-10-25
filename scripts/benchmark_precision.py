"""Compare FP32 and mixed FP16/FP32 video inference on the bundled sequence."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import DEMO_VIDEO_PATH, DemoModels, configure_local_network


def compare_selection_sequences(
    fp32_frames: list[dict[str, object]],
    fp16_frames: list[dict[str, object]],
) -> dict[str, int]:
    """Report safety-selection changes introduced by mixed precision."""
    pairs = list(zip(fp32_frames, fp16_frames))
    return {
        "compared_frames": len(pairs),
        "selection_disagreements": sum(
            left["guard_selection"] != right["guard_selection"]
            for left, right in pairs
        ),
        "fp16_candidate_on_fp32_fallback": sum(
            left["guard_selection"] == "input_fallback"
            and right["guard_selection"] != "input_fallback"
            for left, right in pairs
        ),
    }


def compare_candidate_images(
    models: DemoModels,
    video_path: Path,
    frames: int,
    max_side: int,
) -> dict[str, float]:
    """Measure candidate-image drift before temporal fusion and detection."""
    capture = cv2.VideoCapture(str(video_path))
    absolute_errors: list[float] = []
    squared_errors: list[float] = []
    compared = 0
    try:
        while compared < frames:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            height, width = frame_bgr.shape[:2]
            scale = min(1.0, max_side / max(height, width))
            if scale < 1.0:
                frame_bgr = cv2.resize(
                    frame_bgr,
                    (round(width * scale), round(height * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            _, _, _, fp32_candidate, _ = models._restore_rgb(frame_rgb, use_half=False)
            _, _, _, fp16_candidate, _ = models._restore_rgb(frame_rgb, use_half=True)
            difference = fp32_candidate.astype(np.float32) - fp16_candidate.astype(np.float32)
            absolute_errors.append(float(np.mean(np.abs(difference))))
            squared_errors.append(float(np.mean(np.square(difference))))
            compared += 1
    finally:
        capture.release()
    mse = float(np.mean(squared_errors)) if squared_errors else 0.0
    return {
        "compared_frames": compared,
        "candidate_mae_255": float(np.mean(absolute_errors)) if absolute_errors else 0.0,
        "candidate_psnr_db": 10.0 * math.log10(255.0**2 / mse) if mse > 0 else math.inf,
    }


def render_report(payload: dict[str, object]) -> str:
    fp32 = payload["fp32"]
    fp16 = payload["fp16"]
    selection = payload["selection_comparison"]
    images = payload["candidate_comparison"]
    return "\n".join(
        [
            "# FP16 轻量化验证",
            "",
            "> 复原网络使用 CUDA FP16 autocast，安全检测器保持 FP32。",
            "",
            "| 指标 | FP32 | 混合 FP16/FP32 |",
            "|---|---:|---:|",
            f"| FPS | {fp32['achieved_fps']:.2f} | {fp16['achieved_fps']:.2f} |",
            f"| 平均延迟 | {fp32['average_latency_ms']:.1f} ms | {fp16['average_latency_ms']:.1f} ms |",
            f"| 复原耗时 | {fp32['restoration_ms']:.1f} ms | {fp16['restoration_ms']:.1f} ms |",
            f"| 检测/门控耗时 | {fp32['detection_ms']:.1f} ms | {fp16['detection_ms']:.1f} ms |",
            "",
            f"- 总体加速比：`{payload['speedup']:.3f}x`。",
            f"- 候选图 MAE（0–255）：`{images['candidate_mae_255']:.4f}`。",
            f"- 候选图 PSNR：`{images['candidate_psnr_db']:.2f} dB`。",
            f"- 门控选择不一致：`{selection['selection_disagreements']}/{selection['compared_frames']}` 帧。",
            f"- FP16 在 FP32 会回退时采用候选：`{selection['fp16_candidate_on_fp32_fallback']}` 帧。",
            "",
            "> 该结果只用于当前硬件与序列的工程判断；若存在安全选择不一致，不应把 FP16 设为默认。",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--output-dir", default="runs/performance/precision")
    args = parser.parse_args()

    configure_local_network()
    models = DemoModels()
    models.warm_up()
    results: dict[str, dict[str, object]] = {}
    for precision in ("fp32", "fp16"):
        _, _, _, artifacts = models.process_video(
            str(DEMO_VIDEO_PATH),
            args.frames,
            10.0,
            args.max_side,
            0.18,
            True,
            1,
            precision,
        )
        trace = json.loads(Path(artifacts[0]).read_text(encoding="utf-8"))
        results[precision] = trace

    fp32_summary = results["fp32"]["summary"]
    fp16_summary = results["fp16"]["summary"]
    payload: dict[str, object] = {
        "frames": args.frames,
        "max_side": args.max_side,
        "fp32": fp32_summary,
        "fp16": fp16_summary,
        "speedup": fp16_summary["achieved_fps"] / fp32_summary["achieved_fps"],
        "selection_comparison": compare_selection_sequences(
            results["fp32"]["frames"],
            results["fp16"]["frames"],
        ),
        "candidate_comparison": compare_candidate_images(
            models,
            DEMO_VIDEO_PATH,
            args.frames,
            args.max_side,
        ),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "REPORT.md").write_text(render_report(payload), encoding="utf-8")
    print(f"Saved precision benchmark to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
