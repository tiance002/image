"""Benchmark safety/performance modes on the bundled ACDC video."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import DEMO_VIDEO_PATH, DemoModels, configure_local_network

MODES = (
    ("safety_every_frame", True, 1),
    ("risk_adaptive", True, "adaptive"),
    ("balanced_every_3", True, 3),
    ("speed_single_detector", False, 1),
)


MEAN_METRICS = (
    "processed_frames",
    "average_latency_ms",
    "achieved_fps",
    "raw_temporal_error",
    "stabilized_temporal_error",
    "guard_accepts",
    "guard_fallbacks",
    "temporal_accepts",
    "frame_candidate_accepts",
    "input_fallbacks",
    "guard_evaluations",
    "restoration_ms",
    "temporal_ms",
    "detection_ms",
    "selection_disagreement_vs_every_frame",
    "candidate_on_baseline_fallback",
)


def aggregate_mode(mode: str, runs: list[dict]) -> dict:
    """Aggregate repeated runs while retaining dispersion for every metric."""
    row: dict[str, object] = {"mode": mode, "repeats": len(runs)}
    for metric in MEAN_METRICS:
        values = [float(run[metric]) for run in runs]
        row[metric] = statistics.fmean(values)
        row[f"{metric}_std"] = statistics.pstdev(values)
    return row


def render_report(
    rows: list[dict],
    frames: int,
    max_side: int,
    repeats: int,
    system: dict[str, str],
) -> str:
    lines = [
        "# 视频模式性能基准",
        "",
        f"> 内置 ACDC 序列，{frames} 帧，最大边 {max_side}，{repeats} 轮顺序轮换；数值为均值 ± 总体标准差。",
        f"> 设备：{system['device']}；PyTorch {system['torch']}；CUDA {system['cuda']}；Python {system['python']}。",
        "",
        "| 模式 | FPS | 延迟ms | 复原ms | 时序ms | 检测/门控ms | 分层评估 | 时序/逐帧/输入 | 风险漏判 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        fps = f"{row['achieved_fps']:.2f} ± {row['achieved_fps_std']:.2f}"
        latency = f"{row['average_latency_ms']:.1f} ± {row['average_latency_ms_std']:.1f}"
        lines.append(
            f"| {row['mode']} | {fps} | {latency} "
            f"| {row['restoration_ms']:.1f} | {row['temporal_ms']:.1f} "
            f"| {row['detection_ms']:.1f} | {row['guard_evaluations']:.1f} "
            f"| {row.get('temporal_accepts', row['guard_accepts'])}/"
            f"{row.get('frame_candidate_accepts', 0)}/"
            f"{row.get('input_fallbacks', row['guard_fallbacks'])} "
            f"| {row.get('candidate_on_baseline_fallback', 0):.1f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", default="runs/performance/video_modes")
    args = parser.parse_args()

    configure_local_network()
    models = DemoModels()
    models.warm_up()
    # Warm restoration, detector, and temporal code paths before timed modes.
    models.process_video(str(DEMO_VIDEO_PATH), 2, 10.0, args.max_side, 0.18, True, 1)
    repeats = max(1, int(args.repeats))
    runs_by_mode: dict[str, list[dict]] = {mode: [] for mode, _, _ in MODES}
    run_records: list[dict[str, object]] = []
    for repeat in range(repeats):
        offset = repeat % len(MODES)
        ordered_modes = MODES[offset:] + MODES[:offset]
        current: dict[str, tuple[dict, list[str]]] = {}
        for order, (mode, enabled, interval) in enumerate(ordered_modes):
            _, _, _, traces = models.process_video(
                str(DEMO_VIDEO_PATH),
                args.frames,
                10.0,
                args.max_side,
                0.18,
                enabled,
                interval,
            )
            payload = json.loads(Path(traces[0]).read_text(encoding="utf-8"))
            summary = {**payload["summary"]}
            selections = [
                str(frame["guard_selection"]) for frame in payload["frames"]
            ]
            current[mode] = (summary, selections)
            run_records.append(
                {"repeat": repeat + 1, "order": order + 1, "mode": mode, **summary}
            )

        baseline = current["safety_every_frame"][1]
        for mode, (summary, selections) in current.items():
            summary["selection_disagreement_vs_every_frame"] = sum(
                selected != reference for selected, reference in zip(selections, baseline)
            )
            summary["candidate_on_baseline_fallback"] = sum(
                reference == "input_fallback" and selected != "input_fallback"
                for selected, reference in zip(selections, baseline)
            )
            runs_by_mode[mode].append(summary)
            for record in reversed(run_records):
                if record["repeat"] == repeat + 1 and record["mode"] == mode:
                    record.update(
                        {
                            "selection_disagreement_vs_every_frame": summary[
                                "selection_disagreement_vs_every_frame"
                            ],
                            "candidate_on_baseline_fallback": summary[
                                "candidate_on_baseline_fallback"
                            ],
                        }
                    )
                    break

    rows = [aggregate_mode(mode, runs_by_mode[mode]) for mode, _, _ in MODES]
    system = {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "not available",
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "frames": args.frames,
                "max_side": args.max_side,
                "repeats": repeats,
                "system": system,
                "modes": rows,
                "runs": run_records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "REPORT.md").write_text(
        render_report(rows, args.frames, args.max_side, repeats, system), encoding="utf-8"
    )
    with (output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved video-mode benchmark to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
