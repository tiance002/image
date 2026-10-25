"""Run three deterministic, offline demo rehearsals and record reliability metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def array_digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def validate_result(result: tuple, input_shape: tuple[int, ...]) -> list[str]:
    errors: list[str] = []
    if len(result) != 6:
        return [f"expected 6 outputs, got {len(result)}"]
    for index, array in enumerate(result[:5]):
        if not isinstance(array, np.ndarray):
            errors.append(f"output {index} is not a numpy array")
            continue
        if array.shape[:2] != input_shape[:2]:
            errors.append(f"output {index} shape {array.shape} != input {input_shape}")
        if not np.isfinite(array).all():
            errors.append(f"output {index} contains non-finite values")
        if array.size == 0:
            errors.append(f"output {index} is empty")
    if not isinstance(result[5], str) or "复原模型" not in result[5]:
        errors.append("statistics text is missing the model identity")
    return errors


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output-dir", default="runs/rehearsal")
    args = parser.parse_args()
    if args.rounds < 3:
        raise ValueError("At least three rounds are required for a rehearsal.")

    # Import after timing setup so model initialization is recorded separately.
    sys.path.insert(0, str(ROOT))
    from app import DemoModels

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    samples: dict[str, np.ndarray] = {}
    for weather in ("fog", "rain", "rain_fog"):
        with Image.open(ROOT / "runs" / "assets" / f"{weather}.jpg") as image:
            samples[weather] = np.asarray(image.convert("RGB"))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    models = DemoModels()
    startup_seconds = time.perf_counter() - started

    # One unrecorded warmup removes CUDA kernel setup from repeated-use timings.
    models.process(samples["fog"])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    rows: list[dict] = []
    reference_hashes: dict[str, list[str]] = {}
    all_errors: list[str] = []
    for round_number in range(1, args.rounds + 1):
        for weather, sample in samples.items():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            result = models.process(sample)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            errors = validate_result(result, sample.shape)
            hashes = [array_digest(array) for array in result[:5]]
            if weather not in reference_hashes:
                reference_hashes[weather] = hashes
            deterministic = hashes == reference_hashes[weather]
            if not deterministic:
                errors.append("output hashes changed between rehearsal rounds")
            all_errors.extend(f"round {round_number} {weather}: {error}" for error in errors)
            rows.append(
                {
                    "round": round_number,
                    "weather": weather,
                    "elapsed_ms": round(elapsed_ms, 3),
                    "valid": not errors,
                    "deterministic": deterministic,
                    "restored_sha256": hashes[0],
                    "fused_sha256": hashes[1],
                    "detected_sha256": hashes[4],
                }
            )

    elapsed_values = np.asarray([float(row["elapsed_ms"]) for row in rows])
    peak_gpu_mb = (
        torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    )
    summary = {
        "scope": "offline_bundled_smoke_rehearsal",
        "rounds": args.rounds,
        "samples_per_round": len(samples),
        "total_runs": len(rows),
        "passed": not all_errors,
        "device": str(models.device),
        "startup_seconds": startup_seconds,
        "latency_ms": {
            "mean": float(elapsed_values.mean()),
            "p50": float(np.quantile(elapsed_values, 0.5)),
            "p95": float(np.quantile(elapsed_values, 0.95)),
            "max": float(elapsed_values.max()),
        },
        "peak_gpu_memory_mb": peak_gpu_mb,
        "errors": all_errors,
    }
    write_csv(output_dir / "runs.csv", rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    status = "通过" if summary["passed"] else "失败"
    report = (
        "# 离线演示彩排报告\n\n"
        f"- 结果：**{status}**\n"
        f"- 连续轮数：{args.rounds}\n"
        f"- 总处理次数：{len(rows)}（每轮 fog / rain / rain_fog 各 1 张）\n"
        f"- 设备：{models.device}\n"
        f"- 模型首次加载：{startup_seconds:.2f} s\n"
        f"- 重复处理平均耗时：{summary['latency_ms']['mean']:.2f} ms/张\n"
        f"- P95 耗时：{summary['latency_ms']['p95']:.2f} ms/张\n"
        f"- 峰值 GPU 显存：{peak_gpu_mb:.1f} MB\n"
        f"- 输出逐字节稳定：{'是' if all(row['deterministic'] for row in rows) else '否'}\n\n"
        "说明：该彩排完全使用项目内置权重和图片，不下载任何资源；耗时包括复原、"
        "不确定性、DCP 与 YOLO 检测，不包括模型首次加载。\n"
    )
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if all_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
