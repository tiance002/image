"""Summarize five ACDC methods with dataset-level bootstrap confidence intervals."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.evaluate_coco_detection import evaluate_class
    from scripts.run_acdc_benchmark import METHODS
except ModuleNotFoundError:  # Direct execution places scripts/ rather than the project on sys.path.
    from evaluate_coco_detection import evaluate_class
    from run_acdc_benchmark import METHODS


def detection_inputs(
    records: list[dict], sampled_indices: np.ndarray | list[int] | None = None
) -> tuple[dict[int, list[dict]], dict[int, dict[int, list[np.ndarray]]]]:
    indices = range(len(records)) if sampled_indices is None else sampled_indices
    predictions: dict[int, list[dict]] = defaultdict(list)
    ground_truth: dict[int, dict[int, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for sample_id, record_index in enumerate(indices):
        record = records[int(record_index)]
        for class_id_text, boxes in record.get("ground_truth", {}).items():
            class_id = int(class_id_text)
            ground_truth[class_id][sample_id].extend(
                np.asarray(box, dtype=np.float32) for box in boxes
            )
        for class_id_text, items in record.get("predictions", {}).items():
            class_id = int(class_id_text)
            predictions[class_id].extend(
                {
                    "image_id": sample_id,
                    "confidence": float(item["confidence"]),
                    "box": np.asarray(item["box"], dtype=np.float32),
                }
                for item in items
            )
    return predictions, ground_truth


def dataset_map50(
    records: list[dict], sampled_indices: np.ndarray | list[int] | None = None
) -> float:
    predictions, ground_truth = detection_inputs(records, sampled_indices)
    class_ids = sorted(class_id for class_id, images in ground_truth.items() if images)
    if not class_ids:
        return 0.0
    values = [
        evaluate_class(predictions[class_id], ground_truth[class_id], 0.5)["ap50"]
        for class_id in class_ids
    ]
    return float(np.mean(values))


def bootstrap_indices(sample_count: int, resamples: int, seed: int) -> np.ndarray:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    return np.random.default_rng(seed).integers(
        0, sample_count, size=(resamples, sample_count), dtype=np.int32
    )


def bootstrap_map50(
    records: list[dict], indices: np.ndarray, confidence: float = 0.95
) -> dict[str, float | int]:
    values = np.asarray([dataset_map50(records, row) for row in indices], dtype=np.float64)
    alpha = (1.0 - confidence) / 2.0
    return {
        "observed": dataset_map50(records),
        "ci_lower": float(np.quantile(values, alpha)),
        "ci_upper": float(np.quantile(values, 1.0 - alpha)),
        "confidence": confidence,
        "resamples": int(indices.shape[0]),
    }


def paired_bootstrap_map50(
    candidate: list[dict], reference: list[dict], indices: np.ndarray, confidence: float = 0.95
) -> dict[str, float | int]:
    differences = np.asarray(
        [dataset_map50(candidate, row) - dataset_map50(reference, row) for row in indices],
        dtype=np.float64,
    )
    alpha = (1.0 - confidence) / 2.0
    return {
        "observed_difference": dataset_map50(candidate) - dataset_map50(reference),
        "ci_lower": float(np.quantile(differences, alpha)),
        "ci_upper": float(np.quantile(differences, 1.0 - alpha)),
        "confidence": confidence,
        "resamples": int(indices.shape[0]),
    }


def validate_method_payloads(payloads: dict[str, dict]) -> list[str]:
    reference = payloads["weather_input"]
    reference_files = [item["file_name"] for item in reference["per_image"]]
    frozen_fields = (
        "annotation_sha256",
        "detector_config_sha256",
        "iou_threshold",
        "confidence_threshold",
    )
    for method, payload in payloads.items():
        files = [item["file_name"] for item in payload.get("per_image", [])]
        if files != reference_files:
            raise ValueError(f"Image order differs for method: {method}")
        for field in frozen_fields:
            if payload.get(field) != reference.get(field):
                raise ValueError(f"Frozen field {field} differs for method: {method}")
    return reference_files


def micro_summary(records: list[dict]) -> dict[str, float | int]:
    true_positives = sum(int(item["true_positives"]) for item in records)
    false_positives = sum(int(item["false_positives"]) for item in records)
    false_negatives = sum(int(item["false_negatives"]) for item in records)
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": true_positives / max(true_positives + false_positives, 1),
        "recall": true_positives / max(true_positives + false_negatives, 1),
        "miss_rate": false_negatives / max(true_positives + false_negatives, 1),
        "false_positives_per_image": false_positives / max(len(records), 1),
    }


def group_by_weather(records: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in records:
        grouped[str(item["weather"])].append(item)
    return dict(sorted(grouped.items()))


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ACDC 五方法检测评测报告",
        "",
        "> 该报告使用冻结检测配置与同一组图像；bootstrap 单位为图像。",
        "",
        "| 方法 | mAP50 | 95% CI | 相对输入差值 | 差值95% CI | Precision | Recall | 漏检率 | 每图虚警 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = report["methods"][method]
        interval = item["map50_bootstrap"]
        difference = item["paired_difference_vs_weather_input"]
        lines.append(
            f"| {method} | {interval['observed']:.4f} | "
            f"[{interval['ci_lower']:.4f}, {interval['ci_upper']:.4f}] | "
            f"{difference['observed_difference']:+.4f} | "
            f"[{difference['ci_lower']:+.4f}, {difference['ci_upper']:+.4f}] | "
            f"{item['micro']['precision']:.4f} | {item['micro']['recall']:.4f} | "
            f"{item['micro']['miss_rate']:.4f} | "
            f"{item['micro']['false_positives_per_image']:.3f} |"
        )
    lines.extend(["", "## 分天气结果", ""])
    for weather, weather_methods in report["by_weather"].items():
        lines.extend(
            [
                f"### {weather}",
                "",
                "| 方法 | mAP50 | 95% CI | 相对输入差值 | 差值95% CI | 漏检率 | 每图虚警 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method in METHODS:
            item = weather_methods[method]
            interval = item["map50_bootstrap"]
            difference = item["paired_difference_vs_weather_input"]
            lines.append(
                f"| {method} | {interval['observed']:.4f} | "
                f"[{interval['ci_lower']:.4f}, {interval['ci_upper']:.4f}] | "
                f"{difference['observed_difference']:+.4f} | "
                f"[{difference['ci_lower']:+.4f}, {difference['ci_upper']:+.4f}] | "
                f"{item['micro']['miss_rate']:.4f} | "
                f"{item['micro']['false_positives_per_image']:.3f} |"
            )
        lines.append("")
    lines.extend(
        [
            "",
            "## 解释规则",
            "",
            "- 差值区间完全大于0：相对天气输入存在稳定正增益；",
            "- 差值区间跨0：只记录趋势，不宣称稳定提升；",
            "- 差值区间完全小于0：判定该方法对安全感知有稳定负面影响。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", default="runs/acdc_benchmark/evaluation")
    parser.add_argument("--output-dir", default="runs/acdc_benchmark/report")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    evaluation_dir = Path(args.evaluation_dir)
    output_dir = Path(args.output_dir)
    payloads = {
        method: json.loads((evaluation_dir / f"{method}.json").read_text(encoding="utf-8"))
        for method in METHODS
    }
    image_files = validate_method_payloads(payloads)
    indices = bootstrap_indices(len(image_files), args.resamples, args.seed)
    reference_records = payloads["weather_input"]["per_image"]
    methods: dict[str, Any] = {}
    for method in METHODS:
        records = payloads[method]["per_image"]
        recomputed_map50 = dataset_map50(records)
        reported_map50 = float(payloads[method]["map50"])
        consistency_error = abs(recomputed_map50 - reported_map50)
        if consistency_error > 1e-9:
            raise ValueError(
                f"Reported and recomputed mAP50 differ for {method}: "
                f"{reported_map50} vs {recomputed_map50}"
            )
        methods[method] = {
            "map50_reported": reported_map50,
            "map50_consistency_error": consistency_error,
            "map50_bootstrap": bootstrap_map50(records, indices),
            "paired_difference_vs_weather_input": paired_bootstrap_map50(
                records, reference_records, indices
            ),
            "micro": micro_summary(records),
            "per_class": payloads[method]["per_class"],
        }
    reference_by_weather = group_by_weather(reference_records)
    by_weather: dict[str, dict[str, Any]] = {}
    for weather_index, (weather, weather_reference) in enumerate(reference_by_weather.items()):
        weather_indices = bootstrap_indices(
            len(weather_reference), args.resamples, args.seed + weather_index + 1
        )
        weather_methods: dict[str, Any] = {}
        for method in METHODS:
            records = group_by_weather(payloads[method]["per_image"]).get(weather, [])
            if len(records) != len(weather_reference):
                raise ValueError(f"Weather image count differs for {method}/{weather}")
            weather_methods[method] = {
                "image_count": len(records),
                "map50_bootstrap": bootstrap_map50(records, weather_indices),
                "paired_difference_vs_weather_input": paired_bootstrap_map50(
                    records, weather_reference, weather_indices
                ),
                "micro": micro_summary(records),
            }
        by_weather[weather] = weather_methods
    report = {
        "scope": "acdc_five_method_bootstrap_summary",
        "image_count": len(image_files),
        "annotation_sha256": payloads["weather_input"]["annotation_sha256"],
        "detector_config_sha256": payloads["weather_input"]["detector_config_sha256"],
        "iou_threshold": payloads["weather_input"]["iou_threshold"],
        "confidence_threshold": payloads["weather_input"]["confidence_threshold"],
        "bootstrap": {"resamples": args.resamples, "seed": args.seed, "confidence": 0.95},
        "methods": methods,
        "by_weather": by_weather,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "REPORT.md").write_text(render_markdown(report), encoding="utf-8")
    with (output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "map50",
                "ci_lower",
                "ci_upper",
                "delta_vs_input",
                "delta_ci_lower",
                "delta_ci_upper",
                "precision",
                "recall",
                "miss_rate",
                "false_positives_per_image",
            ]
        )
        for method in METHODS:
            item = methods[method]
            interval = item["map50_bootstrap"]
            difference = item["paired_difference_vs_weather_input"]
            micro = item["micro"]
            writer.writerow(
                [
                    method,
                    interval["observed"],
                    interval["ci_lower"],
                    interval["ci_upper"],
                    difference["observed_difference"],
                    difference["ci_lower"],
                    difference["ci_upper"],
                    micro["precision"],
                    micro["recall"],
                    micro["miss_rate"],
                    micro["false_positives_per_image"],
                ]
            )
    with (output_dir / "by_weather.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "weather",
                "method",
                "image_count",
                "map50",
                "ci_lower",
                "ci_upper",
                "delta_vs_input",
                "delta_ci_lower",
                "delta_ci_upper",
                "miss_rate",
                "false_positives_per_image",
            ]
        )
        for weather, weather_methods in by_weather.items():
            for method in METHODS:
                item = weather_methods[method]
                interval = item["map50_bootstrap"]
                difference = item["paired_difference_vs_weather_input"]
                writer.writerow(
                    [
                        weather,
                        method,
                        item["image_count"],
                        interval["observed"],
                        interval["ci_lower"],
                        interval["ci_upper"],
                        difference["observed_difference"],
                        difference["ci_lower"],
                        difference["ci_upper"],
                        item["micro"]["miss_rate"],
                        item["micro"]["false_positives_per_image"],
                    ]
                )
    with (output_dir / "per_class.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "class",
                "ground_truth",
                "predictions",
                "precision",
                "recall",
                "ap50",
            ]
        )
        for method in METHODS:
            for class_name, metrics in methods[method]["per_class"].items():
                writer.writerow(
                    [
                        method,
                        class_name,
                        metrics["ground_truth"],
                        metrics["predictions"],
                        metrics["precision"],
                        metrics["recall"],
                        metrics["ap50"],
                    ]
                )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
