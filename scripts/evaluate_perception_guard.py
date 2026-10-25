"""Evaluate the detector-consistency safety gate on frozen ACDC predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from saferestore.perception import (
    DEFAULT_PERCEPTION_THRESHOLD,
    DetectionRecord,
    decide_perception_safe,
)

try:
    from scripts.summarize_acdc_benchmark import (
        bootstrap_indices,
        dataset_map50,
        micro_summary,
        paired_bootstrap_map50,
    )
except ModuleNotFoundError:
    from summarize_acdc_benchmark import (
        bootstrap_indices,
        dataset_map50,
        micro_summary,
        paired_bootstrap_map50,
    )


def detection_records(predictions: dict[str, list[dict]]) -> list[DetectionRecord]:
    """Flatten an evaluation-record prediction mapping for the runtime guard."""
    return [
        DetectionRecord(
            class_id=int(class_id),
            confidence=float(item["confidence"]),
            box=np.asarray(item["box"], dtype=np.float32),
        )
        for class_id, items in predictions.items()
        for item in items
    ]


def select_guarded_records(
    reference: list[dict], candidate: list[dict], threshold: float
) -> tuple[list[dict], list[dict]]:
    """Apply the runtime decision rule to aligned frozen predictions."""
    if len(reference) != len(candidate):
        raise ValueError("Reference and candidate record counts differ.")
    selected: list[dict] = []
    decisions: list[dict] = []
    for reference_item, candidate_item in zip(reference, candidate, strict=True):
        if reference_item["file_name"] != candidate_item["file_name"]:
            raise ValueError("Reference and candidate image order differs.")
        decision = decide_perception_safe(
            detection_records(reference_item.get("predictions", {})),
            detection_records(candidate_item.get("predictions", {})),
            minimum_score=threshold,
        )
        chosen = candidate_item if decision.accept_candidate else reference_item
        selected.append(dict(chosen))
        decisions.append(
            {
                "file_name": reference_item["file_name"],
                "weather": reference_item.get("weather", "unknown"),
                "safety_score": decision.safety_score,
                "reference_retention": decision.reference_retention,
                "accept_candidate": decision.accept_candidate,
                "matched_reference": decision.matched_reference,
                "reference_count": decision.reference_count,
                "unmatched_candidate": decision.unmatched_candidate,
            }
        )
    return selected, decisions


def metric_summary(records: list[dict]) -> dict[str, float | int]:
    return {"map50": dataset_map50(records), **micro_summary(records)}


def evaluate_threshold(
    reference: list[dict],
    candidate: list[dict],
    threshold: float,
    bootstrap: np.ndarray,
) -> tuple[dict, list[dict]]:
    selected, decisions = select_guarded_records(reference, candidate, threshold)
    accepts = sum(bool(item["accept_candidate"]) for item in decisions)
    return (
        {
            "threshold": threshold,
            "accept_count": accepts,
            "fallback_count": len(decisions) - accepts,
            "accept_rate": accepts / max(len(decisions), 1),
            "mean_safety_score": float(np.mean([item["safety_score"] for item in decisions])),
            **metric_summary(selected),
            "map50_difference_vs_input": paired_bootstrap_map50(
                selected, reference, bootstrap
            ),
            "map50_difference_vs_candidate": paired_bootstrap_map50(
                selected, candidate, bootstrap
            ),
        },
        decisions,
    )


def render_report(payload: dict) -> str:
    reference = payload["reference"]
    candidate = payload["candidate"]
    lines = [
        "# ACDC 检测安全门控离线评测",
        "",
        "> 输入检测仅作为运行时参考证据；ACDC 真值只用于本报告的离线评价。阈值扫描属于验证集探索，不是独立测试集结论。",
        "",
        f"- 输入基线 mAP50：{reference['map50']:.4f}，Recall：{reference['recall']:.4f}",
        f"- 复原候选 mAP50：{candidate['map50']:.4f}，Recall：{candidate['recall']:.4f}",
        "",
        "| 安全阈值 | 采用候选 | 回退输入 | mAP50 | Recall | Precision | 相对输入mAP50 | 95% CI |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in payload["thresholds"]:
        difference = item["map50_difference_vs_input"]
        lines.append(
            f"| {item['threshold']:.2f} | {item['accept_count']} ({item['accept_rate']:.1%}) "
            f"| {item['fallback_count']} | {item['map50']:.4f} | {item['recall']:.4f} "
            f"| {item['precision']:.4f} | {difference['observed_difference']:+.4f} "
            f"| [{difference['ci_lower']:+.4f}, {difference['ci_upper']:+.4f}] |"
        )
    selected = next(
        item for item in payload["thresholds"] if item["threshold"] == payload["runtime_threshold"]
    )
    lines.extend(
        [
            "",
            "## 当前运行时阈值",
            "",
            (
                f"当前界面使用 `{payload['runtime_threshold']:.2f}`：采用候选 "
                f"{selected['accept_count']} 张，回退 {selected['fallback_count']} 张，"
                f"mAP50 为 {selected['map50']:.4f}。"
            ),
            "",
            "门控若未超过输入基线，结论应表述为“限制复原损失”而不是“提升检测”。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", default="runs/acdc_benchmark/evaluation/weather_input.json"
    )
    parser.add_argument(
        "--candidate", default="runs/acdc_benchmark/evaluation/calibrated_fusion.json"
    )
    parser.add_argument("--output-dir", default="runs/acdc_benchmark/perception_guard")
    parser.add_argument(
        "--runtime-threshold", type=float, default=DEFAULT_PERCEPTION_THRESHOLD
    )
    parser.add_argument("--resamples", type=int, default=500)
    args = parser.parse_args()

    reference_payload = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    candidate_payload = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    reference = reference_payload["per_image"]
    candidate = candidate_payload["per_image"]
    bootstrap = bootstrap_indices(len(reference), args.resamples, seed=42)
    thresholds = sorted({0.0, 0.5, 0.6, args.runtime_threshold, 0.8, 0.9, 1.01})
    summaries: list[dict] = []
    runtime_decisions: list[dict] = []
    for threshold in thresholds:
        summary, decisions = evaluate_threshold(reference, candidate, threshold, bootstrap)
        summaries.append(summary)
        if threshold == args.runtime_threshold:
            runtime_decisions = decisions

    payload = {
        "scope": "acdc_detector_consistency_guard_evaluation",
        "runtime_threshold": args.runtime_threshold,
        "bootstrap_resamples": args.resamples,
        "reference": metric_summary(reference),
        "candidate": metric_summary(candidate),
        "thresholds": summaries,
        "runtime_decisions": runtime_decisions,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "REPORT.md").write_text(render_report(payload), encoding="utf-8")
    with (output_dir / "thresholds.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "threshold",
                "accept_count",
                "fallback_count",
                "accept_rate",
                "mean_safety_score",
                "map50",
                "precision",
                "recall",
                "miss_rate",
                "false_positives_per_image",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Saved perception-guard evaluation to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
