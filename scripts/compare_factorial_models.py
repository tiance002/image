"""Compare legacy and factorial-trained models on the balanced frozen test set."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from saferestore.config import load_config
from saferestore.data import PairedWeatherDataset
from saferestore.losses.restoration import ssim_loss
from saferestore.metrics import spearman_correlation
from saferestore.statistics import bootstrap_mean_interval, paired_bootstrap_difference
from saferestore.train import build_model

ROOT = Path(__file__).resolve().parents[1]
MODEL_SPECS = {
    "v2_confounded": (
        ROOT / "configs" / "demo_v2_control.yaml",
        ROOT / "runs" / "demo_v2_control" / "best.pt",
    ),
    "v3_factorial": (
        ROOT / "configs" / "v3_factorial_seed42.yaml",
        ROOT / "runs" / "v3_factorial_seed42" / "best.pt",
    ),
}


def quality_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    difference = prediction.double() - target.double()
    mae = float(difference.abs().mean())
    mse = float((difference**2).mean())
    psnr = float("inf") if mse == 0.0 else 10.0 * math.log10(1.0 / mse)
    ssim = 1.0 - float(ssim_loss(prediction.unsqueeze(0), target.unsqueeze(0)))
    return {"mae": mae, "psnr_db": psnr, "ssim": ssim}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    output_dir = ROOT / "runs" / "factorial_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_config = load_config(ROOT / "configs" / "v3_factorial_seed42.yaml")
    image_size = tuple(int(value) for value in evaluation_config["data"]["image_size"])
    dataset = PairedWeatherDataset(
        evaluation_config["data"]["test_manifest"], image_size=image_size
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict] = []
    for model_name, (config_path, checkpoint_path) in MODEL_SPECS.items():
        model = build_model(load_config(config_path)).to(device).eval()
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(payload.get("model", payload))
        with torch.inference_mode():
            for index in range(len(dataset)):
                sample = dataset[index]
                weather = sample["weather"].to(device)
                clean = sample["clean"].to(device)
                outputs = model(weather.unsqueeze(0))
                input_metrics = quality_metrics(weather.cpu(), clean.cpu())
                direct_metrics = quality_metrics(outputs["restored"][0].cpu(), clean.cpu())
                raw_metrics = quality_metrics(outputs["conservative"][0].cpu(), clean.cpu())
                rows.append(
                    {
                        "model": model_name,
                        "scene_id": sample["scene_id"],
                        "weather": dataset.rows[index]["weather"],
                        "severity": dataset.rows[index]["severity"],
                        "severity_target": float(sample["severity"]),
                        "severity_predicted": float(outputs["degradation"][0, 2]),
                        "uncertainty_mean": float(outputs["uncertainty"].mean()),
                        **{f"input_{key}": value for key, value in input_metrics.items()},
                        **{f"direct_{key}": value for key, value in direct_metrics.items()},
                        **{f"raw_{key}": value for key, value in raw_metrics.items()},
                    }
                )
    write_csv(output_dir / "per_image_metrics.csv", rows)

    summary_rows: list[dict] = []
    model_summaries: dict[str, dict] = {}
    for model_name in MODEL_SPECS:
        selected = [row for row in rows if row["model"] == model_name]
        summary: dict[str, object] = {
            "sample_count": len(selected),
            "severity_mae": float(
                np.mean(
                    [abs(row["severity_predicted"] - row["severity_target"]) for row in selected]
                )
            ),
            "severity_spearman": spearman_correlation(
                np.asarray([row["severity_predicted"] for row in selected]),
                np.asarray([row["severity_target"] for row in selected]),
            ),
        }
        for weather_group in ("overall", "rain", "fog", "rain_fog"):
            group = selected if weather_group == "overall" else [row for row in selected if row["weather"] == weather_group]
            for method in ("input", "direct", "raw"):
                for metric in ("mae", "psnr_db", "ssim"):
                    interval = bootstrap_mean_interval(
                        [row[f"{method}_{metric}"] for row in group], seed=42
                    )
                    summary_rows.append(
                        {
                            "model": model_name,
                            "weather": weather_group,
                            "method": method,
                            "metric": metric,
                            **interval,
                        }
                    )
        model_summaries[model_name] = summary

    v2_rows = sorted(
        [row for row in rows if row["model"] == "v2_confounded"], key=lambda row: row["scene_id"]
    )
    v3_rows = sorted(
        [row for row in rows if row["model"] == "v3_factorial"], key=lambda row: row["scene_id"]
    )
    if [row["scene_id"] for row in v2_rows] != [row["scene_id"] for row in v3_rows]:
        raise RuntimeError("Model results do not contain identical paired scenes.")
    comparisons: dict[str, dict] = {}
    for metric in ("mae", "psnr_db", "ssim"):
        comparisons[metric] = paired_bootstrap_difference(
            [row[f"direct_{metric}"] for row in v3_rows],
            [row[f"direct_{metric}"] for row in v2_rows],
            seed=42,
        )
    write_csv(output_dir / "bootstrap_summary.csv", summary_rows)
    results = {
        "scope": "balanced_factorial_synthetic_test_only",
        "design": "3_weather_x_3_severity_x_2_test_repetitions",
        "models": model_summaries,
        "v3_minus_v2_direct": comparisons,
    }
    (output_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    v2_psnr = next(
        row["mean"]
        for row in summary_rows
        if row["model"] == "v2_confounded"
        and row["weather"] == "overall"
        and row["method"] == "direct"
        and row["metric"] == "psnr_db"
    )
    v3_psnr = next(
        row["mean"]
        for row in summary_rows
        if row["model"] == "v3_factorial"
        and row["weather"] == "overall"
        and row["method"] == "direct"
        and row["metric"] == "psnr_db"
    )
    psnr_delta = comparisons["psnr_db"]
    report = f"""# V2与V3全因子公平对照

范围：18张合成冻结测试图，覆盖3种天气×3种强度×2次重复；仅用于纠正实验设计和筛选候选模型。

| 模型 | 直接复原PSNR | 强度MAE | 强度Spearman |
|---|---:|---:|---:|
| V2（旧数据因素混杂） | {v2_psnr:.3f} dB | {model_summaries['v2_confounded']['severity_mae']:.4f} | {model_summaries['v2_confounded']['severity_spearman']:.4f} |
| V3（全因子均衡） | {v3_psnr:.3f} dB | {model_summaries['v3_factorial']['severity_mae']:.4f} | {model_summaries['v3_factorial']['severity_spearman']:.4f} |

V3相对V2的配对PSNR变化：{psnr_delta['mean']:+.3f} dB，bootstrap 95%区间 [{psnr_delta['ci_lower']:+.3f}, {psnr_delta['ci_upper']:+.3f}] dB。

只有当V3在该公平测试上通过质量与强度泛化门槛后，才允许进入校准和网页候选；当前报告不能替代公开真实数据结论。
"""
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
