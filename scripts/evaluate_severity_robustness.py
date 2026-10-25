"""Evaluate restoration robustness on continuous, including unseen, severity levels."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import functional as TF

if __package__:
    from .generate_demo_data import add_fog, add_rain, clean_scene
else:
    from generate_demo_data import add_fog, add_rain, clean_scene

from saferestore.config import load_config
from saferestore.losses.restoration import ssim_loss
from saferestore.metrics import spearman_correlation
from saferestore.models import calibrated_conservative_fusion
from saferestore.statistics import bootstrap_mean_interval
from saferestore.train import build_model

ROOT = Path(__file__).resolve().parents[1]
WEATHERS = ("rain", "fog", "rain_fog")


def metrics(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, float, float]:
    difference = prediction.double() - target.double()
    mae = float(difference.abs().mean())
    mse = float((difference**2).mean())
    psnr = float("inf") if mse == 0.0 else 10.0 * math.log10(1.0 / mse)
    ssim = 1.0 - float(ssim_loss(prediction.unsqueeze(0), target.unsqueeze(0)))
    return mae, psnr, ssim


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v3_factorial_seed42.yaml")
    parser.add_argument("--checkpoint", default="runs/v3_factorial_seed42/best.pt")
    parser.add_argument(
        "--fusion-calibration",
        default="runs/v3_factorial_seed42/fusion_calibration.json",
    )
    parser.add_argument("--output-dir", default="runs/v3_severity_robustness")
    parser.add_argument("--scene-count", type=int, default=6)
    parser.add_argument(
        "--severities", default="0.1,0.2,0.3,0.45,0.6,0.75,0.9,1.0"
    )
    args = parser.parse_args()
    severities = [float(value) for value in args.severities.split(",")]
    if not severities or any(value < 0.0 or value > 1.0 for value in severities):
        raise ValueError("severities must be non-empty values in [0, 1]")

    config = load_config(ROOT / args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device).eval()
    payload = torch.load(ROOT / args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(payload.get("model", payload))
    calibration = json.loads((ROOT / args.fusion_calibration).read_text(encoding="utf-8"))
    fusion_parameters = {
        key: float(value) for key, value in calibration["parameters"].items()
    }
    rows: list[dict] = []
    with torch.inference_mode():
        for scene_index in range(args.scene_count):
            clean_image = clean_scene(50_000 + scene_index)
            clean = TF.to_tensor(clean_image).to(device)
            for weather_name in WEATHERS:
                for severity in severities:
                    degraded = clean_image
                    if weather_name in {"fog", "rain_fog"}:
                        degraded = add_fog(degraded, severity)
                    if weather_name in {"rain", "rain_fog"}:
                        degraded = add_rain(
                            degraded,
                            severity,
                            seed=60_000 + scene_index * 100 + int(severity * 100),
                        )
                    weather = TF.to_tensor(degraded).to(device)
                    outputs = model(weather.unsqueeze(0))
                    fused, _ = calibrated_conservative_fusion(
                        outputs["restored"],
                        weather.unsqueeze(0),
                        outputs["uncertainty"],
                        outputs["degradation"][:, 2],
                        **fusion_parameters,
                    )
                    input_mae, input_psnr, input_ssim = metrics(weather, clean)
                    direct_mae, direct_psnr, direct_ssim = metrics(
                        outputs["restored"][0], clean
                    )
                    fused_mae, fused_psnr, fused_ssim = metrics(fused[0], clean)
                    rows.append(
                        {
                            "scene": scene_index,
                            "weather": weather_name,
                            "severity": severity,
                            "predicted_severity": float(outputs["degradation"][0, 2]),
                            "uncertainty_mean": float(outputs["uncertainty"].mean()),
                            "input_mae": input_mae,
                            "input_psnr_db": input_psnr,
                            "input_ssim": input_ssim,
                            "direct_mae": direct_mae,
                            "direct_psnr_db": direct_psnr,
                            "direct_ssim": direct_ssim,
                            "psnr_gain_db": direct_psnr - input_psnr,
                            "fused_mae": fused_mae,
                            "fused_psnr_db": fused_psnr,
                            "fused_ssim": fused_ssim,
                            "fused_psnr_gain_db": fused_psnr - input_psnr,
                        }
                    )
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "per_sample.csv", rows)

    summary_rows: list[dict] = []
    weather_results: dict[str, dict] = {}
    for weather_name in WEATHERS:
        weather_rows = [row for row in rows if row["weather"] == weather_name]
        severity_values = np.asarray([row["severity"] for row in weather_rows])
        predicted_values = np.asarray([row["predicted_severity"] for row in weather_rows])
        uncertainty_values = np.asarray([row["uncertainty_mean"] for row in weather_rows])
        weather_results[weather_name] = {
            "severity_spearman": spearman_correlation(severity_values, predicted_values),
            "uncertainty_severity_spearman": spearman_correlation(
                severity_values, uncertainty_values
            ),
            "mean_psnr_gain_db": float(
                np.mean([row["psnr_gain_db"] for row in weather_rows])
            ),
            "mean_fused_psnr_gain_db": float(
                np.mean([row["fused_psnr_gain_db"] for row in weather_rows])
            ),
            "negative_gain_samples": sum(row["psnr_gain_db"] < 0.0 for row in weather_rows),
            "negative_fused_gain_samples": sum(
                row["fused_psnr_gain_db"] < 0.0 for row in weather_rows
            ),
            "sample_count": len(weather_rows),
        }
        for severity in severities:
            selected = [
                row
                for row in weather_rows
                if math.isclose(float(row["severity"]), severity)
            ]
            gain_interval = bootstrap_mean_interval(
                [row["psnr_gain_db"] for row in selected], resamples=5_000, seed=42
            )
            fused_gain_interval = bootstrap_mean_interval(
                [row["fused_psnr_gain_db"] for row in selected],
                resamples=5_000,
                seed=42,
            )
            summary_rows.append(
                {
                    "weather": weather_name,
                    "severity": severity,
                    "input_psnr_mean_db": float(
                        np.mean([row["input_psnr_db"] for row in selected])
                    ),
                    "direct_psnr_mean_db": float(
                        np.mean([row["direct_psnr_db"] for row in selected])
                    ),
                    "gain_mean_db": gain_interval["mean"],
                    "gain_ci_lower_db": gain_interval["ci_lower"],
                    "gain_ci_upper_db": gain_interval["ci_upper"],
                    "fused_psnr_mean_db": float(
                        np.mean([row["fused_psnr_db"] for row in selected])
                    ),
                    "fused_gain_mean_db": fused_gain_interval["mean"],
                    "fused_gain_ci_lower_db": fused_gain_interval["ci_lower"],
                    "fused_gain_ci_upper_db": fused_gain_interval["ci_upper"],
                    "predicted_severity_mean": float(
                        np.mean([row["predicted_severity"] for row in selected])
                    ),
                    "uncertainty_mean": float(
                        np.mean([row["uncertainty_mean"] for row in selected])
                    ),
                }
            )
    write_csv(output_dir / "severity_summary.csv", summary_rows)
    results = {
        "scope": "continuous_synthetic_severity_stress_test_only",
        "model": str(config["project"]["name"]),
        "scene_count": args.scene_count,
        "severity_levels": severities,
        "weather_results": weather_results,
    }
    (output_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# 连续退化强度鲁棒性报告",
        "",
        f"模型：`{config['project']['name']}`；{args.scene_count}个新场景，{len(severities)}个连续强度，合计{len(rows)}次推理。",
        "",
        "| 天气 | 强度预测Spearman | 不确定性-强度Spearman | 直接复原增益 | 校准融合增益 | 直接/融合负增益 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for weather_name in WEATHERS:
        item = weather_results[weather_name]
        lines.append(
            f"| {weather_name} | {item['severity_spearman']:.3f} | "
            f"{item['uncertainty_severity_spearman']:.3f} | "
            f"{item['mean_psnr_gain_db']:+.3f} dB | "
            f"{item['mean_fused_psnr_gain_db']:+.3f} dB | "
            f"{item['negative_gain_samples']}/{item['negative_fused_gain_samples']}（共{item['sample_count']}） |"
        )
    lines.extend(
        [
            "",
            "该扫描包含训练离散等级之间和边界处的强度，用于发现插值与极端退化问题；仍不等同于真实天气泛化。",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
