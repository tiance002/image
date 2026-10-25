"""Evaluate pixel uncertainty and conservative fusion on paired data."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from saferestore.config import load_config
from saferestore.data import PairedWeatherDataset
from saferestore.metrics import (
    calibration_bins,
    expected_calibration_error,
    risk_coverage_curve,
    risk_separation_ratio,
    spearman_correlation,
)
from saferestore.train import build_model


def image_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    difference = prediction.astype(np.float64) - target.astype(np.float64)
    mae = float(np.mean(np.abs(difference)))
    mse = float(np.mean(difference**2))
    psnr = float("inf") if mse == 0.0 else float(10.0 * math.log10(1.0 / mse))
    return {"mae": mae, "psnr_db": psnr}


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def sample_pixels(
    uncertainty: np.ndarray,
    error: np.ndarray,
    *,
    limit: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    uncertainty_flat = uncertainty.reshape(-1)
    error_flat = error.reshape(-1)
    if uncertainty_flat.size <= limit:
        return uncertainty_flat, error_flat
    indices = rng.choice(uncertainty_flat.size, size=limit, replace=False)
    return uncertainty_flat[indices], error_flat[indices]


def mean_metric(rows: list[dict[str, object]], prefix: str, metric: str) -> float:
    return float(np.mean([float(row[f"{prefix}_{metric}"]) for row in rows]))


def summarize_rows(rows: list[dict[str, object]]) -> dict[str, float | int]:
    return {
        "sample_count": len(rows),
        "input_mae_mean": mean_metric(rows, "input", "mae"),
        "input_psnr_mean_db": mean_metric(rows, "input", "psnr_db"),
        "restored_mae_mean": mean_metric(rows, "restored", "mae"),
        "restored_psnr_mean_db": mean_metric(rows, "restored", "psnr_db"),
        "conservative_mae_mean": mean_metric(rows, "conservative", "mae"),
        "conservative_psnr_mean_db": mean_metric(rows, "conservative", "psnr_db"),
        "uncertainty_mean": float(np.mean([float(row["uncertainty_mean"]) for row in rows])),
        "restoration_strength_mean": float(
            np.mean([float(row["restoration_strength"]) for row in rows])
        ),
        "predicted_severity_mean": float(
            np.mean([float(row["predicted_severity"]) for row in rows])
        ),
        "error_spearman_mean": float(
            np.mean([float(row["error_spearman"]) for row in rows])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--checkpoint", default="runs/demo/best.pt")
    parser.add_argument("--manifest", help="Override the test manifest from the config.")
    parser.add_argument("--output-dir", default="runs/evaluation/uncertainty")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--coverage-points", type=int, default=20)
    parser.add_argument("--max-pixels-per-image", type=int, default=50000)
    args = parser.parse_args()

    config = load_config(args.config)
    manifest = args.manifest or config["data"]["test_manifest"]
    image_size = tuple(int(value) for value in config["data"]["image_size"])
    dataset = PairedWeatherDataset(manifest, image_size=image_size)
    device = choose_device(args.device)
    model = build_model(config).to(device).eval()
    payload = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(payload.get("model", payload))

    rng = np.random.default_rng(int(config["project"]["seed"]))
    per_image: list[dict[str, object]] = []
    sampled_uncertainty: list[np.ndarray] = []
    sampled_error: list[np.ndarray] = []

    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset[index]
            weather = sample["weather"].unsqueeze(0).to(device)
            clean = sample["clean"].unsqueeze(0).to(device)
            outputs = model(weather)

            weather_np = weather[0].cpu().numpy()
            clean_np = clean[0].cpu().numpy()
            restored_np = outputs["restored"][0].cpu().numpy()
            conservative_np = outputs["conservative"][0].cpu().numpy()
            uncertainty_np = outputs["uncertainty"][0, 0].cpu().numpy()
            degradation_np = outputs["degradation"][0].cpu().numpy()
            restoration_strength = float(outputs["restoration_strength"][0].mean().cpu())
            error_np = np.mean(np.abs(restored_np - clean_np), axis=0)
            selected_uncertainty, selected_error = sample_pixels(
                uncertainty_np,
                error_np,
                limit=args.max_pixels_per_image,
                rng=rng,
            )
            sampled_uncertainty.append(selected_uncertainty)
            sampled_error.append(selected_error)

            input_metrics = image_metrics(weather_np, clean_np)
            restored_metrics = image_metrics(restored_np, clean_np)
            conservative_metrics = image_metrics(conservative_np, clean_np)
            manifest_row = dataset.rows[index]
            per_image.append(
                {
                    "scene_id": sample["scene_id"],
                    "weather": manifest_row["weather"],
                    "severity": manifest_row["severity"],
                    "severity_target": float(sample["severity"]),
                    "input_mae": input_metrics["mae"],
                    "input_psnr_db": input_metrics["psnr_db"],
                    "restored_mae": restored_metrics["mae"],
                    "restored_psnr_db": restored_metrics["psnr_db"],
                    "conservative_mae": conservative_metrics["mae"],
                    "conservative_psnr_db": conservative_metrics["psnr_db"],
                    "uncertainty_mean": float(uncertainty_np.mean()),
                    "uncertainty_std": float(uncertainty_np.std()),
                    "uncertainty_p90": float(np.quantile(uncertainty_np, 0.9)),
                    "high_risk_fraction": float(np.mean(uncertainty_np >= 0.7)),
                    "error_spearman": spearman_correlation(uncertainty_np, error_np),
                    "predicted_rain": float(degradation_np[0]),
                    "predicted_fog": float(degradation_np[1]),
                    "predicted_severity": float(degradation_np[2]),
                    "predicted_mixed": float(degradation_np[3]),
                    "restoration_strength": restoration_strength,
                }
            )
            print(f"Evaluated {index + 1}/{len(dataset)}: {sample['scene_id']}")

    uncertainty_all = np.concatenate(sampled_uncertainty)
    error_all = np.concatenate(sampled_error)
    bins = calibration_bins(uncertainty_all, error_all, num_bins=args.bins)
    coverage, risk, aurc = risk_coverage_curve(
        uncertainty_all,
        error_all,
        num_points=args.coverage_points,
    )
    weather_names = sorted({str(row["weather"]) for row in per_image})
    by_weather = {
        weather_name: summarize_rows(
            [row for row in per_image if str(row["weather"]) == weather_name]
        )
        for weather_name in weather_names
    }
    restored_mae = mean_metric(per_image, "restored", "mae")
    restored_psnr = mean_metric(per_image, "restored", "psnr_db")
    conservative_mae = mean_metric(per_image, "conservative", "mae")
    conservative_psnr = mean_metric(per_image, "conservative", "psnr_db")
    summary = {
        "result_scope": "synthetic_demo_prototype_only",
        "device": str(device),
        "sample_count": len(per_image),
        "sampled_pixel_count": int(uncertainty_all.size),
        "input": {
            "mae_mean": mean_metric(per_image, "input", "mae"),
            "psnr_mean_db": mean_metric(per_image, "input", "psnr_db"),
        },
        "restored": {
            "mae_mean": restored_mae,
            "psnr_mean_db": restored_psnr,
        },
        "conservative": {
            "mae_mean": conservative_mae,
            "psnr_mean_db": conservative_psnr,
            "mae_delta_vs_restored": conservative_mae - restored_mae,
            "psnr_delta_vs_restored_db": conservative_psnr - restored_psnr,
        },
        "uncertainty": {
            "mean": float(uncertainty_all.mean()),
            "std": float(uncertainty_all.std()),
            "minimum": float(uncertainty_all.min()),
            "maximum": float(uncertainty_all.max()),
            "low_fraction_below_0_05": float(np.mean(uncertainty_all < 0.05)),
            "high_fraction_above_0_95": float(np.mean(uncertainty_all > 0.95)),
            "high_risk_fraction_above_0_7": float(np.mean(uncertainty_all >= 0.7)),
            "spearman_global": spearman_correlation(uncertainty_all, error_all),
            "spearman_per_image_mean": float(
                np.mean([float(row["error_spearman"]) for row in per_image])
            ),
            "expected_calibration_error": expected_calibration_error(bins),
            "risk_coverage_auc": aurc,
            "top_bottom_20_error_ratio": risk_separation_ratio(
                uncertainty_all,
                error_all,
                fraction=0.2,
            ),
        },
        "degradation_gate": {
            "predicted_severity_mae": float(
                np.mean(
                    [
                        abs(float(row["predicted_severity"]) - float(row["severity_target"]))
                        for row in per_image
                    ]
                )
            ),
            "restoration_strength_mean": float(
                np.mean([float(row["restoration_strength"]) for row in per_image])
            ),
            "restoration_strength_minimum": float(
                np.min([float(row["restoration_strength"]) for row in per_image])
            ),
            "restoration_strength_maximum": float(
                np.max([float(row["restoration_strength"]) for row in per_image])
            ),
        },
        "by_weather": by_weather,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (output_dir / "per_image.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)
    with (output_dir / "calibration_bins.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bins[0]))
        writer.writeheader()
        writer.writerows(bins)
    with (output_dir / "risk_coverage.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["coverage", "risk_mae"])
        writer.writeheader()
        writer.writerows(
            {"coverage": float(coverage_value), "risk_mae": float(risk_value)}
            for coverage_value, risk_value in zip(coverage, risk, strict=True)
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved uncertainty evaluation to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
