"""Select conservative-fusion parameters on validation data, then evaluate once on test."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch

from saferestore.config import load_config
from saferestore.data import PairedWeatherDataset
from saferestore.train import build_model


def parse_grid(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Grid must contain at least one number.")
    return values


def collect_predictions(
    model: torch.nn.Module,
    dataset: PairedWeatherDataset,
    device: torch.device,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset[index]
            weather = sample["weather"].unsqueeze(0).to(device)
            clean = sample["clean"].unsqueeze(0).to(device)
            outputs = model(weather)
            records.append(
                {
                    "scene_id": sample["scene_id"],
                    "weather_name": dataset.rows[index]["weather"],
                    "weather": weather[0].cpu().numpy(),
                    "clean": clean[0].cpu().numpy(),
                    "restored": outputs["restored"][0].cpu().numpy(),
                    "raw_conservative": outputs["conservative"][0].cpu().numpy(),
                    "uncertainty": outputs["uncertainty"][0].cpu().numpy(),
                    "predicted_severity": float(outputs["degradation"][0, 2].cpu()),
                }
            )
    return records


def fuse_numpy(record: dict[str, object], parameters: dict[str, float]) -> np.ndarray:
    severity = float(record["predicted_severity"])
    exponent = np.clip(
        (severity - parameters["severity_threshold"])
        / parameters["severity_temperature"],
        -60.0,
        60.0,
    )
    image_gate = 1.0 / (1.0 + math.exp(float(exponent)))
    uncertainty = np.asarray(record["uncertainty"])
    pixel_gate = np.clip(
        (uncertainty - parameters["uncertainty_threshold"])
        / (1.0 - parameters["uncertainty_threshold"]),
        0.0,
        1.0,
    ) ** parameters["power"]
    weight = np.clip(parameters["strength"] * image_gate * pixel_gate, 0.0, 1.0)
    restored = np.asarray(record["restored"])
    weather = np.asarray(record["weather"])
    return (1.0 - weight) * restored + weight * weather


def mae(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(prediction.astype(np.float64) - target.astype(np.float64))))


def psnr(prediction: np.ndarray, target: np.ndarray) -> float:
    difference = prediction.astype(np.float64) - target.astype(np.float64)
    mse = float(np.mean(difference**2))
    return float("inf") if mse == 0.0 else float(10.0 * math.log10(1.0 / mse))


def evaluate_records(
    records: list[dict[str, object]],
    parameters: dict[str, float] | None,
    source: str = "calibrated",
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for record in records:
        if source == "direct":
            prediction = np.asarray(record["restored"])
        elif source == "raw_conservative":
            prediction = np.asarray(record["raw_conservative"])
        else:
            if parameters is None:
                raise ValueError("Calibrated evaluation requires parameters.")
            prediction = fuse_numpy(record, parameters)
        target = np.asarray(record["clean"])
        rows.append(
            {
                "scene_id": record["scene_id"],
                "weather": record["weather_name"],
                "mae": mae(prediction, target),
                "psnr_db": psnr(prediction, target),
            }
        )
    weather_names = sorted({str(row["weather"]) for row in rows})
    by_weather = {
        name: {
            "sample_count": sum(str(row["weather"]) == name for row in rows),
            "mae_mean": float(
                np.mean([float(row["mae"]) for row in rows if str(row["weather"]) == name])
            ),
            "psnr_mean_db": float(
                np.mean(
                    [float(row["psnr_db"]) for row in rows if str(row["weather"]) == name]
                )
            ),
        }
        for name in weather_names
    }
    return {
        "mae_mean": float(np.mean([float(row["mae"]) for row in rows])),
        "psnr_mean_db": float(np.mean([float(row["psnr_db"]) for row in rows])),
        "macro_weather_mae": float(
            np.mean([float(metrics["mae_mean"]) for metrics in by_weather.values()])
        ),
        "by_weather": by_weather,
        "per_image": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/demo_v2_control.yaml")
    parser.add_argument("--checkpoint", default="runs/demo_v2_control/best.pt")
    parser.add_argument("--output", default="runs/demo_v2_control/fusion_calibration.json")
    parser.add_argument(
        "--grid-output", default="runs/evaluation/fusion_calibration_grid.csv"
    )
    parser.add_argument("--severity-thresholds", type=parse_grid, default=parse_grid("0.54,0.56,0.58,0.60,0.62"))
    parser.add_argument("--severity-temperatures", type=parse_grid, default=parse_grid("0.01,0.02,0.04,0.08"))
    parser.add_argument("--uncertainty-thresholds", type=parse_grid, default=parse_grid("0.0,0.15,0.2,0.25,0.3"))
    parser.add_argument("--strengths", type=parse_grid, default=parse_grid("0.0,0.25,0.5,0.75,1.0"))
    parser.add_argument("--powers", type=parse_grid, default=parse_grid("0.5,1.0,2.0"))
    parser.add_argument(
        "--max-weather-mae-regression",
        type=float,
        default=0.0005,
        help="Maximum allowed validation MAE increase for any weather group.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device).eval()
    payload = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(payload.get("model", payload))
    image_size = tuple(int(value) for value in config["data"]["image_size"])
    validation = PairedWeatherDataset(config["data"]["val_manifest"], image_size=image_size)
    test = PairedWeatherDataset(config["data"]["test_manifest"], image_size=image_size)
    validation_records = collect_predictions(model, validation, device)
    test_records = collect_predictions(model, test, device)

    validation_direct = evaluate_records(validation_records, None, source="direct")
    validation_raw = evaluate_records(validation_records, None, source="raw_conservative")
    grid_rows: list[dict[str, float | bool]] = []
    best_parameters: dict[str, float] | None = None
    best_metrics: dict[str, object] | None = None
    best_score = float("inf")
    parameter_grid = itertools.product(
        args.severity_thresholds,
        args.severity_temperatures,
        args.uncertainty_thresholds,
        args.strengths,
        args.powers,
    )
    for severity_threshold, severity_temperature, uncertainty_threshold, strength, power in parameter_grid:
        parameters = {
            "severity_threshold": severity_threshold,
            "severity_temperature": severity_temperature,
            "uncertainty_threshold": uncertainty_threshold,
            "strength": strength,
            "power": power,
        }
        metrics = evaluate_records(validation_records, parameters)
        weather_regressions = [
            float(metrics["by_weather"][name]["mae_mean"])
            - float(validation_direct["by_weather"][name]["mae_mean"])
            for name in validation_direct["by_weather"]
        ]
        feasible = max(weather_regressions) <= args.max_weather_mae_regression
        score = float(metrics["macro_weather_mae"])
        grid_rows.append(
            {
                **parameters,
                "validation_mae": float(metrics["mae_mean"]),
                "validation_psnr_db": float(metrics["psnr_mean_db"]),
                "validation_macro_weather_mae": score,
                "max_weather_mae_regression": max(weather_regressions),
                "feasible": feasible,
            }
        )
        if feasible and score < best_score:
            best_score = score
            best_parameters = parameters
            best_metrics = metrics

    if best_parameters is None or best_metrics is None:
        raise RuntimeError("No fusion candidate satisfied the validation constraints.")

    test_direct = evaluate_records(test_records, None, source="direct")
    test_raw = evaluate_records(test_records, None, source="raw_conservative")
    test_calibrated = evaluate_records(test_records, best_parameters)
    result = {
        "result_scope": "synthetic_demo_prototype_only",
        "selection_split": "validation",
        "selection_objective": "minimum_macro_weather_mae_with_per_weather_constraint",
        "max_weather_mae_regression": args.max_weather_mae_regression,
        "parameters": best_parameters,
        "validation": {
            "direct": validation_direct,
            "raw_conservative": validation_raw,
            "calibrated": best_metrics,
        },
        "test_frozen_once": {
            "direct": test_direct,
            "raw_conservative": test_raw,
            "calibrated": test_calibrated,
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    grid_path = Path(args.grid_output)
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    with grid_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(grid_rows[0]))
        writer.writeheader()
        writer.writerows(sorted(grid_rows, key=lambda row: float(row["validation_macro_weather_mae"])))
    compact = {
        "parameters": best_parameters,
        "validation": {
            name: {key: value for key, value in metrics.items() if key != "per_image"}
            for name, metrics in result["validation"].items()
        },
        "test_frozen_once": {
            name: {key: value for key, value in metrics.items() if key != "per_image"}
            for name, metrics in result["test_frozen_once"].items()
        },
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))
    print(f"Saved frozen fusion calibration to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
