"""Evaluate three V3 training seeds on one frozen balanced test set."""

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
from saferestore.statistics import bootstrap_mean_interval
from saferestore.train import build_model

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)


def quality_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    difference = prediction.double() - target.double()
    mae = float(difference.abs().mean())
    mse = float((difference**2).mean())
    psnr = float("inf") if mse == 0.0 else 10.0 * math.log10(1.0 / mse)
    ssim = 1.0 - float(ssim_loss(prediction.unsqueeze(0), target.unsqueeze(0)))
    return {"mae": mae, "psnr_db": psnr, "ssim": ssim}


def minimum_validation_loss(history_path: Path) -> tuple[int, float]:
    records = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    best = min(records, key=lambda record: float(record["val"]["total"]))
    return int(best["epoch"]), float(best["val"]["total"])


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    output_dir = ROOT / "runs" / "v3_multiseed"
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT / "configs" / "v3_factorial_seed42.yaml")
    image_size = tuple(int(value) for value in config["data"]["image_size"])
    dataset = PairedWeatherDataset(config["data"]["test_manifest"], image_size=image_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    per_image_rows: list[dict] = []
    per_seed_rows: list[dict] = []
    for seed in SEEDS:
        run_dir = ROOT / "runs" / f"v3_factorial_seed{seed}"
        checkpoint_path = run_dir / "best.pt"
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model = build_model(payload.get("config", config)).to(device).eval()
        model.load_state_dict(payload.get("model", payload))
        seed_rows: list[dict] = []
        with torch.inference_mode():
            for index in range(len(dataset)):
                sample = dataset[index]
                weather = sample["weather"].to(device)
                clean = sample["clean"].to(device)
                outputs = model(weather.unsqueeze(0))
                metrics = quality_metrics(outputs["restored"][0].cpu(), clean.cpu())
                row = {
                    "seed": seed,
                    "scene_id": sample["scene_id"],
                    "weather": dataset.rows[index]["weather"],
                    "severity": dataset.rows[index]["severity"],
                    "severity_target": float(sample["severity"]),
                    "severity_predicted": float(outputs["degradation"][0, 2]),
                    "uncertainty_mean": float(outputs["uncertainty"].mean()),
                    **metrics,
                }
                seed_rows.append(row)
                per_image_rows.append(row)
        best_epoch, best_val_loss = minimum_validation_loss(run_dir / "history.jsonl")
        severity_targets = np.asarray([row["severity_target"] for row in seed_rows])
        severity_predictions = np.asarray([row["severity_predicted"] for row in seed_rows])
        per_seed_rows.append(
            {
                "seed": seed,
                "best_epoch": best_epoch,
                "best_validation_loss": best_val_loss,
                "test_mae": float(np.mean([row["mae"] for row in seed_rows])),
                "test_psnr_db": float(np.mean([row["psnr_db"] for row in seed_rows])),
                "test_ssim": float(np.mean([row["ssim"] for row in seed_rows])),
                "severity_mae": float(np.mean(np.abs(severity_predictions - severity_targets))),
                "severity_spearman": spearman_correlation(
                    severity_predictions, severity_targets
                ),
            }
        )
    write_csv(output_dir / "per_image_metrics.csv", per_image_rows)
    write_csv(output_dir / "per_seed_summary.csv", per_seed_rows)

    aggregate: dict[str, dict[str, float | int]] = {}
    for metric in ("test_mae", "test_psnr_db", "test_ssim", "severity_mae", "severity_spearman"):
        values = np.asarray([float(row[metric]) for row in per_seed_rows])
        aggregate[metric] = {
            "seed_count": len(values),
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    scene_bootstrap = bootstrap_mean_interval(
        [float(row["psnr_db"]) for row in per_image_rows], seed=42
    )
    selected = min(per_seed_rows, key=lambda row: float(row["best_validation_loss"]))
    results = {
        "scope": "three_seed_balanced_factorial_synthetic_only",
        "selection_rule": "minimum validation total loss; test metrics not used for selection",
        "selected_seed": int(selected["seed"]),
        "selected_checkpoint": str(
            (ROOT / "runs" / f"v3_factorial_seed{selected['seed']}" / "best.pt").resolve()
        ),
        "aggregate_across_seeds": aggregate,
        "pooled_scene_bootstrap_psnr": scene_bootstrap,
    }
    (output_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_lines = [
        "# V3三随机种子复现报告",
        "",
        "范围：全因子合成冻结测试集；模型选择仅依据验证损失，不依据测试指标。",
        "",
        "| Seed | 最佳epoch | 验证损失 | 测试PSNR | SSIM | 强度MAE | 强度Spearman |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in per_seed_rows:
        report_lines.append(
            f"| {row['seed']} | {row['best_epoch']} | {row['best_validation_loss']:.4f} | "
            f"{row['test_psnr_db']:.3f} dB | {row['test_ssim']:.4f} | "
            f"{row['severity_mae']:.4f} | {row['severity_spearman']:.4f} |"
        )
    report_lines.extend(
        [
            "",
            f"三种子PSNR：{aggregate['test_psnr_db']['mean']:.3f} ± {aggregate['test_psnr_db']['sample_std']:.3f} dB（均值±样本标准差）。",
            f"按验证损失选择seed {selected['seed']}，其测试指标只用于最终报告。",
            "",
            "该报告验证工程复现性，但合成数据不能替代公开真实天气数据。",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
