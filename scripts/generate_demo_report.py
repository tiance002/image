"""Generate reproducible metric tables and visual boards for the two-week demo."""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".cache" / "matplotlib").resolve()))

import cv2
import matplotlib
import numpy as np
import torch
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from saferestore.baselines import dark_channel_dehaze
from saferestore.config import load_config
from saferestore.data import PairedWeatherDataset
from saferestore.losses.restoration import ssim_loss
from saferestore.models import calibrated_conservative_fusion
from saferestore.statistics import bootstrap_mean_interval, paired_bootstrap_difference
from saferestore.train import build_model


METHODS = ("input", "dcp", "restored", "raw_conservative", "calibrated")
METHOD_LABELS = {
    "input": "Weather input",
    "dcp": "DCP baseline",
    "restored": "Learned restoration",
    "raw_conservative": "Raw conservative",
    "calibrated": "Calibrated fusion",
}


def image_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    difference = prediction.astype(np.float64) - target.astype(np.float64)
    mae = float(np.mean(np.abs(difference)))
    mse = float(np.mean(difference**2))
    psnr = float("inf") if mse == 0.0 else float(10.0 * math.log10(1.0 / mse))
    prediction_tensor = torch.from_numpy(prediction).unsqueeze(0)
    target_tensor = torch.from_numpy(target).unsqueeze(0)
    ssim = 1.0 - float(ssim_loss(prediction_tensor, target_tensor))
    return {"mae": mae, "psnr_db": psnr, "ssim": ssim}


def to_uint8_hwc(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return np.rint(array * 255.0).astype(np.uint8)


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def aggregate(rows: list[dict[str, object]], weather: str | None = None) -> list[dict[str, object]]:
    selected = rows if weather is None else [row for row in rows if row["weather"] == weather]
    output: list[dict[str, object]] = []
    for method in METHODS:
        output.append(
            {
                "weather": weather or "overall",
                "method": method,
                "sample_count": len(selected),
                "mae_mean": float(np.mean([float(row[f"{method}_mae"]) for row in selected])),
                "psnr_mean_db": float(
                    np.mean([float(row[f"{method}_psnr_db"]) for row in selected])
                ),
                "ssim_mean": float(
                    np.mean([float(row[f"{method}_ssim"]) for row in selected])
                ),
            }
        )
    return output


def bootstrap_statistics(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    weather_groups: list[str | None] = [None, *sorted({str(row["weather"]) for row in rows})]
    for weather in weather_groups:
        selected = rows if weather is None else [row for row in rows if row["weather"] == weather]
        for method in METHODS:
            for metric in ("mae", "psnr_db", "ssim"):
                values = [float(row[f"{method}_{metric}"]) for row in selected]
                interval = bootstrap_mean_interval(values, seed=42)
                output.append(
                    {
                        "weather": weather or "overall",
                        "method": method,
                        "metric": metric,
                        **interval,
                    }
                )
    return output


def paired_comparisons(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    comparisons = (
        ("restored", "input"),
        ("raw_conservative", "restored"),
        ("calibrated", "restored"),
        ("calibrated", "raw_conservative"),
        ("calibrated", "dcp"),
    )
    output: list[dict[str, object]] = []
    for candidate, reference in comparisons:
        for metric in ("mae", "psnr_db", "ssim"):
            candidate_values = [float(row[f"{candidate}_{metric}"]) for row in rows]
            reference_values = [float(row[f"{reference}_{metric}"]) for row in rows]
            interval = paired_bootstrap_difference(
                candidate_values, reference_values, seed=42
            )
            output.append(
                {
                    "candidate": candidate,
                    "reference": reference,
                    "metric": metric,
                    **interval,
                }
            )
    return output


def render_case_board(
    records: list[dict[str, object]],
    output_path: Path,
    *,
    title: str,
) -> None:
    columns = (
        "input",
        "dcp",
        "restored",
        "raw_conservative",
        "calibrated",
        "uncertainty",
        "clean",
    )
    labels = (
        "Weather input",
        "DCP",
        "Learned restoration",
        "Raw conservative",
        "Calibrated fusion",
        "Uncertainty",
        "Clean target",
    )
    figure, axes = plt.subplots(len(records), len(columns), figsize=(21, 3.2 * len(records)))
    axes = np.atleast_2d(axes)
    for row_index, record in enumerate(records):
        for column_index, (key, label) in enumerate(zip(columns, labels, strict=True)):
            axis = axes[row_index, column_index]
            axis.imshow(record[key])
            if row_index == 0:
                axis.set_title(label, fontsize=11)
            axis.axis("off")
        delta_input = float(record["calibrated_psnr_db"]) - float(record["input_psnr_db"])
        delta_direct = float(record["calibrated_psnr_db"]) - float(record["restored_psnr_db"])
        axes[row_index, 0].text(
            -0.04,
            0.5,
            f"{record['scene_id']}\n{record['weather']}\n"
            f"vs input {delta_input:+.2f} dB\nvs direct {delta_direct:+.2f} dB",
            transform=axes[row_index, 0].transAxes,
            horizontalalignment="right",
            verticalalignment="center",
            fontsize=10,
        )
    figure.suptitle(title, fontsize=16)
    figure.tight_layout(rect=(0.08, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def render_metric_chart(summary_rows: list[dict[str, object]], output_path: Path) -> None:
    groups = ["overall", "fog", "rain", "rain_fog"]
    x_positions = np.arange(len(groups))
    width = 0.19
    figure, axis = plt.subplots(figsize=(12, 6.5))
    for method_index, method in enumerate(METHODS):
        values = [
            next(
                float(row["psnr_mean_db"])
                for row in summary_rows
                if row["weather"] == group and row["method"] == method
            )
            for group in groups
        ]
        axis.bar(
            x_positions + (method_index - 1.5) * width,
            values,
            width,
            label=METHOD_LABELS[method],
        )
    axis.set_title("Frozen test-set PSNR by weather type")
    axis.set_ylabel("PSNR (dB, higher is better)")
    axis.set_xticks(x_positions, ["Overall", "Fog", "Rain", "Rain + Fog"])
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=2)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/demo_v2_control.yaml")
    parser.add_argument("--checkpoint", default="runs/demo_v2_control/best.pt")
    parser.add_argument(
        "--fusion-calibration", default="runs/demo_v2_control/fusion_calibration.json"
    )
    parser.add_argument("--output-dir", default="runs/demo_report")
    parser.add_argument("--case-count", type=int, default=3)
    args = parser.parse_args()

    config = load_config(args.config)
    calibration = json.loads(Path(args.fusion_calibration).read_text(encoding="utf-8"))
    fusion_parameters = {
        name: float(value) for name, value in calibration["parameters"].items()
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device).eval()
    payload = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(payload.get("model", payload))
    image_size = tuple(int(value) for value in config["data"]["image_size"])
    dataset = PairedWeatherDataset(config["data"]["test_manifest"], image_size=image_size)
    output_dir = Path(args.output_dir)
    sample_dir = output_dir / "samples"
    rows: list[dict[str, object]] = []
    visual_records: list[dict[str, object]] = []

    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset[index]
            weather = sample["weather"].unsqueeze(0).to(device)
            clean = sample["clean"].unsqueeze(0).to(device)
            outputs = model(weather)
            calibrated, preservation_weight = calibrated_conservative_fusion(
                outputs["restored"],
                weather,
                outputs["uncertainty"],
                outputs["degradation"][:, 2],
                **fusion_parameters,
            )
            input_uint8 = to_uint8_hwc(weather[0])
            dcp_uint8, _ = dark_channel_dehaze(input_uint8)
            restored_uint8 = to_uint8_hwc(outputs["restored"][0])
            raw_conservative_uint8 = to_uint8_hwc(outputs["conservative"][0])
            calibrated_uint8 = to_uint8_hwc(calibrated[0])
            clean_uint8 = to_uint8_hwc(clean[0])
            uncertainty_uint8 = np.rint(
                outputs["uncertainty"][0, 0].cpu().numpy() * 255.0
            ).astype(np.uint8)
            uncertainty_heatmap = cv2.cvtColor(
                cv2.applyColorMap(uncertainty_uint8, cv2.COLORMAP_TURBO),
                cv2.COLOR_BGR2RGB,
            )
            float_images = {
                "input": weather[0].cpu().numpy(),
                "dcp": dcp_uint8.astype(np.float32).transpose(2, 0, 1) / 255.0,
                "restored": outputs["restored"][0].cpu().numpy(),
                "raw_conservative": outputs["conservative"][0].cpu().numpy(),
                "calibrated": calibrated[0].cpu().numpy(),
            }
            clean_float = clean[0].cpu().numpy()
            row: dict[str, object] = {
                "scene_id": sample["scene_id"],
                "weather": dataset.rows[index]["weather"],
                "severity": dataset.rows[index]["severity"],
                "predicted_severity": float(outputs["degradation"][0, 2].cpu()),
                "uncertainty_mean": float(outputs["uncertainty"].mean().cpu()),
                "preservation_weight_mean": float(preservation_weight.mean().cpu()),
            }
            for method, prediction in float_images.items():
                metrics = image_metrics(prediction, clean_float)
                for metric_name, value in metrics.items():
                    row[f"{method}_{metric_name}"] = value
            row["calibrated_delta_vs_input_db"] = float(row["calibrated_psnr_db"]) - float(
                row["input_psnr_db"]
            )
            row["calibrated_delta_vs_direct_db"] = float(
                row["calibrated_psnr_db"]
            ) - float(row["restored_psnr_db"])
            rows.append(row)

            scene_id = str(sample["scene_id"])
            images = {
                "input": input_uint8,
                "dcp": dcp_uint8,
                "restored": restored_uint8,
                "raw_conservative": raw_conservative_uint8,
                "calibrated": calibrated_uint8,
                "uncertainty": uncertainty_heatmap,
                "clean": clean_uint8,
            }
            for name, image in images.items():
                save_rgb(sample_dir / f"{scene_id}_{name}.png", image)
            visual_records.append({**row, **images})
            print(f"Prepared report sample {index + 1}/{len(dataset)}: {scene_id}")

    weather_names = sorted({str(row["weather"]) for row in rows})
    summary_rows = aggregate(rows)
    for weather_name in weather_names:
        summary_rows.extend(aggregate(rows, weather_name))
    write_csv(output_dir / "per_image_metrics.csv", rows)
    write_csv(output_dir / "method_summary.csv", aggregate(rows))
    write_csv(output_dir / "weather_summary.csv", summary_rows)
    bootstrap_rows = bootstrap_statistics(rows)
    comparison_rows = paired_comparisons(rows)
    write_csv(output_dir / "bootstrap_intervals.csv", bootstrap_rows)
    write_csv(output_dir / "paired_comparisons.csv", comparison_rows)

    success_records = sorted(
        visual_records,
        key=lambda row: float(row["calibrated_delta_vs_input_db"]),
        reverse=True,
    )[: args.case_count]
    failure_records = sorted(
        visual_records,
        key=lambda row: float(row["calibrated_delta_vs_direct_db"]),
    )[: args.case_count]
    render_case_board(
        success_records,
        output_dir / "boards" / "success_cases.png",
        title="Strongest restoration gains on the frozen prototype test set",
    )
    render_case_board(
        failure_records,
        output_dir / "boards" / "failure_cases.png",
        title="Least favorable calibrated-fusion cases (shown without filtering)",
    )
    render_metric_chart(summary_rows, output_dir / "boards" / "weather_psnr.png")

    selection = {
        "success_cases": [str(record["scene_id"]) for record in success_records],
        "failure_cases": [str(record["scene_id"]) for record in failure_records],
        "success_rule": "largest calibrated PSNR gain over weather input",
        "failure_rule": "smallest calibrated PSNR change relative to direct restoration",
    }
    (output_dir / "case_selection.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    overall = {row["method"]: row for row in aggregate(rows)}
    calibrated_vs_direct = next(
        row
        for row in comparison_rows
        if row["candidate"] == "calibrated"
        and row["reference"] == "restored"
        and row["metric"] == "psnr_db"
    )
    calibrated_vs_raw = next(
        row
        for row in comparison_rows
        if row["candidate"] == "calibrated"
        and row["reference"] == "raw_conservative"
        and row["metric"] == "psnr_db"
    )
    conclusions = {
        "scope": "synthetic_frozen_test_prototype_only",
        "sample_count": len(rows),
        "bootstrap_confidence": 0.95,
        "bootstrap_resamples": 10_000,
        "calibrated_vs_direct_psnr": calibrated_vs_direct,
        "calibrated_vs_raw_conservative_psnr": calibrated_vs_raw,
        "calibrated_beats_direct_with_95pct_ci": float(calibrated_vs_direct["ci_lower"]) > 0.0,
        "calibrated_beats_raw_with_95pct_ci": float(calibrated_vs_raw["ci_lower"]) > 0.0,
    }
    (output_dir / "ablation_conclusions.json").write_text(
        json.dumps(conclusions, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    project_name = str(config["project"]["name"])
    report = f"""# SafeRestore-RF 展示结果包

实验配置：`{project_name}`。

生成范围：{len(rows)}张工程合成冻结测试样本，仅用于原型展示。

| 方法 | MAE | PSNR | SSIM |
|---|---:|---:|---:|
| 恶劣天气输入 | {overall['input']['mae_mean']:.4f} | {overall['input']['psnr_mean_db']:.3f} dB | {overall['input']['ssim_mean']:.4f} |
| DCP传统基线 | {overall['dcp']['mae_mean']:.4f} | {overall['dcp']['psnr_mean_db']:.3f} dB | {overall['dcp']['ssim_mean']:.4f} |
| 学习式直接复原 | {overall['restored']['mae_mean']:.4f} | {overall['restored']['psnr_mean_db']:.3f} dB | {overall['restored']['ssim_mean']:.4f} |
| 原始保守融合 | {overall['raw_conservative']['mae_mean']:.4f} | {overall['raw_conservative']['psnr_mean_db']:.3f} dB | {overall['raw_conservative']['ssim_mean']:.4f} |
| 验证集校准融合 | {overall['calibrated']['mae_mean']:.4f} | {overall['calibrated']['psnr_mean_db']:.3f} dB | {overall['calibrated']['ssim_mean']:.4f} |

校准融合相对直接复原的配对PSNR变化为 {calibrated_vs_direct['mean']:+.3f} dB，bootstrap 95%置信区间为 [{calibrated_vs_direct['ci_lower']:+.3f}, {calibrated_vs_direct['ci_upper']:+.3f}] dB。

校准融合相对原始保守融合的配对PSNR变化为 {calibrated_vs_raw['mean']:+.3f} dB，bootstrap 95%置信区间为 [{calibrated_vs_raw['ci_lower']:+.3f}, {calibrated_vs_raw['ci_upper']:+.3f}] dB。

成功案例按照“相对恶劣天气输入的PSNR提升”自动选择；失败案例按照“相对直接复原最不利的PSNR变化”自动选择，未进行人工挑选。

本结果包不能作为正式论文结论。正式结果必须替换为许可明确的公开数据，并补充检测指标、多随机种子和真实天气泛化实验。
"""
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved demo report package to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
