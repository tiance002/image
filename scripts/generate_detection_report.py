"""Build a reproducible, fixed-protocol detection preservation report.

The bundled road scene is a deliberately small smoke-test set: one aligned scene,
four annotated objects, and three deterministic weather variants.  The report is
therefore evidence that the evaluation pipeline works, not a public-benchmark
claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from evaluate_voc_detection import evaluate_class, parse_voc
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as TF
from ultralytics import YOLO

from saferestore.baselines import dark_channel_dehaze
from saferestore.config import load_config
from saferestore.models import calibrated_conservative_fusion
from saferestore.train import build_model

ROOT = Path(__file__).resolve().parents[1]
WEATHER_NAMES = ("fog", "rain", "rain_fog")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def restore_variants(
    output_dir: Path,
    config_path: Path,
    checkpoint_path: Path,
    calibration_path: Path,
) -> tuple[dict[str, Path], list[dict[str, float | str]]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(load_config(config_path)).to(device).eval()
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(payload.get("model", payload))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    fusion_parameters = {
        key: float(value) for key, value in calibration["parameters"].items()
    }

    image_paths: dict[str, Path] = {
        "clear_reference": ROOT / "runs" / "assets" / "clear.jpg"
    }
    timing_rows: list[dict[str, float | str]] = []
    variants_dir = output_dir / "variants"
    variants_dir.mkdir(parents=True, exist_ok=True)

    # Warm up once so the recorded times describe repeated demo use, not model loading.
    with Image.open(ROOT / "runs" / "assets" / "fog.jpg") as sample:
        warmup = TF.to_tensor(sample.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        model(warmup)
    synchronize(device)

    for weather in WEATHER_NAMES:
        source_path = ROOT / "runs" / "assets" / f"{weather}.jpg"
        image_paths[f"{weather}_input"] = source_path
        with Image.open(source_path) as source:
            tensor = TF.to_tensor(source.convert("RGB")).unsqueeze(0).to(device)

        synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = model(tensor)
        synchronize(device)
        restoration_ms = (time.perf_counter() - started) * 1000.0

        raw_conservative_path = variants_dir / f"{weather}_raw_conservative.png"
        TF.to_pil_image(outputs["conservative"][0].cpu()).save(raw_conservative_path)
        image_paths[f"{weather}_raw_conservative"] = raw_conservative_path

        direct_path = variants_dir / f"{weather}_direct.png"
        TF.to_pil_image(outputs["restored"][0].cpu()).save(direct_path)
        image_paths[f"{weather}_direct"] = direct_path

        synchronize(device)
        started = time.perf_counter()
        fused, preservation_weight = calibrated_conservative_fusion(
            outputs["restored"],
            tensor,
            outputs["uncertainty"],
            outputs["degradation"][:, 2],
            **fusion_parameters,
        )
        synchronize(device)
        fusion_ms = (time.perf_counter() - started) * 1000.0
        fused_path = variants_dir / f"{weather}_fused.png"
        TF.to_pil_image(fused[0].cpu()).save(fused_path)
        image_paths[f"{weather}_fused"] = fused_path

        with Image.open(source_path) as source:
            source_uint8 = np.asarray(source.convert("RGB"))
        started = time.perf_counter()
        dcp_uint8, _ = dark_channel_dehaze(source_uint8)
        dcp_ms = (time.perf_counter() - started) * 1000.0
        dcp_path = variants_dir / f"{weather}_dcp.png"
        Image.fromarray(dcp_uint8).save(dcp_path)
        image_paths[f"{weather}_dcp"] = dcp_path

        timing_rows.append(
            {
                "weather": weather,
                "device": str(device),
                "restoration_ms": restoration_ms,
                "fusion_ms": fusion_ms,
                "dcp_ms": dcp_ms,
                "predicted_severity": float(outputs["degradation"][0, 2]),
                "mean_uncertainty": float(outputs["uncertainty"].mean()),
                "mean_preservation_weight": float(preservation_weight.mean()),
            }
        )
    return image_paths, timing_rows


def detect_and_score(
    image_paths: dict[str, Path], output_dir: Path
) -> tuple[dict[str, dict], list[dict[str, float | int | str]]]:
    detector_config = yaml.safe_load(
        (ROOT / "configs" / "detector.yaml").read_text(encoding="utf-8")
    )
    class_names = {int(key): value for key, value in detector_config["classes"].items()}
    targets = parse_voc(ROOT / "runs" / "voc_smoke" / "Annotations" / "clear.xml")
    target_classes = sorted({int(target["class_id"]) for target in targets})
    ground_truth: dict[int, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for target in targets:
        ground_truth[int(target["class_id"])]["scene"].append(target["box"])

    model_path = ROOT / str(detector_config["model"])
    detector = YOLO(str(model_path if model_path.is_file() else detector_config["model"]))
    keys = list(image_paths)
    results = detector.predict(
        source=[str(image_paths[key]) for key in keys],
        imgsz=int(detector_config["image_size"]),
        conf=float(detector_config["confidence"]),
        iou=float(detector_config["iou"]),
        device=detector_config["device"] if torch.cuda.is_available() else "cpu",
        classes=target_classes,
        verbose=False,
    )

    detected_dir = output_dir / "detected"
    detected_dir.mkdir(parents=True, exist_ok=True)
    details: dict[str, dict] = {}
    table_rows: list[dict[str, float | int | str]] = []
    for key, result in zip(keys, results, strict=True):
        predictions: dict[int, list[dict]] = defaultdict(list)
        detections: list[dict] = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                xyxy = np.asarray(box.xyxy[0].cpu(), dtype=np.float32)
                confidence = float(box.conf.item())
                predictions[class_id].append(
                    {"image_id": "scene", "confidence": confidence, "box": xyxy}
                )
                detections.append(
                    {
                        "class": class_names[class_id],
                        "confidence": round(confidence, 6),
                        "xyxy": [round(float(value), 2) for value in xyxy],
                    }
                )

        per_class = {
            class_names[class_id]: evaluate_class(
                predictions[class_id], ground_truth[class_id], 0.5
            )
            for class_id in target_classes
        }
        valid_ap = [metrics["ap50"] for metrics in per_class.values()]
        tp = sum(int(metrics["true_positives"]) for metrics in per_class.values())
        fp = sum(int(metrics["false_positives"]) for metrics in per_class.values())
        fn = sum(int(metrics["false_negatives"]) for metrics in per_class.values())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        map50 = float(np.mean(valid_ap)) if valid_ap else 0.0
        weather, variant = split_key(key)
        row = {
            "weather": weather,
            "variant": variant,
            "map50": map50,
            "micro_precision": precision,
            "micro_recall": recall,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "detections": len(detections),
            "detector_inference_ms": float(result.speed.get("inference", 0.0)),
        }
        table_rows.append(row)
        details[key] = {
            "source": str(image_paths[key].resolve()),
            "summary": row,
            "per_class": per_class,
            "detections": detections,
        }
        cv2.imwrite(str(detected_dir / f"{key}.jpg"), result.plot())
    return details, table_rows


def split_key(key: str) -> tuple[str, str]:
    if key == "clear_reference":
        return "clear", "reference"
    for variant in ("raw_conservative", "input", "direct", "fused", "dcp"):
        suffix = f"_{variant}"
        if key.endswith(suffix):
            return key[: -len(suffix)], variant
    raise ValueError(f"Unknown report key: {key}")


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_board(output_dir: Path) -> None:
    columns = ("input", "dcp", "direct", "raw_conservative", "fused")
    keys = ["clear_reference"]
    keys.extend(f"{weather}_{variant}" for weather in WEATHER_NAMES for variant in columns)
    images: list[tuple[str, Image.Image]] = []
    for key in keys:
        path = output_dir / "detected" / f"{key}.jpg"
        with Image.open(path) as image:
            thumbnail = image.convert("RGB")
            thumbnail.thumbnail((300, 300))
            images.append((key, thumbnail.copy()))
    cell_width, cell_height = 320, 340
    board = Image.new("RGB", (cell_width * 6, cell_height * 3), "white")
    draw = ImageDraw.Draw(board)
    font = ImageFont.load_default()
    positions = {"clear_reference": (0, 0)}
    positions.update(
        {
            f"{weather}_{variant}": (row, column + 1)
            for row, weather in enumerate(WEATHER_NAMES)
            for column, variant in enumerate(columns)
        }
    )
    for label, image in images:
        row, column = positions[label]
        x = column * cell_width + (cell_width - image.width) // 2
        y = row * cell_height + 24
        board.paste(image, (x, y))
        draw.text((column * cell_width + 8, row * cell_height + 6), label, fill="black", font=font)
    draw.text((8, cell_height + 6), "clear reference applies to every aligned row", fill="black", font=font)
    board.save(output_dir / "detection_board.jpg", quality=92)


def write_report(
    output_dir: Path,
    rows: list[dict[str, float | int | str]],
    timing_rows: list[dict[str, float | str]],
    model_label: str,
) -> None:
    by_key = {(str(row["weather"]), str(row["variant"])): row for row in rows}
    lines = [
        "# 固定场景检测保真报告",
        "",
        "## 结论边界",
        "",
        "本报告使用 1 个对齐道路场景、4 个 VOC 目标框（3 person + 1 bus）和 3 种天气退化。",
        "它只验证检测评价链路和复原前后目标保真趋势，不代表公开数据集精度，也不能作为论文最终 mAP。",
        "",
        "## 固定协议",
        "",
        "- 检测器：YOLOv8n，权重和阈值由 `configs/detector.yaml` 固定。",
        "- IoU 判定阈值：0.5；检测置信度阈值：0.25。",
        f"- 复原模型：{model_label}；融合参数仅由对应合成验证集冻结。",
        "- 所有天气图与清晰图像素对齐，共用同一份人工核对的 VOC 标注。",
        "",
        "## 指标表",
        "",
        "| 天气 | 版本 | mAP50 | Micro Precision | Micro Recall | TP | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['weather']} | {row['variant']} | {float(row['map50']):.3f} | "
            f"{float(row['micro_precision']):.3f} | {float(row['micro_recall']):.3f} | "
            f"{row['true_positives']} | {row['false_positives']} | {row['false_negatives']} |"
        )
    lines.extend(["", "## 自动检查", ""])
    for weather in WEATHER_NAMES:
        input_row = by_key[(weather, "input")]
        fused_row = by_key[(weather, "fused")]
        recall_delta = float(fused_row["micro_recall"]) - float(input_row["micro_recall"])
        status = "通过" if recall_delta >= 0.0 else "需关注"
        lines.append(
            f"- {weather}: 融合后召回率相对输入变化 {recall_delta:+.3f}（{status}）。"
        )
    mean_restoration = float(np.mean([float(row["restoration_ms"]) for row in timing_rows]))
    mean_fusion = float(np.mean([float(row["fusion_ms"]) for row in timing_rows]))
    mean_dcp = float(np.mean([float(row["dcp_ms"]) for row in timing_rows]))
    lines.extend(
        [
            "",
            "## 效率（模型预热后）",
            "",
            f"- 平均复原推理：{mean_restoration:.2f} ms/张。",
            f"- 平均校准融合：{mean_fusion:.2f} ms/张。",
            f"- 平均DCP传统处理：{mean_dcp:.2f} ms/张。",
            "- 详细逐图记录见 `timings.csv`；检测时间见 `metrics.csv`。",
            "",
            "## 后续科研要求",
            "",
            "1. 接入至少一个公开真实恶劣天气检测数据集，并保持同一检测协议。",
            "2. 报告不少于 3 个随机种子或 bootstrap 置信区间。",
            "3. 做 direct / raw conservative / calibrated fusion / DCP 的完整消融。",
            "4. 将该单场景结果保留为回归测试，不与正式测试集混合。",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/demo_v2_control.yaml")
    parser.add_argument("--checkpoint", default="runs/demo_v2_control/best.pt")
    parser.add_argument(
        "--fusion-calibration", default="runs/demo_v2_control/fusion_calibration.json"
    )
    parser.add_argument("--output-dir", default="runs/detection_report")
    args = parser.parse_args()
    config_path = ROOT / args.config
    checkpoint_path = ROOT / args.checkpoint
    calibration_path = ROOT / args.fusion_calibration
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths, timing_rows = restore_variants(
        output_dir, config_path, checkpoint_path, calibration_path
    )
    details, table_rows = detect_and_score(image_paths, output_dir)
    save_csv(output_dir / "metrics.csv", table_rows)
    save_csv(output_dir / "timings.csv", timing_rows)
    (output_dir / "details.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    make_board(output_dir)
    model_label = str(load_config(config_path)["project"]["name"])
    write_report(output_dir, table_rows, timing_rows, model_label)
    print(f"Detection preservation report: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
