"""诊断：复原为何在真实雨雾上损害检测（归因实验）。

目的
----
在固定的 ACDC 200 张 fog/rain 验证集上，对已经生成好的复原图做三组
"只改一处"的对照，判断检测退化的主导原因是哪一类：

1. color_matched：把复原图的 LAB 全局均值/方差强制拉回输入图统计量，
   其余结构完全不动。用于检验"全局色偏/统计分布漂移"假设。
2. luma_matched：只对齐亮度通道 L，保留复原图自身的色彩。
   用于区分"亮度分布"与"色度分布"各自的贡献。
3. blend50：直接 0.5*输入 + 0.5*复原，不做任何色彩修正。
   如果它明显好于 color_matched，说明主导因素是高通细节被破坏，
   而不是颜色。

同时会重跑一次 weather_input 作为链路自检；只有自检能复现报告中的
0.2042 左右，后面的对照才可信。

用法：
    python scripts/diagnose_detection_rootcause.py --split-dir runs/acdc_benchmark
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402

from scripts.evaluate_coco_detection import evaluate_class  # noqa: E402


def lab_stat_transfer(source: np.ndarray, reference: np.ndarray, mode: str) -> np.ndarray:
    """把 source 的 LAB 统计量迁移到 reference 的统计量上。

    参数里的 source 是"要被修正的图"（复原图），reference 是"统计量来源"
    （输入的恶劣天气图）。mode 为 "lab" 时三个通道全部对齐，"luma" 只对齐 L。
    两张图都必须是 uint8 RGB。
    """
    src_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB).astype(np.float32)
    channels = 3 if mode == "lab" else 1
    for channel in range(channels):
        src = src_lab[:, :, channel]
        ref = ref_lab[:, :, channel]
        src_std = float(src.std()) + 1e-6
        ref_std = float(ref.std())
        src_lab[:, :, channel] = (src - float(src.mean())) / src_std * ref_std + float(ref.mean())
    # 8 位 LAB 的合法范围就是 [0, 255]
    return cv2.cvtColor(np.clip(src_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def run_detector(
    model: YOLO, image_rgb: np.ndarray, config: dict, target_classes: list[int]
) -> dict[int, list[dict]]:
    """按冻结协议跑一次检测，返回 {类别id: [{box, confidence}]}。

    注意两处协议细节，写错会让指标整体偏低且很难察觉：
    1. Ultralytics 对 numpy 输入按 BGR 解释（与 cv2 一致），必须先转 BGR，
       否则会静默造成红蓝通道互换；
    2. 官方评测脚本实际传给检测器的是"真值里出现过的全部类别"
       （ACDC 子集里还包含类别 6），而不是 configs/detector.yaml 列的六类。
       这里保持与官方一致，便于和已发表报告逐位对照。
    """
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    results = model.predict(
        source=image_bgr,
        imgsz=int(config["image_size"]),
        conf=float(config["confidence"]),
        iou=float(config["iou"]),
        device=config.get("device", 0),
        classes=target_classes,
        verbose=False,
    )[0]
    predictions: dict[int, list[dict]] = {}
    for box, score, label in zip(
        results.boxes.xyxy.cpu().numpy(),
        results.boxes.conf.cpu().numpy(),
        results.boxes.cls.cpu().numpy().astype(int),
    ):
        predictions.setdefault(int(label), []).append(
            {"confidence": float(score), "box": [float(v) for v in box]}
        )
    return predictions


def dataset_map50(per_image: list[dict]) -> dict[str, float]:
    """与 summarize_acdc_benchmark.dataset_map50 完全一致的类别均值 mAP50。"""
    predictions: dict[int, list[dict]] = {}
    ground_truth: dict[int, dict[int, list[np.ndarray]]] = {}
    for sample_id, record in enumerate(per_image):
        for class_id_text, boxes in record["ground_truth"].items():
            class_id = int(class_id_text)
            ground_truth.setdefault(class_id, {})[sample_id] = [
                np.asarray(box, dtype=np.float32) for box in boxes
            ]
        for class_id_text, items in record["predictions"].items():
            class_id = int(class_id_text)
            predictions.setdefault(class_id, []).extend(
                {
                    "image_id": sample_id,
                    "confidence": float(item["confidence"]),
                    "box": np.asarray(item["box"], dtype=np.float32),
                }
                for item in items
            )
    values = [
        evaluate_class(predictions.get(class_id, []), images, 0.5)
        for class_id, images in ground_truth.items()
        if images
    ]
    map50 = float(np.mean([item["ap50"] for item in values]))
    # micro 指标用 TP/FP/FN 直接累加，避免小类别被平均掩盖
    tp = sum(int(item["true_positives"]) for item in values)
    fp = sum(int(item["false_positives"]) for item in values)
    fn = sum(int(item["false_negatives"]) for item in values)
    return {
        "map50": map50,
        "recall": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "false_positives_per_image": fp / max(len(per_image), 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="复原损害检测的归因诊断")
    parser.add_argument("--split-dir", default="runs/acdc_benchmark")
    parser.add_argument("--images-dir", default="data/raw/ACDC_images/rgb_anon")
    parser.add_argument("--method", default="direct")
    parser.add_argument("--config", default="configs/detector.yaml")
    parser.add_argument("--output-dir", default="runs/rootcause_diagnosis")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 张，用于冒烟")
    args = parser.parse_args()

    split_dir = ROOT / args.split_dir
    evaluation_path = split_dir / "evaluation" / f"{args.method}.json"
    with evaluation_path.open("r", encoding="utf-8") as handle:
        per_image = json.load(handle)["per_image"]
    if args.limit:
        per_image = per_image[: args.limit]

    with (ROOT / args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    weights = config["model"]
    model = YOLO(str(ROOT / weights) if not Path(weights).is_absolute() else weights)
    target_classes = sorted(
        {int(key) for record in per_image for key in record["ground_truth"]}
    )

    variants = {
        "weather_input_check": None,
        "restored_raw": None,
        "color_matched": "lab",
        "luma_matched": "luma",
        "blend75": "blend75",
        "blend50": "blend50",
        "blend25": "blend25",
    }
    records: dict[str, list[dict]] = {name: [] for name in variants}
    color_shift: list[float] = []

    for index, record in enumerate(per_image):
        relative = record["file_name"]
        input_path = ROOT / args.images_dir / relative
        restored_path = split_dir / "images" / args.method / relative
        input_bgr = cv2.imread(str(input_path))
        restored_bgr = cv2.imread(str(restored_path))
        if input_bgr is None or restored_bgr is None:
            raise FileNotFoundError(f"缺少图像：{input_path} / {restored_path}")
        input_rgb = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB)
        restored_rgb = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
        if restored_rgb.shape != input_rgb.shape:
            raise ValueError(f"尺寸不一致：{relative}")

        # 顺带记录原图的整体色偏强度（LAB 的 a/b 均值差），用于后续分组分析
        lab_in = cv2.cvtColor(input_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab_re = cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        color_shift.append(
            float(np.hypot(lab_in[:, :, 1].mean() - lab_re[:, :, 1].mean(),
                           lab_in[:, :, 2].mean() - lab_re[:, :, 2].mean()))
        )

        candidates = {
            "weather_input_check": input_rgb,
            "restored_raw": restored_rgb,
            "color_matched": lab_stat_transfer(restored_rgb, input_rgb, "lab"),
            "luma_matched": lab_stat_transfer(restored_rgb, input_rgb, "luma"),
            # 复原强度扫描：alpha 是复原图权重，0 等于原图、1 等于全复原
            "blend25": cv2.addWeighted(input_rgb, 0.75, restored_rgb, 0.25, 0.0),
            "blend50": cv2.addWeighted(input_rgb, 0.5, restored_rgb, 0.5, 0.0),
            "blend75": cv2.addWeighted(input_rgb, 0.25, restored_rgb, 0.75, 0.0),
        }
        for name, image in candidates.items():
            records[name].append(
                {
                    "ground_truth": record["ground_truth"],
                    "predictions": run_detector(model, image, config, target_classes),
                }
            )
        if (index + 1) % 20 == 0:
            print(f"已处理 {index + 1}/{len(per_image)} 张", flush=True)

    summary: dict[str, dict[str, float]] = {}
    for name, items in records.items():
        summary[name] = dataset_map50(items)

    baseline = dataset_map50(per_image)
    output = {
        "reference_report": {"direct_from_saved_json": baseline},
        "diagnosis": summary,
        "mean_lab_shift": float(np.mean(color_shift)),
        "image_count": len(per_image),
    }
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)

    print("\n方法 | mAP50 | Recall | Precision | 每图虚警")
    for name, metrics in summary.items():
        print(
            f"{name} | {metrics['map50']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['precision']:.4f} | {metrics['false_positives_per_image']:.3f}"
        )
    print(f"保存到 {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
