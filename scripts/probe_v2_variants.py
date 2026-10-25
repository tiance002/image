"""诊断 v2 复原图在 ACDC 真实雨雾上的失效点。

动机
----
v2 在 200 张上的整体指标是 mAP50 0.1930 / Recall 0.4244 / Precision 0.7092 /
每图虚警 1.185，而"雨雾图直接送检测器"的基线是 0.2042 / 0.4332 / 0.6966 / 1.285。
也就是说，我们的 Precision 和虚警率**都比基线更好**，Recall 只低 0.009，
但 mAP 却低了 0.011。

这种"各项都不差、只有平均值差"的形态，说明不是均匀地弱一点，而是
mAP 的构成里有某个部分塌了。mAP 是在**每个类别内部**对 PR 曲线做积分再平均，
所以只有拆到逐类 AP 才能定位。本脚本一次性排查四个假设：

1. **某一类告急**：少数类别 AP 崩了，被平均掩盖（逐类 AP 表）；
2. **改动过头**：与原始输入按比例混合能否回血（blend 0.9 / 0.8）；
3. **尺度不匹配**：训练用 320 裁剪、推理用 1024 分块，检测器看到的纹理
   统计可能偏离训练分布（tile 320 vs 1024）；
4. **置信度被截断**：评测沿用 conf=0.25，若真目标置信度整体下移，会先被
   这道闸门切掉，Recall 与 AP 都被系统性压低（conf 0.05 对照）。

用法：
    PYTHONPATH=src;. python scripts/probe_v2_variants.py \
        --ckpt runs/real_v2_detfeat/iter_016000.pt --limit 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from scripts.diagnose_detection_rootcause import evaluate_class, run_detector  # noqa: E402

# 变体定义：tile 是复原时的分块边长，alpha 是与原始输入的混合比，
# conf 为 None 表示沿用官方协议里的 0.25。
VARIANTS: list[tuple[str, dict]] = [
    ("input", {"tile": 0, "alpha": 1.0, "conf": None}),
    ("base_t1024", {"tile": 1024, "alpha": 1.0, "conf": None}),
    ("blend090", {"tile": 1024, "alpha": 0.90, "conf": None}),
    ("blend080", {"tile": 1024, "alpha": 0.80, "conf": None}),
    ("tile320", {"tile": 320, "alpha": 1.0, "conf": None}),
    ("conf005", {"tile": 1024, "alpha": 1.0, "conf": 0.05}),
]


def per_class_ap(per_image: list[dict], names: dict[int, str]) -> list[dict]:
    """逐类算 AP50，同时给出该类别的真值框数量，避免小类别结论被放大。"""
    predictions: dict[int, list[dict]] = {}
    ground_truth: dict[int, dict[int, list[np.ndarray]]] = {}
    counts: dict[int, int] = {}

    for sample_id, record in enumerate(per_image):
        for class_id_text, boxes in record["ground_truth"].items():
            class_id = int(class_id_text)
            ground_truth.setdefault(class_id, {})[sample_id] = [
                np.asarray(box, dtype=np.float32) for box in boxes
            ]
            counts[class_id] = counts.get(class_id, 0) + len(boxes)
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

    rows: list[dict] = []
    for class_id, images in ground_truth.items():
        if not images:
            continue
        result = evaluate_class(predictions.get(class_id, []), images, 0.5)
        rows.append(
            {
                "class_id": class_id,
                "class_name": names.get(class_id, str(class_id)),
                "n_gt": counts[class_id],
                "ap50": float(result["ap50"]),
                "true_positives": int(result["true_positives"]),
                "false_positives": int(result["false_positives"]),
                "false_negatives": int(result["false_negatives"]),
            }
        )
    rows.sort(key=lambda row: row["n_gt"], reverse=True)
    return rows


def summarize(per_image: list[dict], names: dict[int, str]) -> dict:
    rows = per_class_ap(per_image, names)
    tp = sum(row["true_positives"] for row in rows)
    fp = sum(row["false_positives"] for row in rows)
    fn = sum(row["false_negatives"] for row in rows)
    # 与官方 dataset_map50 一致：类别等权平均
    map50 = float(np.mean([row["ap50"] for row in rows])) if rows else 0.0
    return {
        "map50": map50,
        "recall": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "false_positives_per_image": fp / max(len(per_image), 1),
        "n_images": len(per_image),
        "per_class": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="诊断 v2 复原图的检测失效点")
    parser.add_argument("--ckpt", default="runs/real_v2_detfeat/iter_016000.pt")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--out", default="runs/probe_v2")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import yaml

    from scripts.train_restorer import (
        build_detector,
        imread_rgb,
        load_gt_records,
        restore_tiled,
    )
    from saferestore.models.nafnet import build_nafnet

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load((ROOT / "configs/detector.yaml").read_text(encoding="utf-8"))
    names = {int(key): str(value) for key, value in (config.get("classes") or {}).items()}
    # 官方协议里目标类别取自真值，子集里还包含 detector.yaml 未列出的类别
    records = load_gt_records()
    subset = records[: args.limit] if args.limit > 0 else records
    target_classes = sorted({int(c) for rec in subset for c in rec["ground_truth"]})

    detector, det_config = build_detector()

    model = build_nafnet("standard")
    payload = torch.load(ROOT / args.ckpt, map_location="cpu")
    # 训练时的 dev 评测走的是 EMA 权重，诊断必须用同一套，否则数字复现不出来
    if "ema" in payload:
        state = payload["ema"]
    elif "model" in payload:
        state = payload["model"]
    else:
        state = payload
    model.load_state_dict(state, strict=True)
    model = model.to(args.device).eval()

    acc: dict[str, list[dict]] = {name: [] for name, _ in VARIANTS}
    t_start = time.time()

    with torch.inference_mode():
        for index, rec in enumerate(subset):
            image = imread_rgb(ROOT / "data/raw/ACDC_images/rgb_anon" / rec["file_name"])
            tensor = torch.from_numpy(image.transpose(2, 0, 1))[None].to(args.device)
            # 复原只做两档分块，混合与置信度变体复用它，避免重复前向
            restored = {}
            for tile in (1024, 320):
                with torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"
                ):
                    restored[tile] = restore_tiled(model, tensor, tile=tile)[0].float().clamp(0, 1)

            for name, cfg in VARIANTS:
                if cfg["tile"] == 0:
                    base = tensor[0].float().clamp(0, 1)
                else:
                    base = restored[cfg["tile"]]
                alpha = float(cfg["alpha"])
                mixed = alpha * base + (1.0 - alpha) * tensor[0].float().clamp(0, 1)
                image_rgb = mixed.permute(1, 2, 0).cpu().numpy()
                image_u8 = (image_rgb * 255.0).round().clip(0, 255).astype(np.uint8)

                current = dict(det_config)
                if cfg["conf"] is not None:
                    current["confidence"] = float(cfg["conf"])
                preds = run_detector(detector, image_u8, current, target_classes)
                acc[name].append(
                    {"ground_truth": rec["ground_truth"], "predictions": preds}
                )

            if (index + 1) % 20 == 0 or index + 1 == len(subset):
                elapsed = time.time() - t_start
                eta = elapsed / (index + 1) * (len(subset) - index - 1)
                print(
                    f"  进度 {index + 1}/{len(subset)} | 已用 {elapsed / 60:.1f} min"
                    f" | 预计剩余 {eta / 60:.1f} min",
                    flush=True,
                )

    report = {name: summarize(acc[name], names) for name, _ in VARIANTS}
    (out_dir / "probe.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---------------------------------------------------------- Markdown
    lines = [
        "# v2 复原图检测失效点诊断",
        "",
        f"> 权重：`{args.ckpt}`；数据：ACDC fog 100 + rain 100，官方检测框。",
        f"> 检测协议：YOLOv8n @640，NMS IoU 0.7，mAP 用 101 点插值。",
        "",
        "## 总体对比",
        "",
        "| 变体 | mAP50 | Recall | Precision | 每图虚警 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, _ in VARIANTS:
        item = report[name]
        lines.append(
            f"| {name} | {item['map50']:.4f} | {item['recall']:.4f} | "
            f"{item['precision']:.4f} | {item['false_positives_per_image']:.3f} |"
        )

    base_classes = {row["class_id"]: row for row in report["input"]["per_class"]}
    lines += [
        "",
        "## 逐类 AP50",
        "",
        "| 类别 | 真值数 | 原图直检 | v2 复原 | 差值 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["base_t1024"]["per_class"]:
        class_id = row["class_id"]
        ref = base_classes.get(class_id)
        ref_ap = ref["ap50"] if ref else float("nan")
        delta = row["ap50"] - ref_ap if ref else float("nan")
        lines.append(
            f"| {row['class_name']}（{class_id}） | {row['n_gt']} | {ref_ap:.4f} | "
            f"{row['ap50']:.4f} | {delta:+.4f} |"
        )

    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "\n".join(lines))
    print(f"\n结果已写入 {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
