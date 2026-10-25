"""导出逐图预测，供配对 bootstrap 与置信度阈值对照使用。

为什么单独写一个脚本
--------------------
训练内的 `evaluate_detection` 只返回汇总指标，逐图预测不落盘，导致两件事做不了：
1. **配对 bootstrap**：200 张就是 ACDC val 的全部，无法扩样本，只能靠重采样
   量化 mAP 的抽样噪声；
2. **阈值对照**：诊断发现 v2 在 conf=0.05 下 mAP 从 0.1930 跳到 0.2267，
   但这需要基线在**同一阈值**下的数字才能比较，不能拿一个方法的 0.05
   去比另一个方法的 0.25。

用法：
    # 复原后导出（用于 bootstrap）
    PYTHONPATH=src;. python scripts/dump_predictions.py \
        --ckpt runs/real_v2_detfeat/best.pt --conf 0.25 --out runs/probe_v2/preds_v2.json

    # 直检原图、低阈值（用于阈值对照）
    PYTHONPATH=src;. python scripts/dump_predictions.py \
        --conf 0.05 --out runs/probe_v2/preds_input_conf005.json
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


def main() -> int:
    parser = argparse.ArgumentParser(description="导出逐图预测")
    parser.add_argument("--ckpt", default="", help="复原模型权重；留空表示直检原图")
    parser.add_argument("--conf", type=float, default=None, help="覆盖检测置信度阈值")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument(
        "--records",
        default="",
        help="自定义评测记录 json（含 per_image，如 406 张全天气）；默认用 200 张雨雾口径",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--blend",
        type=float,
        default=1.0,
        help=(
            "保守混合系数：输出 = 输入 + blend * (复原 - 输入)。"
            "1.0 = 完全用复原结果；0.0 = 完全用输入原图。"
            "小于 1 等价于按比例缩小模型的改动量，"
            "用于在'复原画质'与'检测安全'之间找折中点。"
        ),
    )
    args = parser.parse_args()

    from scripts.diagnose_detection_rootcause import run_detector
    from scripts.train_restorer import (
        build_detector,
        imread_rgb,
        load_gt_records,
        restore_tiled,
    )

    if args.records:
        payload = json.loads((ROOT / args.records).read_text(encoding="utf-8"))
        records = payload["per_image"]
    else:
        records = load_gt_records()
    subset = records[: args.limit] if args.limit > 0 else records
    target_classes = sorted({int(c) for rec in subset for c in rec["ground_truth"]})
    detector, det_config = build_detector()
    if args.conf is not None:
        det_config = dict(det_config)
        det_config["confidence"] = float(args.conf)

    model = None
    if args.ckpt:
        from saferestore.models.nafnet import build_nafnet

        payload = torch.load(ROOT / args.ckpt, map_location="cpu", weights_only=False)
        # 训练评测走 EMA 权重，导出必须一致，否则数字对不上
        state = payload.get("ema") or payload.get("model") or payload
        model = build_nafnet("standard")
        model.load_state_dict(state, strict=True)
        model = model.to(args.device).eval()

    outputs: list[dict] = []
    t_start = time.time()
    with torch.inference_mode():
        for index, rec in enumerate(subset):
            image = imread_rgb(ROOT / "data/raw/ACDC_images/rgb_anon" / rec["file_name"])
            if model is not None:
                tensor = torch.from_numpy(image.transpose(2, 0, 1))[None].to(args.device)
                with torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"
                ):
                    restored = restore_tiled(model, tensor)
                restored_np = (
                    restored[0].float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                )
                # 保守混合：把复原结果按 blend 比例往输入原图回拉。
                # 训练里 --w-delta 已经在惩罚改动量，这里是推理侧的同一思路的补充，
                # 可以在不重训的前提下直接调节"复原强度"这个旋钮。
                alpha = float(args.blend)
                image = image + alpha * (restored_np - image) if alpha != 1.0 else restored_np
            image_u8 = (image * 255.0).round().clip(0, 255).astype(np.uint8)
            preds = run_detector(detector, image_u8, det_config, target_classes)
            outputs.append(
                {
                    "image_id": rec["file_name"],
                    "ground_truth": {
                        str(class_id): [np.asarray(box).tolist() for box in boxes]
                        for class_id, boxes in rec["ground_truth"].items()
                    },
                    "predictions": {
                        str(class_id): [
                            {
                                "box": np.asarray(item["box"]).tolist(),
                                "confidence": float(item["confidence"]),
                            }
                            for item in items
                        ]
                        for class_id, items in preds.items()
                    },
                }
            )
            if (index + 1) % 50 == 0 or index + 1 == len(subset):
                elapsed = time.time() - t_start
                print(
                    f"  {index + 1}/{len(subset)} | {elapsed / 60:.1f} min"
                    f" | 剩余约 {elapsed / (index + 1) * (len(subset) - index - 1) / 60:.1f} min",
                    flush=True,
                )

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(outputs, ensure_ascii=False), encoding="utf-8")
    print(f"已写入 {out_path}（{len(outputs)} 张，conf={det_config['confidence']}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
