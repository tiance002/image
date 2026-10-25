"""加载训练好的复原权重，在真实 ACDC 雨雾上做完整评测。

输出检测指标（mAP50 / Recall / OPR）、亮度与暗部变化，并导出可视化样例。
所有指标复用 train_restorer 里与官方协议对齐的实现，避免出现第二套口径。

用法：
    PYTHONPATH=src python scripts/eval_restorer.py \
        --checkpoint runs/real_v1/best.pt --tag real_v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from saferestore.models.nafnet import build_nafnet  # noqa: E402
from scripts.train_restorer import (  # noqa: E402
    build_detector,
    evaluate_detection,
    imread_rgb,
    load_gt_records,
    restore_tiled,
)


def save_visuals(model, records: list[dict], out_dir: Path, count: int, device: str) -> None:
    """导出输入/复原对照图，供报告与失败案例分析使用。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    import cv2

    for rec in records[:count]:
        image = imread_rgb(ROOT / "data/raw/ACDC_images/rgb_anon" / rec["file_name"])
        tensor = torch.from_numpy(image.transpose(2, 0, 1))[None].to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            restored = restore_tiled(model, tensor)
        restored_np = restored[0].float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        h, w = image.shape[:2]
        scale = 900 / w
        size = (int(w * scale), int(h * scale))
        pair = np.hstack(
            [
                cv2.resize((image * 255).astype(np.uint8), size),
                cv2.resize((restored_np * 255).astype(np.uint8), size),
            ]
        )
        name = rec["file_name"].replace("/", "__")
        ok, buf = cv2.imencode(".jpg", pair[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
        if ok:
            buf.tofile(str(out_dir / name))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--variant", default="standard")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--visuals", type=int, default=8)
    parser.add_argument("--output-root", default="runs/restorer_eval")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = args.tag or Path(args.checkpoint).parent.name
    out_dir = ROOT / args.output_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_nafnet(variant=args.variant).to(device)
    state = torch.load(ROOT / args.checkpoint, map_location=device, weights_only=False)
    weights = state.get("ema") or state.get("model")
    model.load_state_dict(weights)
    model.eval()

    records = load_gt_records()
    baseline = json.loads(
        (ROOT / "runs/acdc_benchmark/evaluation/weather_input.json").read_text(encoding="utf-8")
    )["per_image"]
    for rec, base in zip(records, baseline):
        rec["baseline_predictions"] = base["predictions"]

    # 全量复原图落盘，供 scripts/diagnose_detection_rootcause.py 直接复用做强度扫描
    dump_dir = out_dir / "images" / tag
    detector, config = build_detector()
    metrics = evaluate_detection(
        model, records, detector, config, args.limit, device, save_dir=dump_dir
    )
    metrics["checkpoint"] = str(args.checkpoint)
    metrics["iter"] = int(state.get("iter", -1))

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # 生成与 acdc_benchmark 同构的 per_image 文件，使归因脚本可以直接复用：
    #   python scripts/diagnose_detection_rootcause.py \
    #       --split-dir <output_root> --method <tag>
    # 归因脚本只需要 file_name 与 ground_truth（检测由它自己重跑），
    # 所以这里不重复保存预测框，省体积也避免两份数字不一致。
    evaluation_dir = out_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    (evaluation_dir / f"{tag}.json").write_text(
        json.dumps(
            {
                "method": tag,
                **metrics,
                "per_image": [
                    {
                        "file_name": rec["file_name"],
                        "weather": rec.get("weather"),
                        "ground_truth": rec["ground_truth"],
                    }
                    for rec in records
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(
        f"{tag}: mAP50 {metrics['map50']:.4f}  Recall {metrics['recall']:.4f}  "
        f"Precision {metrics['precision']:.4f}  OPR {metrics['opr']:.4f}  "
        f"暗部 {metrics['dark_before']:.3f}->{metrics['dark_after']:.3f}  "
        f"FP/图 {metrics['false_positives_per_image']:.3f}  n={metrics['n_images']}"
    )
    print(f"全量复原图已写入 {dump_dir}")
    for weather, item in (metrics.get("per_weather") or {}).items():
        print(
            f"  [{weather}] mAP50 {item['map50']:.4f}  Recall {item['recall']:.4f}  "
            f"OPR {item['opr']:.4f}  n={item['n_images']}"
        )
    if args.visuals > 0:
        save_visuals(model, records, out_dir / "pairs", args.visuals, device)
        print(f"左右对照样例已写入 {out_dir / 'pairs'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
