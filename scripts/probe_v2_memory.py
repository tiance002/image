"""在切换 v2 之前探测显存，决定检测器特征损失能开多大。

背景：8GB 卡上 v1（bs4 / crop320 / fp32）已占 7.7GB，再挂一个 YOLO 的前向+反向
很容易 OOM。检测器显存与输入边长成平方关系，所以要先量出可用的裁剪尺寸，
而不是等训练跑到一半崩掉。
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from saferestore.losses.perceptual import DetectorFeatureLoss  # noqa: E402
from saferestore.models.nafnet import build_nafnet  # noqa: E402


def measure(crop: int, batch: int = 4, canvas: int = 320, steps: int = 3) -> dict:
    """按真实训练的 batch/canvas 组合跑几步，返回峰值显存。"""
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    model = build_nafnet("standard").cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.99))
    feat_loss = DetectorFeatureLoss(ROOT / "yolov8n.pt").cuda()

    degraded = torch.rand(batch, 3, canvas, canvas, device="cuda")
    clean = torch.rand(batch, 3, canvas, canvas, device="cuda")

    peak = 0.0
    try:
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            pred = model(degraded)
            loss = (pred - clean).abs().mean()
            if crop and canvas > crop:
                top = (canvas - crop) // 2
                left = (canvas - crop) // 2
                p = pred[:, :, top : top + crop, left : left + crop].clamp(0, 1)
                c = clean[:, :, top : top + crop, left : left + crop]
            else:
                p, c = pred.clamp(0, 1), clean
            loss = loss + 2.0 * feat_loss(p, c)
            loss.backward()
            opt.step()
            peak = max(peak, torch.cuda.max_memory_allocated() / 1e9)
    finally:
        feat_loss.close()
        del model, opt, feat_loss, degraded, clean
        torch.cuda.empty_cache()
        gc.collect()
    return {"crop": crop, "peak_gb": round(peak, 3)}


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("无 CUDA，跳过探测")
        return 0
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"显存总量 {total:.2f} GB")
    # 从大到小试，第一次成功的就是可行配置
    for crop in (256, 192, 128):
        try:
            result = measure(crop)
        except torch.cuda.OutOfMemoryError:
            print(f"crop={crop}: OOM")
            torch.cuda.empty_cache()
            continue
        print(f"crop={crop}: 峰值 {result['peak_gb']:.2f} GB")
        if result["peak_gb"] < total * 0.94:
            print(f"建议使用 --det-feat-crop {crop}")
            return 0
    print("所有配置都吃紧，建议降到 --batch 3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
