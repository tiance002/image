"""NP2：分块推理与全幅前向的一致性检查。

全幅 1920x1080 会超出本机 8GB 显存，必须分块；这里在能放进显存的 1024x1024 裁剪上
对比「全幅一次前向」与「640/64 汉宁融合分块」的输出差异，量化 tiling 引入的偏差。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.np2_histoformer_probe import build_model  # noqa: E402
from scripts.np2_histoformer_restore import forward_patch, restore_tiled, read_rgb  # noqa: E402


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float('inf') if mse == 0 else -10 * np.log10(mse)


def main():
    weights = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'third_party/weights/histoformer_official_net_g_real.pth'
    device = 'cuda'
    net = build_model(weights, device)
    cases = [
        ('rain', 'rain/val/GP030400/GP030400_frame_000129_rgb_anon.png', 300, 500),
        ('snow', None, 400, 600),
        ('fog', None, 300, 500),
    ]
    payload = json.loads((ROOT / 'data/raw/ACDC_detection/gt_detection/instancesonly_val_gt_detection.json').read_text(encoding='utf-8'))
    picked = {}
    for image in payload['images']:
        weather = image['file_name'].split('/')[0]
        if weather not in picked:
            picked[weather] = image['file_name']

    results = []
    for weather, explicit, top, left in cases:
        rel = explicit or picked.get(weather)
        if rel is None:
            continue
        rgb = read_rgb(ROOT / 'data/raw/ACDC_images/rgb_anon' / rel).astype(np.float32) / 255.0
        h, w = rgb.shape[:2]
        top = min(top, h - 1024)
        left = min(left, w - 1024)
        crop = rgb[top:top + 1024, left:left + 1024]
        full = forward_patch(net, crop, device, use_bf16=False)
        tiled, n_tiles = restore_tiled(net, crop, device, 640, 64, False)
        diff = np.abs(full - tiled)
        # 接缝带：块边界附近 ±16 像素
        seam = np.zeros((1024, 1024), dtype=bool)
        for pos in (576,):
            seam[pos - 16:pos + 16, :] = True
            seam[:, pos - 16:pos + 16] = True
        results.append(dict(weather=weather, file_name=rel, crop=[top, left, 1024, 1024], tiles=n_tiles,
                            psnr_full_vs_tiled=psnr(full, tiled), max_abs_diff=float(diff.max()),
                            mean_abs_diff=float(diff.mean()),
                            seam_mean_abs_diff=float(diff[seam].mean()),
                            interior_mean_abs_diff=float(diff[~seam].mean()),
                            full_out_of_range=int(((full < 0) | (full > 1)).sum())))
        print(f'{weather}: PSNR={results[-1]["psnr_full_vs_tiled"]:.2f}dB max={diff.max():.4f} '
              f'seam均值={diff[seam].mean():.5f} 内部均值={diff[~seam].mean():.5f}', flush=True)

    out = ROOT / 'results/np2_external_probe/tiling_consistency.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(weights=str(weights), tile=640, overlap=64, dtype='float32',
                                   note='全幅前向仅用于 1024x1024 裁剪的等价性检查；正式推理仍为分块',
                                   cases=results), ensure_ascii=False, indent=2), encoding='utf-8')
    print('已写入', out)


if __name__ == '__main__':
    main()
