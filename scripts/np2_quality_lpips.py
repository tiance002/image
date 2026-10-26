"""NP2：在对齐 dev 上补 LPIPS，并允许把外部复原模型纳入同一画质口径。

口径与 next_phase 的 PSNR/SSIM 完全一致（RGB 0..1、全像素 PSNR、11x11 sigma1.5 valid SSIM），
新增 LPIPS：官方 alex 网络 + 官方 linear 权重（lpips 0.1.4 自带），输入按 [-1,1] 归一化。
只用像素对齐的合成 dev（fog100/rain100），不代表真实夜雪画质。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from scripts.train_restorer import imread_rgb, restore_tiled
from saferestore.models.nafnet import build_nafnet


def quality_metrics(prediction, target):
    prediction = prediction.float(); target = target.float()
    mse = (prediction - target).square().mean()
    coords = torch.arange(11, device=prediction.device, dtype=torch.float32) - 5
    weights = torch.exp(-coords.square() / (2 * 1.5 ** 2)); weights /= weights.sum()
    kernel = (weights[:, None] * weights[None, :]).view(1, 1, 11, 11).repeat(3, 1, 1, 1)
    blur = lambda t: F.conv2d(t, kernel, groups=3)  # noqa: E731
    mx, my = blur(prediction), blur(target)
    vx, vy = blur(prediction.square()) - mx.square(), blur(target.square()) - my.square()
    cov = blur(prediction * target) - mx * my
    ssim = ((2 * mx * my + .01 ** 2) * (2 * cov + .03 ** 2) /
            ((mx.square() + my.square() + .01 ** 2) * (vx + vy + .03 ** 2))).mean()
    return dict(psnr=float(-10 * torch.log10(mse.clamp_min(1e-12))), ssim=float(ssim), mse=float(mse))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='data/real_weather_rgb_v2/manifests/dev.csv')
    parser.add_argument('--methods', nargs='+', default=['input', 'v4', 'C0', 'C1'])
    parser.add_argument('--restored-root', default='', help='外部模型复原图根目录（可选）')
    parser.add_argument('--out', default='results/np2_external_probe/quality_lpips.json')
    args = parser.parse_args()

    import lpips
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    perceptual = lpips.LPIPS(net='alex', verbose=False).to(device).eval()
    for parameter in perceptual.parameters():
        parameter.requires_grad_(False)

    with (ROOT / args.manifest).open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    checkpoints = dict(input=None, v4='runs/real_v4_allweather/latest.pt',
                       C0='runs/gt_next_phase_20261025_C0/latest.pt',
                       C1='runs/gt_next_phase_20261025_C1/latest.pt')

    result = dict(manifest=args.manifest, n=len(rows),
                  metric='PSNR allpixels; SSIM 11x11 sigma1.5 valid; LPIPS alex(v0.1) RGB[-1,1]',
                  scope='合成对齐 dev（fog100/rain100），不是真实跨时刻配对，不能推断真实夜雪',
                  methods={})
    for name in args.methods:
        model = None
        if name in checkpoints and checkpoints[name]:
            state = torch.load(ROOT / checkpoints[name], map_location='cpu', weights_only=False)
            model = build_nafnet().to(device).eval()
            model.load_state_dict(state['ema'])
            del state
        per_image = []
        for row in rows:
            source = torch.from_numpy(imread_rgb(ROOT / row['degraded_path']).transpose(2, 0, 1).copy())[None].to(device)
            target = torch.from_numpy(imread_rgb(ROOT / row['clean_path']).transpose(2, 0, 1).copy())[None].to(device)
            if source.shape != target.shape:
                raise ValueError('非像素对齐尺寸')
            with torch.inference_mode():
                if args.restored_root and name not in checkpoints:
                    restored_path = ROOT / args.restored_root / Path(row['degraded_path']).relative_to('data/real_weather_rgb_v2/degraded')
                    buf = np.frombuffer(restored_path.read_bytes(), dtype=np.uint8)
                    import cv2
                    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    prediction = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].float().to(device) / 255
                elif model is not None:
                    prediction = restore_tiled(model, source, tile=1024, overlap=64).clamp(0, 1)
                else:
                    prediction = source
                metrics = quality_metrics(prediction, target)
                lp = perceptual(prediction * 2 - 1, target * 2 - 1)
                per_image.append(dict(weather=row['weather'], sequence=row['sequence'],
                                      **{k: round(v, 6) for k, v in metrics.items()},
                                      lpips=round(float(lp), 6)))
            if len(per_image) % 50 == 0:
                print(f'{name} {len(per_image)}/{len(rows)}', flush=True)
        by_weather = {}
        for weather in sorted({r['weather'] for r in per_image}):
            subset = [r for r in per_image if r['weather'] == weather]
            by_weather[weather] = {k: float(np.mean([r[k] for r in subset])) for k in ('psnr', 'ssim', 'lpips')}
        result['methods'][name] = dict(
            mean={k: float(np.mean([r[k] for r in per_image])) for k in ('psnr', 'ssim', 'lpips')},
            by_weather=by_weather, per_image=per_image)
        print(name, result['methods'][name]['mean'], flush=True)
        if model is not None:
            del model
        if device == 'cuda':
            torch.cuda.empty_cache()

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('已写入', out)


if __name__ == '__main__':
    main()
