"""NP2：把 Histoformer 官方权重跑在对齐 dev（合成 fog100/rain100）上，供同口径画质对照。

为什么单独一个脚本：
- `np2_histoformer_restore.py` 只从 ACDC val 清单取图（用于检测评测）；
- 画质必须有像素对齐的 clean 参考，只有 `data/real_weather_rgb_v2/manifests/dev.csv` 提供这种配对。
- 复原逻辑（官方预处理、分块、汉宁融合）完全复用同一份实现，不另写一套，保证与检测侧一致。

产物写到 `results/np2_external_probe/restored/histoformer_dev/<degraded 下的相对路径>`，
目录结构与 `degraded/` 对齐，好让 `np2_quality_lpips.py --restored-root` 直接套用。
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from scripts.np2_histoformer_probe import build_model  # noqa: E402
from scripts.np2_histoformer_restore import read_rgb, restore_tiled, sha256, write_png  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--name', default='histoformer_dev')
    parser.add_argument('--manifest', default='data/real_weather_rgb_v2/manifests/dev.csv')
    parser.add_argument('--degraded-prefix', default='data/real_weather_rgb_v2/degraded')
    parser.add_argument('--tile', type=int, default=640)
    parser.add_argument('--overlap', type=int, default=64)
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--out-root', default='results/np2_external_probe/restored')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    with (ROOT / args.manifest).open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]

    out_dir = ROOT / args.out_root / args.name
    weights = Path(args.weights)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = build_model(weights, device)
    params = sum(p.numel() for p in net.parameters())

    records, timings = [], []
    begin_all = time.perf_counter()
    for index, row in enumerate(rows):
        src = ROOT / row['degraded_path']
        rgb = read_rgb(src).astype(np.float32) / 255.0
        torch.cuda.synchronize() if device == 'cuda' else None
        begin = time.perf_counter()
        restored, n_tiles = restore_tiled(net, rgb, device, args.tile, args.overlap, args.bf16)
        torch.cuda.synchronize() if device == 'cuda' else None
        elapsed = time.perf_counter() - begin
        if restored.shape != rgb.shape:
            raise RuntimeError(f'尺寸不一致：{src.name} {restored.shape} vs {rgb.shape}')
        uint8 = (restored * 255).round().clip(0, 255).astype(np.uint8)
        # 保留 degraded/ 下的相对层级（dev/xxx.jpg），供画质脚本按同一规则拼路径
        rel = src.relative_to(ROOT / args.degraded_prefix)
        write_png(out_dir / rel, uint8)
        diff = np.abs(restored - rgb)
        records.append(dict(file_name=str(rel).replace('\\', '/'), weather=row['weather'],
                            scene_id=row['scene_id'], tiles=n_tiles,
                            diff_mean=float(diff.mean()),
                            changed_frac_002=float((diff.max(axis=2) > 0.02).mean()),
                            restore_ms=elapsed * 1000))
        timings.append(elapsed * 1000)
        if (index + 1) % 20 == 0 or index + 1 == len(rows):
            print(f'{args.name} {index+1}/{len(rows)} 平均 {np.mean(timings):.0f}ms/图', flush=True)

    meta = dict(method=args.name, weights=str(weights), weights_sha256=sha256(weights),
                weights_bytes=weights.stat().st_size, restore_parameters=params,
                official_repo='https://github.com/sunshangquan/Histoformer',
                official_commit='1f045f06c03551c31504d8042dbe6ff9b9569108',
                preprocessing='BGR->RGB, /255, reflect pad to multiple of 8, clamp[0,1], ->BGR uint8',
                adaptation=dict(tiling=True, tile=args.tile, overlap=args.overlap, fusion='hann',
                                dtype='bfloat16' if args.bf16 else 'float32'),
                manifest=args.manifest, images=len(rows),
                mean_restore_ms=float(np.mean(timings)), median_restore_ms=float(np.median(timings)),
                per_image=records)
    (out_dir / 'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print('完成：', out_dir)


if __name__ == '__main__':
    main()
