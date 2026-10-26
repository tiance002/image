"""NP2-R：分块融合的恒等模型验收。

为什么必须做这一步：融合公式写错时，输出仍然是「正常的图、正常的尺寸、没有 NaN」，
唯一的破绽是**逐像素不等于输入**。所以这里用恒等网络（Identity）包住整条分块管线，
要求输出与输入逐像素一致；不一致就直接失败。

覆盖：
- 常量图（最容易被权重误差放大成可测偏差）
- 随机图
- 1920x1080（真实 ACDC 尺寸，会走 6 块）
- 小于 tile 的尺寸（500x700，只有 1 块，且用到窗口左上角的小权重区）
- 非整除尺寸（641x961，最后一块需要对齐到末尾）
- 恰为 tile 的尺寸（640x640）

另外做两组「真实网络、块参数不同」的对照，用来把残差归因到上下文而不是融合：
- 1024x1024 裁剪：整幅单块 vs tile=640 分块
- 1920x1080：tile=1024 分块 vs tile=640 分块
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.np2_histoformer_probe import build_model  # noqa: E402
from scripts.np2_histoformer_restore import forward_patch, read_rgb, restore_tiled, tile_positions  # noqa: E402


def case_report(name, restored, source, extra=None):
    diff = np.abs(restored.astype(np.float64) - source.astype(np.float64))
    h, w = source.shape[:2]
    corners = {
        'tl': float(diff[0, 0].max()), 'tr': float(diff[0, -1].max()),
        'bl': float(diff[-1, 0].max()), 'br': float(diff[-1, -1].max()),
    }
    return dict(case=name, height=h, width=w,
                max_abs_diff=float(diff.max()), mean_abs_diff=float(diff.mean()),
                p999_abs_diff=float(np.percentile(diff, 99.9)),
                corner_max_abs_diff=corners,
                identity_ok=bool(diff.max() <= 1e-6), **(extra or {}))


def run_identity_cases(tile=640, overlap=64):
    net = torch.nn.Identity()
    results = []
    sizes = [(1080, 1920), (1920, 1080), (700, 500), (961, 641), (640, 640), (130, 100)]
    for height, width in sizes:
        rng = np.random.default_rng(20261026)
        constant = np.full((height, width, 3), 0.5, dtype=np.float32)
        random = rng.random((height, width, 3)).astype(np.float32)
        for label, image in (('constant', constant), ('random', random)):
            restored, tiles, min_w = restore_tiled(net, image, 'cpu', tile, overlap, False)
            results.append(case_report(f'{label}_{height}x{width}', restored, image,
                                       dict(tiles=tiles, fusion_min_weight=min_w)))
    return results


def run_network_cases(weights, tile=640, overlap=64):
    """真实权重下比较不同块参数；差异应归因于上下文，而不是融合误差。"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = build_model(Path(weights), device)
    payload = json.loads((ROOT / 'data/raw/ACDC_detection/gt_detection/instancesonly_val_gt_detection.json')
                         .read_text(encoding='utf-8'))
    # 固定取一张 snow 图，与既有 tiling_consistency.json 同源思路
    image = sorted([i for i in payload['images'] if i['file_name'].split('/')[0] == 'snow'],
                   key=lambda x: x['id'])[0]
    rgb = read_rgb(ROOT / 'data/raw/ACDC_images/rgb_anon' / image['file_name']).astype(np.float32) / 255.0
    h, w = rgb.shape[:2]
    out = []
    # 全幅 vs 分块：只对能放进显存的裁剪（1024x1024）做单块对照
    crop = rgb[100:100 + 1024, 100:100 + 1024]
    single, _, _ = restore_tiled(net, crop, device, 1024, 0, False)
    tiled, tiles, _ = restore_tiled(net, crop, device, tile, overlap, False)
    diff = np.abs(single - tiled)
    out.append(dict(case='net_1024crop_single_vs_640tile', file=image['file_name'],
                    tiles=tiles, max_abs_diff=float(diff.max()), mean_abs_diff=float(diff.mean()),
                    p999_abs_diff=float(np.percentile(diff, 99.9)),
                    changed_frac_maxchan_gt_0p02=float((diff.max(axis=2) > 0.02).mean())))
    # 全幅不同块大小（都不做单块，避免 8GB 溢出）
    a, tiles_a, _ = restore_tiled(net, rgb, device, 1024, 64, False)
    b, tiles_b, _ = restore_tiled(net, rgb, device, tile, overlap, False)
    diff = np.abs(a - b)
    out.append(dict(case='net_full_1024tile_vs_640tile', file=image['file_name'],
                    height=h, width=w, tiles_1024=tiles_a, tiles_640=tiles_b,
                    max_abs_diff=float(diff.max()), mean_abs_diff=float(diff.mean()),
                    p999_abs_diff=float(np.percentile(diff, 99.9)),
                    changed_frac_maxchan_gt_0p02=float((diff.max(axis=2) > 0.02).mean())))
    return out


def legacy_restore_tiled(net, rgb, device, tile, overlap, use_bf16, denominator_floor=1e-8):
    """最小复现：旧实现（1e-6 下限的一维窗 + 分母截断到 1e-8）。

    保留这段是为了让「角点变黑」这个缺陷可以被逐像素复现，而不是只写在报告里。
    预期：常量图四角输出约为 1e-4 倍原值（接近黑），中心正常。
    """
    h, w = rgb.shape[:2]
    stride = tile - overlap
    ys, xs = tile_positions(h, tile, stride), tile_positions(w, tile, stride)
    acc = np.zeros_like(rgb, dtype=np.float64)
    weight = np.zeros((h, w, 1), dtype=np.float64)
    wy = torch.hann_window(tile, periodic=False, dtype=torch.float32).clamp_min(1e-6).numpy()
    wx = torch.hann_window(tile, periodic=False, dtype=torch.float32).clamp_min(1e-6).numpy()
    window = (wy[:, None] * wx[None, :])[:, :, None]
    for y in ys:
        for x in xs:
            patch = rgb[y:y + tile, x:x + tile]
            ph, pw = patch.shape[:2]
            out = forward_patch(net, patch, device, use_bf16)
            win = window[:ph, :pw]
            acc[y:y + ph, x:x + pw] += out.astype(np.float64) * win
            weight[y:y + ph, x:x + pw] += win
    return (acc / np.maximum(weight, denominator_floor)).clip(0, 1).astype(np.float32)


def run_legacy_repro(tile=640, overlap=64):
    """在常量图上跑旧公式，量化角点被压掉的比例。"""
    net = torch.nn.Identity()
    rows = []
    for height, width in [(1080, 1920), (700, 500)]:
        image = np.full((height, width, 3), 0.5, dtype=np.float32)
        old = legacy_restore_tiled(net, image, 'cpu', tile, overlap, False)
        ratio = old.astype(np.float64) / image.astype(np.float64)
        rows.append(dict(case=f'legacy_constant_{height}x{width}',
                         corner_tl=float(old[0, 0, 0]), expected=0.5,
                         corner_ratio=float(ratio[0, 0, 0]),
                         min_ratio=float(ratio.min()), max_ratio=float(ratio.max()),
                         black_fraction=float((old.max(axis=2) < 0.4 * image.max()).mean())))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', default='third_party/weights/histoformer_official_net_g_real.pth')
    parser.add_argument('--tile', type=int, default=640)
    parser.add_argument('--overlap', type=int, default=64)
    parser.add_argument('--out', default='results/np2_revision/fusion_identity_check.json')
    parser.add_argument('--skip-network', action='store_true')
    args = parser.parse_args()

    report = dict(identity_cases=run_identity_cases(args.tile, args.overlap))
    failures = [r['case'] for r in report['identity_cases'] if not r['identity_ok']]
    report['identity_all_ok'] = not failures
    report['identity_failures'] = failures
    report['legacy_repro'] = run_legacy_repro(args.tile, args.overlap)
    print('恒等用例：', {r['case']: (round(r['max_abs_diff'], 12), r['identity_ok'])
                     for r in report['identity_cases']}, flush=True)
    print('旧公式复现：', [(r['case'], round(r['corner_ratio'], 6), round(r['black_fraction'], 4))
                      for r in report['legacy_repro']], flush=True)
    if not failures and not args.skip_network:
        report['network_cases'] = run_network_cases(args.weights, args.tile, args.overlap)
        for row in report['network_cases']:
            print(row['case'], 'max', round(row['max_abs_diff'], 6),
                  'p99.9', round(row['p999_abs_diff'], 6),
                  '>0.02 占比', round(row['changed_frac_maxchan_gt_0p02'], 5), flush=True)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('已写入', out)
    if failures:
        raise SystemExit(f'融合恒等验收失败：{failures}')


if __name__ == '__main__':
    main()
