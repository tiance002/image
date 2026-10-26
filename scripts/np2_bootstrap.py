"""NP2：外部复原模型与对照方法的配对图像 bootstrap（只重采样，不重跑检测）。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from saferestore.metrics.next_phase import coco_evaluate, normalize_ground_truth, resample_summary

KEYS = ('map50', 'map50_95', 'bus', 'train', 'bicycle', 'car', 'person')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', nargs='+',
                        default=['histoformer_real:input', 'histoformer_real:v4',
                                 'histoformer_real:C0', 'histoformer_real:C1'])
    parser.add_argument('--resamples', type=int, default=1000)
    parser.add_argument('--scopes', nargs='+', default=['rain', 'snow'])
    parser.add_argument('--out', default='results/np2_external_probe/bootstrap.json')
    args = parser.parse_args()

    payload = json.loads((ROOT / 'data/raw/ACDC_detection/gt_detection/instancesonly_val_gt_detection.json').read_text(encoding='utf-8'))
    images, annotations = normalize_ground_truth(payload)
    folders = dict(
        input=ROOT / 'results/next_phase/predictions/input',
        v4=ROOT / 'results/next_phase/predictions/v4',
        C0=ROOT / 'results/next_phase/predictions/C0',
        C1=ROOT / 'results/next_phase/predictions/C1',
        histoformer_real=ROOT / 'results/np2_external_probe/predictions/histoformer_real',
    )
    methods = sorted({m for pair in args.pairs for m in pair.split(':')})
    predictions = {m: json.loads((folders[m] / 'ap001.json').read_text(encoding='utf-8'))['per_image'] for m in methods}

    result = dict(resamples=args.resamples, seed=20261025, unit='paired image bootstrap, percentile 95',
                  limitation='ACDC val 相邻帧高度相关；这是图像级重采样，不是场景独立、也不是多随机种子证据。'
                             '区间跨零只表示证据不足，不能当成「无损」证明。',
                  scopes={})
    for scope in args.scopes:
        subset = [i for i in images if i['file_name'].split('/')[0] == scope]
        ids = {i['id'] for i in subset}
        gt = [a for a in annotations if a['image_id'] in ids]
        evaluators = {m: coco_evaluate(subset, gt, [r for r in predictions[m] if r['image_id'] in ids]) for m in methods}
        full = np.arange(len(subset))
        points = {m: resample_summary(evaluators[m], full) for m in methods}
        rng = np.random.default_rng(20261025)
        samples = {pair: {k: [] for k in KEYS} for pair in args.pairs}
        for index in range(args.resamples):
            indices = rng.choice(len(subset), len(subset), replace=True)
            metrics = {m: resample_summary(evaluators[m], indices) for m in methods}
            for pair in args.pairs:
                candidate, reference = pair.split(':')
                for key in KEYS:
                    if key in metrics[candidate] and key in metrics[reference]:
                        samples[pair][key].append(metrics[candidate][key] - metrics[reference][key])
            if (index + 1) % 250 == 0:
                print(scope, 'bootstrap', index + 1, '/', args.resamples, flush=True)
        rows = {}
        for pair, values in samples.items():
            candidate, reference = pair.split(':')
            summary = {}
            for key, series in values.items():
                finite = np.asarray(series)[np.isfinite(series)]
                if not len(finite):
                    summary[key] = None
                    continue
                lo, hi = np.quantile(finite, [.025, .975])
                delta = points[candidate][key] - points[reference][key]
                summary[key] = dict(delta=float(delta) if np.isfinite(delta) else None,
                                    ci95=[float(lo), float(hi)], sign_positive=bool(lo > 0),
                                    sign_negative=bool(hi < 0), valid=len(finite))
            rows[pair] = summary
        result['scopes'][scope] = rows

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

    for scope, rows in result['scopes'].items():
        print(f'\n=== {scope} ===')
        for pair, summary in rows.items():
            cells = []
            for key in ('map50', 'map50_95', 'bus', 'train'):
                item = summary.get(key)
                if item is None:
                    cells.append(f'{key}=N/A')
                else:
                    cells.append(f"{key}={item['delta']:+.5f}[{item['ci95'][0]:+.5f},{item['ci95'][1]:+.5f}]")
            print(pair, ' | '.join(cells))
    print('\n已写入', out)


if __name__ == '__main__':
    main()
