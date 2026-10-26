"""NP2：把 bus/train 的每个 GT 在 input/v4/C0/C1/外部模型上的检出情况并到同一张表。

要求对应的原文是「失败样例按 image_id+GT_id 去重，同一目标不同方法表现放同一张对照表」。
这里不重新跑检测，只复用官方 COCO 匹配结果里的 `dtMatches`（IoU=.5 一对一、忽略 crowd），
因此与 `metrics.json` 的 recovered/lost 完全同源。

GT id 的编号规则与 `normalize_ground_truth` 一致：源标注 id + 1。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from saferestore.metrics.next_phase import (CLASSES, coco_evaluate, normalize_ground_truth,
                                            runtime_counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scopes', nargs='+', default=['rain', 'snow'])
    parser.add_argument('--conf', type=float, default=0.25, help='运行时协议用的置信度')
    parser.add_argument('--max-det', type=int, default=300)
    parser.add_argument('--out', default='results/np2_external_probe/bus_train_gt_matrix.csv')
    args = parser.parse_args()

    payload = json.loads((ROOT / 'data/raw/ACDC_detection/gt_detection/instancesonly_val_gt_detection.json').read_text(encoding='utf-8'))
    images, annotations = normalize_ground_truth(payload)
    image_by_id = {i['id']: i for i in images}

    folders = dict(
        input=ROOT / 'results/next_phase/predictions/input',
        v4=ROOT / 'results/next_phase/predictions/v4',
        C0=ROOT / 'results/next_phase/predictions/C0',
        C1=ROOT / 'results/next_phase/predictions/C1',
        histoformer_real=ROOT / 'results/np2_external_probe/predictions/histoformer_real',
    )
    predictions = {name: json.loads((folder / 'runtime025.json').read_text(encoding='utf-8'))['per_image']
                   for name, folder in folders.items()}

    rows = []
    for scope in args.scopes:
        subset = [i for i in images if i['file_name'].split('/')[0] == scope]
        ids = {i['id'] for i in subset}
        gt = [a for a in annotations if a['image_id'] in ids]
        hits = {}
        for name, per_image in predictions.items():
            covered = [r for r in per_image if r['image_id'] in ids]
            if not covered:
                hits[name] = None
                continue
            evaluator = coco_evaluate(subset, gt, covered, args.conf, args.max_det)
            hits[name] = runtime_counts(evaluator)['hits']
        for annotation in gt:
            if annotation['category_id'] not in (5, 6):
                continue
            image = image_by_id[annotation['image_id']]
            row = dict(scope=scope, image_id=image['id'], gt_id=annotation['id'],
                       category=CLASSES[annotation['category_id']], file_name=image['file_name'],
                       bbox=' '.join(f'{v:.1f}' for v in annotation['bbox']))
            for name, value in hits.items():
                row[name] = 'N/A' if value is None else ('hit' if annotation['id'] in value else 'miss')
            row['n_methods_hit'] = sum(1 for name in hits if row[name] == 'hit')
            rows.append(row)

    columns = ['scope', 'image_id', 'gt_id', 'category', 'file_name', 'bbox',
               'input', 'v4', 'C0', 'C1', 'histoformer_real', 'n_methods_hit']
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    disagree = [r for r in rows if len({r[k] for k in ('input', 'v4', 'C0', 'C1', 'histoformer_real')}) > 1]
    print(f'共 {len(rows)} 个 bus/train GT，方法间表现不一致的 {len(disagree)} 个')
    for row in disagree:
        print(f"  {row['scope']:5s} gt={row['gt_id']:5d} {row['category']:5s} "
              f"input={row['input']:4s} v4={row['v4']:4s} C0={row['C0']:4s} C1={row['C1']:4s} "
              f"histo={row['histoformer_real']:4s} {row['file_name']}")
    print('已写入', out)


if __name__ == '__main__':
    main()
