"""NP2：外部复原模型与 input/v4/C0/C1 在同一冻结协议下的汇总对照。

规则：
- 完全复用 next_phase 的 COCO 官方 AP（conf=.001）与运行时计数（conf=.25, maxDet300）。
- 外部模型只在其覆盖的天气子集上给分；未覆盖天气写 None（N/A），不出现在总均值里冒充。
- 回退组合（外部模型覆盖雨天/雪天，其余用原图）单独命名，不与纯模型成绩混写。
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from saferestore.metrics.next_phase import (CLASSES, coco_evaluate, normalize_ground_truth,
                                            precision_summary, runtime_counts)

WEATHERS = ('fog', 'rain', 'night', 'snow')


def load_predictions(folder: Path):
    ap = json.loads((folder / 'ap001.json').read_text(encoding='utf-8'))['per_image']
    runtime = json.loads((folder / 'runtime025.json').read_text(encoding='utf-8'))['per_image']
    return ap, runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--external', nargs='+', default=['histoformer_real'])
    parser.add_argument('--reference-root', default='results/next_phase/predictions')
    parser.add_argument('--external-root', default='results/np2_external_probe/predictions')
    parser.add_argument('--out', default='results/np2_external_probe/metrics.json')
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / 'configs/next_phase_evaluation.yaml').read_text(encoding='utf-8'))
    payload = json.loads((ROOT / config['annotations']).read_text(encoding='utf-8'))
    images, annotations = normalize_ground_truth(payload)
    cat_of = {a['id']: a['category_id'] for a in annotations}

    methods = {}
    ref_root = ROOT / args.reference_root
    for name in ('input', 'v4', 'C0', 'C1'):
        folder = ref_root / name
        if folder.is_dir():
            methods[name] = load_predictions(folder)
    ext_root = ROOT / args.external_root
    covered = {}
    for name in args.external:
        folder = ext_root / name
        if not folder.is_dir():
            print(f'跳过（无预测）：{name}')
            continue
        ap, runtime = load_predictions(folder)
        methods[name] = (ap, runtime)
        covered[name] = sorted({row['weather'] for row in ap})
        print(f'{name} 覆盖天气：{covered[name]}')

    # 回退组合：外部模型覆盖的天气用外部预测，其余用 input
    for name, weathers in list(covered.items()):
        if 'input' not in methods:
            continue
        input_ap, input_runtime = methods['input']
        ext_ap, ext_runtime = methods[name]
        keep = set(weathers)
        merged_ap = [row for row in input_ap if row['weather'] not in keep] + [row for row in ext_ap if row['weather'] in keep]
        merged_runtime = [row for row in input_runtime if row['weather'] not in keep] + [row for row in ext_runtime if row['weather'] in keep]
        merged_ap.sort(key=lambda r: r['image_id'])
        merged_runtime.sort(key=lambda r: r['image_id'])
        methods[f'{name}+input_fallback'] = (merged_ap, merged_runtime)
        covered[f'{name}+input_fallback'] = list(WEATHERS)

    scopes = dict(all406=images, rain_fog200=[i for i in images if i['file_name'].split('/')[0] in ('fog', 'rain')])
    scopes.update({w: [i for i in images if i['file_name'].split('/')[0] == w] for w in WEATHERS})

    result = dict(protocol=config, scopes_note='406/200 都是 ACDC val 模型选择集，不是独立 test',
                  coverage=covered, methods={})
    for name, (ap, runtime) in methods.items():
        entry = dict(scopes={})
        for scope, subset in scopes.items():
            weathers_in_scope = {i['file_name'].split('/')[0] for i in subset}
            ids = {i['id'] for i in subset}
            gt = [a for a in annotations if a['image_id'] in ids]
            ap_rows = [r for r in ap if r['image_id'] in ids]
            rt_rows = [r for r in runtime if r['image_id'] in ids]
            if len(ap_rows) != len(subset):
                entry['scopes'][scope] = dict(n_images=len(subset), available=False,
                                              missing=len(subset) - len(ap_rows))
                continue
            e = coco_evaluate(subset, gt, ap_rows)
            er = coco_evaluate(subset, gt, rt_rows, .25)
            er300 = coco_evaluate(subset, gt, rt_rows, .25, 300)
            counts = runtime_counts(er300)
            hits = counts.pop('hits')
            counts['detected_gt_ids'] = sorted(hits)
            entry['scopes'][scope] = dict(n_images=len(subset), available=True,
                                          standard=precision_summary(e),
                                          runtime=counts)
        result['methods'][name] = entry

    # 相对 input 的 OPR 与 bus/train 级恢复/丢失
    if 'input' in result['methods']:
        for name, entry in result['methods'].items():
            for scope, row in entry['scopes'].items():
                base_row = result['methods']['input']['scopes'].get(scope)
                if not row.get('available') or not base_row or not base_row.get('available'):
                    continue
                base = set(base_row['runtime']['detected_gt_ids'])
                hit = set(row['runtime']['detected_gt_ids'])
                row['runtime']['opr'] = len(base & hit) / len(base) if base else None
                row['runtime']['lost_input_gt'] = sorted(base - hit)
                row['runtime']['gained_input_gt'] = sorted(hit - base)
                for code, label in ((5, 'bus'), (6, 'train')):
                    base_c = {g for g in base if cat_of.get(g) == code}
                    hit_c = {g for g in hit if cat_of.get(g) == code}
                    row.setdefault('bus_train_gt', {})[label] = dict(
                        input_detected=len(base_c), detected=len(hit_c),
                        recovered=sorted(hit_c - base_c), lost=sorted(base_c - hit_c),
                        gt_total=sum(1 for a in annotations
                                     if a['category_id'] == code and a['image_id'] in {i['id'] for i in scopes[scope]}))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

    # 控制台摘要
    order = ['input', 'v4', 'C0', 'C1'] + args.external + [f'{n}+input_fallback' for n in args.external if f'{n}+input_fallback' in result['methods']]
    for scope in ('rain', 'snow', 'all406', 'rain_fog200'):
        print(f'\n=== {scope} ===')
        print(f"{'方法':<34}{'图片':>5}{'mAP50':>9}{'mAP50:95':>10}{'bus':>8}{'train':>8}{'R(.25)':>9}{'FP/图':>8}{'OPR':>8}")
        for name in order:
            if name not in result['methods']:
                continue
            row = result['methods'][name]['scopes'].get(scope)
            if not row:
                continue
            if not row.get('available'):
                print(f'{name:<34}{row["n_images"]:>5}{"N/A":>9}')
                continue
            s, r = row['standard'], row['runtime']
            opr = r.get('opr')
            print(f"{name:<34}{row['n_images']:>5}{s['map50']:>9.6f}{s['map50_95']:>10.6f}"
                  f"{s['per_class']['bus']['ap50']:>8.4f}{s['per_class']['train']['ap50']:>8.4f}"
                  f"{r['recall']:>9.4f}{r['fp_per_image']:>8.3f}{('%.4f' % opr) if opr is not None else '-':>8}")
    print('\n已写入', out)


if __name__ == '__main__':
    main()
