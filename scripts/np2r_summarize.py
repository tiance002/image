"""NP2-R CPU-only scoring of corrected predictions; historical evidence is read-only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from saferestore.metrics.next_phase import (CLASSES, coco_evaluate, normalize_ground_truth,
                                            precision_summary, resample_summary, runtime_counts)

OUT = ROOT / 'results/np2_revision'
FIX = 'histoformer_real_fusionfix'
OLD = 'histoformer_real_legacy_invalidated'


def write_new(path, value):
    if path.exists():
        raise FileExistsError(path)
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    path.write_text(data, encoding='utf-8', newline='\n')
    assert json.loads(path.read_text(encoding='utf-8')) == value


def load_methods():
    folders = {n: ROOT / 'results/next_phase/predictions' / n for n in ('input', 'v4', 'C0', 'C1')}
    folders[FIX] = OUT / 'predictions' / FIX
    folders[OLD] = ROOT / 'results/np2_external_probe/predictions/histoformer_real'
    methods, hashes = {}, {}
    for name, folder in folders.items():
        methods[name] = {}
        for protocol, filename in [('ap001', 'ap001.json'), ('runtime025', 'runtime025.json')]:
            path = folder / filename
            value = json.loads(path.read_text(encoding='utf-8'))
            rows = value['per_image']
            assert len({r['image_id'] for r in rows}) == len(rows)
            methods[name][protocol] = rows
            hashes[f'{name}/{filename}'] = hashlib.sha256(path.read_bytes()).hexdigest()
        if name in (FIX, OLD):
            assert len(methods[name]['ap001']) == len(methods[name]['runtime025']) == 200
            assert {r['weather'] for r in methods[name]['ap001']} == {'rain', 'snow'}
    return methods, hashes


def score():
    payload = json.loads((ROOT / 'data/raw/ACDC_detection/gt_detection/instancesonly_val_gt_detection.json').read_text(encoding='utf-8'))
    images, annotations = normalize_ground_truth(payload)
    methods, hashes = load_methods()
    result = dict(prediction_sha256=hashes,
                  protocol={'ap': 'conf.001 COCO IoU.50:.95 maxDets100/category',
                            'runtime': 'conf.25 COCO IoU.50 maxDets300/category; global prediction maxDet300',
                            'nms': 'same frozen YOLOv8n FP32, class-aware iou.7'},
                  historical_status={OLD: '旧版本待修订：融合缺陷版，仅供修复前后比较'},
                  limitation='rain/snow val used for model selection; correlated frames/sequences, not independent test. net_g_real is real-snow checkpoint; rain is transfer stress evaluation.',
                  scopes={})
    matrices = []
    for scope, weathers in [('rain', {'rain'}), ('snow', {'snow'}), ('rain_snow200', {'rain', 'snow'})]:
        subset = [i for i in images if i['file_name'].split('/')[0] in weathers]
        ids = {i['id'] for i in subset}
        gt = [a for a in annotations if a['image_id'] in ids]
        bucket, hits_by_protocol = {}, {'ap001': {}, 'runtime025': {}}
        for name, preds in methods.items():
            rows = {p: [r for r in v if r['image_id'] in ids] for p, v in preds.items()}
            ap = coco_evaluate(subset, gt, rows['ap001'])
            rt = coco_evaluate(subset, gt, rows['runtime025'], .25, 300)
            runtime = runtime_counts(rt)
            hits_by_protocol['ap001'][name] = runtime_counts(ap)['hits']
            hits_by_protocol['runtime025'][name] = runtime.pop('hits')
            bucket[name] = dict(standard=precision_summary(ap), runtime=runtime)
        for name, row in bucket.items():
            hit = hits_by_protocol['runtime025'][name]
            base = hits_by_protocol['runtime025']['input']
            row['runtime'].update(detected_gt_ids=sorted(hit), opr=len(hit & base)/len(base) if base else None,
                                  gained_input_gt=sorted(hit-base), lost_input_gt=sorted(base-hit))
            row['bus_train_gt'] = {}
            for code in (5, 6):
                valid = {a['id'] for a in gt if a['category_id'] == code and not a.get('iscrowd', 0)}
                own, original = hit & valid, base & valid
                row['bus_train_gt'][CLASSES[code]] = dict(gt=len(valid), detected=len(own), input_detected=len(original),
                                                        recovered=sorted(own-original), lost=sorted(original-own))
        result['scopes'][scope] = dict(n_images=len(subset), methods=bucket)
        if scope != 'rain_snow200':
            for a in gt:
                if a['category_id'] not in (5, 6) or a.get('iscrowd', 0):
                    continue
                for protocol, all_hits in hits_by_protocol.items():
                    matrices.append(dict(weather=scope, image_id=a['image_id'], gt_id=a['id'],
                                         source_gt_id=a['source_annotation_id'], category=CLASSES[a['category_id']],
                                         protocol=protocol, maxDet=100 if protocol=='ap001' else 300,
                                         **{name: 'hit' if a['id'] in values else 'miss' for name, values in all_hits.items()}))
    assert len({(r['image_id'], r['gt_id'], r['protocol']) for r in matrices}) == len(matrices)
    path = OUT / 'bus_train_gt_matrix.csv'
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(matrices[0]))
        writer.writeheader(); writer.writerows(matrices)
    assert len(list(csv.DictReader(path.open(encoding='utf-8')))) == len(matrices)
    write_new(OUT / 'metrics.json', result)
    print('Metrics and dual-protocol GT matrix saved', flush=True)


def bootstrap(count):
    payload = json.loads((ROOT / 'data/raw/ACDC_detection/gt_detection/instancesonly_val_gt_detection.json').read_text(encoding='utf-8'))
    images, annotations = normalize_ground_truth(payload)
    methods, prediction_hashes = load_methods()
    results = dict(resamples=count, seed=20261025, limitation='paired image percentile CI; val with correlated frames. Crossing zero is not equivalence or no-harm proof.', scopes={})
    keys = ['map50', 'map50_95', *CLASSES.values()]
    for weather in ['rain', 'snow']:
        target = OUT / f'bootstrap_{weather}.json'
        fingerprint = dict(resamples=count, seed=20261025, weather=weather,
                           prediction_sha256=prediction_hashes,
                           annotation_sha256=hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest())
        if target.exists():
            cached = json.loads(target.read_text(encoding='utf-8'))
            if cached.get('fingerprint') != fingerprint:
                raise ValueError(f'Preserve incompatible bootstrap cache: {target}')
            results['scopes'][weather] = cached['comparisons']
            continue
        subset = [i for i in images if i['file_name'].split('/')[0] == weather]
        ids = {i['id'] for i in subset}; gt = [a for a in annotations if a['image_id'] in ids]
        evaluators = {n: coco_evaluate(subset, gt, [r for r in v['ap001'] if r['image_id'] in ids]) for n, v in methods.items()}
        points = {n: resample_summary(e, np.arange(len(subset))) for n, e in evaluators.items()}
        pairs = [(FIX, n) for n in methods if n != FIX]
        samples = {f'{a}:{b}': {k: [] for k in keys} for a, b in pairs}
        rng = np.random.default_rng(20261025)
        for index in range(count):
            indices = rng.choice(len(subset), len(subset), replace=True)
            values = {n: resample_summary(e, indices) for n, e in evaluators.items()}
            for a, b in pairs:
                for k in keys:
                    samples[f'{a}:{b}'][k].append(values[a][k]-values[b][k])
            if (index+1) % 250 == 0:
                print(weather, 'paired bootstrap', index+1, '/', count, flush=True)
        summary = {}
        for a, b in pairs:
            pair = f'{a}:{b}'; summary[pair] = {}
            for k, series in samples[pair].items():
                finite = np.asarray(series)[np.isfinite(series)]
                delta = points[a][k]-points[b][k]
                summary[pair][k] = dict(delta=float(delta) if np.isfinite(delta) else None,
                                        ci95=np.quantile(finite, [.025, .975]).tolist() if len(finite) else None,
                                        valid=int(len(finite)))
        write_new(target, dict(fingerprint=fingerprint, comparisons=summary))
        results['scopes'][weather] = summary
    write_new(OUT / 'bootstrap.json', results)


def compare_png_outputs():
    """Compare newly inferred PNGs with preserved invalid fusion outputs, never rescale them."""
    import cv2
    metadata = json.loads((OUT/'restored'/FIX/'metadata.json').read_text(encoding='utf-8'))
    old_root = ROOT/'results/np2_external_probe/restored/histoformer_real'
    new_root = OUT/'restored'/FIX
    rows = []
    for entry in metadata['per_image']:
        arrays = []
        for folder in [old_root, new_root]:
            bgr = cv2.imdecode(np.frombuffer((folder/entry['file_name']).read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            assert bgr is not None and bgr.shape == (1080,1920,3)
            arrays.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        old, new = arrays
        diff = np.abs(old.astype(np.int16)-new.astype(np.int16))
        corners = lambda x: [x[y,z].tolist() for y,z in [(0,0),(0,-1),(-1,0),(-1,-1)]]
        rows.append(dict(image_id=entry['image_id'], file_name=entry['file_name'], weather=entry['weather'],
                         mean_abs_u8=float(diff.mean()), max_abs_u8=int(diff.max()),
                         changed_pixel_fraction=float((diff.max(-1)>0).mean()),
                         old_corners_rgb=corners(old), new_corners_rgb=corners(new),
                         changed_frac_maxchan_gt_0p02=entry['changed_frac_maxchan_gt_0p02']))
    assert len(rows)==200
    write_new(OUT/'fusion_png_changes.json', dict(n=200, per_image=rows,
        mean_abs_u8=float(np.mean([r['mean_abs_u8'] for r in rows])),
        maximum_abs_u8=max(r['max_abs_u8'] for r in rows),
        changed_frac_maxchan_gt_0p02_median=float(np.median([r['changed_frac_maxchan_gt_0p02'] for r in rows])),
        note='New forward from same official checkpoint; no amplification of quantized historical PNGs.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resamples', type=int, default=1000)
    args = parser.parse_args()
    if not (OUT/'metrics.json').exists():
        score()
    if not (OUT/'fusion_png_changes.json').exists():
        compare_png_outputs()
    bootstrap(args.resamples)
