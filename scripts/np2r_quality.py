"""Recompute only fusion-affected Histoformer quality and small crop checks.

No training, no reference-model reinference, no downloads. All outputs belong
under results/np2_revision. The old np2_external_probe evidence stays untouched.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.np2_histoformer_probe import build_model
from scripts.np2_histoformer_restore import forward_patch, read_rgb, restore_tiled, tile_positions
from scripts.np2_quality_lpips import quality_metrics


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                         encoding='utf-8', newline='\n')
    assert json.loads(temporary.read_text(encoding='utf-8'))
    temporary.replace(path)
    assert path.is_file()


def mean_metrics(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in ('psnr', 'ssim', 'lpips')}


def seam_mask(height, width, tile, overlap, radius=16):
    ys = tile_positions(height, tile, tile - overlap)
    xs = tile_positions(width, tile, tile - overlap)
    y_edges = sorted({v for y in ys for v in (y, min(y + tile, height)) if 0 < v < height})
    x_edges = sorted({v for x in xs for v in (x, min(x + tile, width)) if 0 < v < width})
    mask = np.zeros((height, width), dtype=bool)
    for y in y_edges:
        mask[max(0, y-radius):min(height, y+radius), :] = True
    for x in x_edges:
        mask[:, max(0, x-radius):min(width, x+radius)] = True
    return mask, {'tile_starts_y': ys, 'tile_starts_x': xs,
                  'interior_tile_edges_y': y_edges, 'interior_tile_edges_x': x_edges,
                  'band_radius_pixels': radius}


def consistency(net, device, weights, out):
    target = out / 'tiling_consistency_fusionfix.json'
    assert not target.exists(), 'Existing consistency output is preserved'
    old_path = ROOT / 'results/np2_external_probe/tiling_consistency.json'
    old = json.loads(old_path.read_text(encoding='utf-8'))
    records = []
    for old_case in old['cases']:
        if old_case['weather'] not in ('rain', 'snow'):
            continue
        image = read_rgb(ROOT / 'data/raw/ACDC_images/rgb_anon' / old_case['file_name']).astype(np.float32) / 255
        top, left, height, width = old_case['crop']
        crop = image[top:top+height, left:left+width].copy()
        assert crop.shape == (1024, 1024, 3), crop.shape
        torch.cuda.synchronize() if device == 'cuda' else None
        begin = time.perf_counter()
        single = forward_patch(net, crop, device, False)
        tiled, tiles, min_weight = restore_tiled(net, crop, device, 640, 64, False)
        torch.cuda.synchronize() if device == 'cuda' else None
        difference = np.abs(single - tiled)
        seam, boundaries = seam_mask(height, width, 640, 64)
        mse = float(np.mean((single.astype(np.float64) - tiled.astype(np.float64)) ** 2))
        assert np.isfinite(single).all() and np.isfinite(tiled).all()
        records.append(dict(weather=old_case['weather'], file_name=old_case['file_name'],
            crop=old_case['crop'], tiles=tiles, fusion_min_weight=min_weight,
            max_abs_diff=float(difference.max()), mean_abs_diff=float(difference.mean()),
            p999_abs_diff=float(np.quantile(difference, .999)),
            psnr_full_vs_tiled=None if mse == 0 else float(-10*np.log10(mse)),
            corner_abs_differences={name: difference[y, x].tolist() for name, y, x in
                [('top_left', 0, 0), ('top_right', 0, -1), ('bottom_left', -1, 0), ('bottom_right', -1, -1)]},
            seam_mean_abs_diff=float(difference[seam].mean()),
            interior_mean_abs_diff=float(difference[~seam].mean()),
            boundaries=boundaries, elapsed_seconds=time.perf_counter()-begin,
            legacy_fusion_result_status='旧版本待修订; old maximum included fusion-normalization defect'))
        print('corrected tiling:', records[-1]['weather'], records[-1]['max_abs_diff'], flush=True)
        del single, tiled
    save(target, dict(status='completed', weights_sha256=sha(weights), dtype='float32',
        official_preprocessing='RGB float[0,1]; right/bottom reflect to multiple8; clamp; no quantization',
        old_case_source=str(old_path.relative_to(ROOT)), old_case_source_sha256=sha(old_path),
        comparison='single1024 forward vs corrected640 overlap64; identical fixed rain/snow crops',
        limitation='Real network contexts differ. This is not a numerical-equivalence assertion. Full1920 forward not attempted.',
        cases=records))
    return target


def quality(net, device, weights, out, resume):
    import lpips
    from importlib.metadata import version
    final = out / 'quality_lpips_fusionfix.json'
    progress = out / 'quality_lpips_fusionfix.progress.json'
    assert not final.exists(), 'Existing final quality output is preserved'
    manifest = ROOT / 'data/real_weather_rgb_v2/manifests/dev.csv'
    old_path = ROOT / 'results/np2_external_probe/quality_lpips.json'
    old = json.loads(old_path.read_text(encoding='utf-8'))
    with manifest.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 200
    alex_path = Path(torch.hub.get_dir()) / 'checkpoints/alexnet-owt-7be5be79.pth'
    linear_path = Path(lpips.__file__).parent / 'weights/v0.1/alex.pth'
    assert alex_path.is_file() and linear_path.is_file(), 'No download: required LPIPS weights must already be cached'
    fingerprint = dict(weights_sha256=sha(weights), manifest_sha256=sha(manifest),
                       lpips_alex_sha256=sha(alex_path), lpips_linear_sha256=sha(linear_path))
    records = []
    if progress.exists():
        assert resume, 'Use --resume-quality to resume own partial output without losing evidence'
        partial = json.loads(progress.read_text(encoding='utf-8'))
        assert partial['fingerprint'] == fingerprint
        records = partial['per_image']
        assert [r['scene_id'] for r in records] == [r['scene_id'] for r in rows[:len(records)]]
    perceptual = lpips.LPIPS(net='alex', verbose=False).to(device).eval().requires_grad_(False)
    start = time.perf_counter()
    for row in rows[len(records):]:
        rgb = read_rgb(ROOT / row['degraded_path']).astype(np.float32) / 255
        clean = read_rgb(ROOT / row['clean_path']).astype(np.float32) / 255
        assert rgb.shape == clean.shape
        torch.cuda.synchronize() if device == 'cuda' else None
        begin = time.perf_counter()
        restored, tiles, min_weight = restore_tiled(net, rgb, device, 640, 64, False)
        torch.cuda.synchronize() if device == 'cuda' else None
        restore_ms = 1000 * (time.perf_counter()-begin)
        assert np.isfinite(restored).all() and restored.shape == rgb.shape
        quantized = np.rint(restored * 255).clip(0,255).astype(np.uint8).astype(np.float32) / 255
        prediction = torch.from_numpy(quantized.transpose(2,0,1).copy())[None].to(device)
        target = torch.from_numpy(clean.transpose(2,0,1).copy())[None].to(device)
        with torch.inference_mode():
            metrics = quality_metrics(prediction, target)
            metrics['lpips'] = float(perceptual(prediction*2-1, target*2-1))
            float_prediction = torch.from_numpy(restored.transpose(2,0,1).copy())[None].to(device)
            float_metrics = quality_metrics(float_prediction, target)
            float_metrics['lpips'] = float(perceptual(float_prediction*2-1, target*2-1))
        records.append(dict(scene_id=row['scene_id'], weather=row['weather'], sequence=row['sequence'],
            degraded_path=row['degraded_path'].replace(chr(92),'/'), clean_path=row['clean_path'].replace(chr(92),'/'),
            **metrics, float_metrics=float_metrics, tiles=tiles, fusion_min_weight=min_weight, restore_ms=restore_ms,
            quantization_abs_mean=float(np.abs(quantized-restored).mean()),
            diff_mean=float(np.abs(restored-rgb).mean()),
            changed_frac_maxchan_gt_0p02=float((np.abs(restored-rgb).max(axis=2)>.02).mean())))
        if len(records) % 10 == 0 or len(records) == len(rows):
            save(progress, dict(status='partial' if len(records)<len(rows) else 'inference_completed',
                                fingerprint=fingerprint, completed=len(records), per_image=records))
            print('fusionfix dev',len(records),'/',len(rows),'wall_seconds',round(time.perf_counter()-start,1),flush=True)
    by_weather = {w: mean_metrics([r for r in records if r['weather']==w]) for w in ('rain','fog')}
    references = {}
    for name in ('input','v4','C0','C1'):
        source = old['methods'][name]
        assert len(source['per_image']) == len(rows)
        for actual, row in zip(source['per_image'],rows):
            assert actual['weather']==row['weather'] and actual['sequence']==row['sequence']
        references[name] = dict(source='reused unchanged old quality_lpips.json; no new model inference',
            mean=source['mean'], by_weather=source['by_weather'],
            per_image=[dict(**r, scene_id=row['scene_id'],degraded_path=row['degraded_path'].replace(chr(92),'/'))
                       for r,row in zip(source['per_image'],rows)])
    payload = dict(status='completed',fingerprint=fingerprint, method='histoformer_real_fusionfix',
        n=len(rows), manifest=str(manifest.relative_to(ROOT)), dtype='float32', tile=640, overlap=64,
        metric='PSNR allpixels; SSIM11x11 sigma1.5 valid; LPIPS alex(v0.1) RGB[-1,1]',
        quantization='identical to PNG: round(255*x), clip, uint8, /255; no intermediate PNGs saved',
        scope='aligned synthetic dev fog100/rain100; rain transfer probe; fog out-of-scope stress, not official supported task; no real snow/night aligned quality',
        by_weather=by_weather, mean=mean_metrics(records),
        float_by_weather={w:mean_metrics([r['float_metrics'] for r in records if r['weather']==w]) for w in ('rain','fog')},
        float_mean=mean_metrics([r['float_metrics'] for r in records]),
        comparison_precision_note='Old internal reference outputs used FP32 floats, external old output used PNG uint8. Use new float_metrics for float-vs-float reference comparison; main fields preserve old external PNG protocol.',
        pooled_mean_label='legacy dev200 aggregate including out-of-scope fog; not supported-task score',
        references=references, reference_source_sha256=sha(old_path),
        old_histoformer_status='旧版本待修订; old fused Histoformer quality invalidated, not reused',
        mean_restore_ms=float(np.mean([r['restore_ms'] for r in records])),
        median_restore_ms=float(np.median([r['restore_ms'] for r in records])),
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else None,
        versions=dict(torch=torch.__version__, numpy=np.__version__, lpips=version('lpips')),
        per_image=records)
    save(final,payload)
    assert fingerprint['weights_sha256']==sha(weights)
    return final


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['quality','consistency','both'],default='both')
    parser.add_argument('--weights',default='third_party/weights/histoformer_official_net_g_real.pth')
    parser.add_argument('--out-root',default='results/np2_revision')
    parser.add_argument('--resume-quality',action='store_true')
    args=parser.parse_args()
    out=(ROOT/args.out_root).resolve()
    assert out.is_relative_to((ROOT/'results/np2_revision').resolve()), 'Revision outputs only'
    out.mkdir(parents=True,exist_ok=True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(4)
    weights=(ROOT/args.weights).resolve()
    weight_sha=sha(weights)
    net=build_model(weights,device).eval().requires_grad_(False)
    if args.mode in ('consistency','both'):
        print('Wrote',consistency(net,device,weights,out),flush=True)
    if args.mode in ('quality','both'):
        print('Wrote',quality(net,device,weights,out,args.resume_quality),flush=True)
    assert weight_sha==sha(weights)


if __name__=='__main__':
    main()
