"""下一阶段：冻结低阈值 AP/运行时双协议并保存全部逐图预测。"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]

import cv2
import numpy as np
import torch
import yaml
import ultralytics
from ultralytics import YOLO
from saferestore.metrics.next_phase import (CLASSES, normalize_ground_truth, coco_evaluate, precision_summary, runtime_counts, resample_summary)
from scripts.train_restorer import imread_rgb, restore_tiled
from saferestore.models.nafnet import build_nafnet


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def infer(name, checkpoint=None, limit=0):
    config = yaml.safe_load((ROOT/'configs/next_phase_evaluation.yaml').read_text(encoding='utf-8'))
    out = ROOT/'results/next_phase/predictions'/name
    if out.exists():
        raise FileExistsError(f'不覆盖评估产物：{out}')
    out.mkdir(parents=True)
    payload = json.loads((ROOT/config['annotations']).read_text(encoding='utf-8'))
    images, annotations = normalize_ground_truth(payload)
    if limit:
        images = images[:limit]
    weights = ROOT/config['model']
    if not weights.is_file():
        raise FileNotFoundError('只接受本地YOLO权重')
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    detector = YOLO(str(weights))
    detector_params = sum(p.numel() for p in detector.model.parameters())
    model = None
    if checkpoint:
        path = ROOT/checkpoint
        if not path.is_file():
            raise FileNotFoundError(path)
        state = torch.load(path, map_location='cpu', weights_only=False)
        model = build_nafnet().to(device).eval()
        model.load_state_dict(state['ema'])
        del state
    torch.cuda.reset_peak_memory_stats() if device == 'cuda' else None
    all_rows, all_runtime, timing = [], [], []
    by_id = {i['id']: [] for i in images}
    for a in annotations:
        if a['image_id'] in by_id:
            by_id[a['image_id']].append(a)
    raw_capture = {}
    current = {}
    # 捕捉当前 eval forward 的 decoded 输出，只有 bus/train 图像保存 pre-NMS。
    def hook(module, inputs, output):
        if current.get('focus'):
            decoded = output[0] if isinstance(output, tuple) else output
            if isinstance(decoded, torch.Tensor) and decoded.ndim == 3 and decoded.shape[1] == 84:
                raw_capture['decoded'] = decoded.detach().float().cpu()
                raw_capture['shape'] = tuple(inputs[0].shape[-2:])
    handle = detector.model.register_forward_hook(hook)
    for index, image in enumerate(images):
        original = imread_rgb(ROOT/config['images_root']/image['file_name'])
        current['focus'] = any(a['category_id'] in (5,6) for a in by_id[image['id']])
        bgr_input = cv2.cvtColor((original*255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
        if device == 'cuda': torch.cuda.synchronize()
        begin = time.perf_counter()
        if model is not None:
            tensor = torch.from_numpy(original.transpose(2,0,1).copy())[None].to(device)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=device=='cuda'):
                restored = restore_tiled(model, tensor)
            if not torch.isfinite(restored).all():
                raise FloatingPointError(f'复原输出非有限：{name} {image["file_name"]}')
            rgb = (restored[0].float().clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
            source = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            del tensor, restored
        else:
            source = bgr_input
        if device == 'cuda': torch.cuda.synchronize()
        restore_time = time.perf_counter()-begin
        rows = []
        for confidence in (config['ap_confidence'], config['runtime_confidence']):
            if device == 'cuda': torch.cuda.synchronize()
            detector_begin = time.perf_counter()
            raw_capture.clear()
            result = detector.predict(source=source, imgsz=640, conf=confidence,
                iou=config['nms_iou'], agnostic_nms=False, max_det=config['max_det'],
                classes=list(CLASSES), device=0 if device=='cuda' else 'cpu',
                rect=True, verbose=False)[0]
            if device == 'cuda': torch.cuda.synchronize()
            detector_time = time.perf_counter()-detector_begin
            predictions = {}
            for box, score, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
                predictions.setdefault(str(int(cls)), []).append(dict(box=box.tolist(), confidence=float(score)))
            row = dict(image_id=image['id'], file_name=image['file_name'], weather=image['file_name'].split('/')[0],
                width=image['width'], height=image['height'], predictions=predictions)
            if confidence == config['ap_confidence'] and 'decoded' in raw_capture:
                decoded = raw_capture['decoded'][0]
                scores, labels = decoded[4:].max(0)
                chosen = (scores>=0.001) & torch.isin(labels, torch.tensor(list(CLASSES)))
                b = decoded[:4,chosen].T
                xyxy = torch.cat((b[:,:2]-b[:,2:]/2, b[:,:2]+b[:,2:]/2),1)
                ih, iw = raw_capture['shape']
                gain = min(ih/image['height'], iw/image['width'])
                pad_x = (iw-image['width']*gain)/2
                pad_y = (ih-image['height']*gain)/2
                xyxy[:,[0,2]] = (xyxy[:,[0,2]]-pad_x)/gain
                xyxy[:,[1,3]] = (xyxy[:,[1,3]]-pad_y)/gain
                xyxy[:,[0,2]] = xyxy[:,[0,2]].clamp(0,image['width'])
                xyxy[:,[1,3]] = xyxy[:,[1,3]].clamp(0,image['height'])
                row['pre_nms'] = [dict(box=b.tolist(), confidence=float(s), category_id=int(c))
                    for b,s,c in zip(xyxy,scores[chosen],labels[chosen])]
            rows.append(row)
            if confidence == config['runtime_confidence']:
                timing.append(dict(image_id=image['id'], restore_ms=restore_time*1000,
                    detect_runtime_ms=detector_time*1000, end_to_end_runtime_ms=(restore_time+detector_time)*1000))
        all_rows.append(rows[0]); all_runtime.append(rows[1])
        if (index+1)%25==0 or index+1==len(images):
            print(f'{name} eval {index+1}/{len(images)} elapsed {sum(t["end_to_end_runtime_ms"] for t in timing)/1000:.1f}s',flush=True)
    handle.remove()
    for filename, rows in [('ap001.json',all_rows), ('runtime025.json',all_runtime)]:
        (out/filename).write_text(json.dumps(dict(protocol=config, per_image=rows),ensure_ascii=False),encoding='utf-8')
    times = timing[5:] if len(timing)>5 else timing
    metadata = dict(method=name, checkpoint=checkpoint, checkpoint_sha256=sha(ROOT/checkpoint) if checkpoint else None,
        detector_sha256=sha(weights), annotation_sha256=sha(ROOT/config['annotations']),
        protocol=config, python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
        ultralytics=ultralytics.__version__, opencv=cv2.__version__, images=len(images),
        restoration_parameters=sum(p.numel() for p in model.parameters()) if model else 0,
        detector_parameters=detector_params, device=torch.cuda.get_device_name() if device=='cuda' else 'cpu',
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else None,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if device=='cuda' else None,
        mean_e2e_runtime_ms=float(np.mean([t['end_to_end_runtime_ms'] for t in times])),
        median_e2e_runtime_ms=float(np.median([t['end_to_end_runtime_ms'] for t in times])),
        latency_note='Sequential batch1; CPU decode excluded; restoration+CPU quantize/color+D2H+runtime YOLO/NMS included; first5 excluded; AP pass not included', per_image_timing=timing)
    (out/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    return metadata


def score(names):
    config = yaml.safe_load((ROOT/'configs/next_phase_evaluation.yaml').read_text(encoding='utf-8'))
    images, annotations = normalize_ground_truth(json.loads((ROOT/config['annotations']).read_text(encoding='utf-8')))
    scopes = dict(all406=images, rain_fog200=[i for i in images if i['file_name'].split('/')[0] in ('fog','rain')])
    scopes.update({w:[i for i in images if i['file_name'].split('/')[0]==w] for w in ('fog','rain','night','snow')})
    result = dict(protocol=config, annotation_id_policy='raw_id+1; source_annotation_id preserved; image IDs unchanged', methods={})
    frozen_signature = None
    for name in names:
        folder = ROOT/'results/next_phase/predictions'/name
        ap = json.loads((folder/'ap001.json').read_text(encoding='utf-8'))['per_image']
        runtime = json.loads((folder/'runtime025.json').read_text(encoding='utf-8'))['per_image']
        # 独立运行时 NMS 与低阈值后过滤通常等价，但必须核查，不默认假定。
        different = 0
        for a,b in zip(ap,runtime):
            filtered = {k:[p for p in v if p['confidence']>=.25] for k,v in a['predictions'].items()}
            filtered = {k:v for k,v in filtered.items() if v}
            different += int(filtered != b['predictions'])
        method = dict(metadata=json.loads((folder/'metadata.json').read_text(encoding='utf-8')), scopes={},
                      lowconf_filtered_vs_runtime_different_images=different)
        metadata = method['metadata']
        signature = {key: metadata[key] for key in ('detector_sha256','annotation_sha256','protocol',
            'python','torch','numpy','ultralytics','opencv','device')}
        if metadata['protocol'] != config:
            raise ValueError(f'{name}预测协议与当前配置不一致')
        if frozen_signature is not None and signature != frozen_signature:
            raise ValueError(f'{name}的检测器/标注/依赖/设备与对照不一致')
        frozen_signature = signature
        for scope, subset in scopes.items():
            ids = {i['id'] for i in subset}
            gt = [a for a in annotations if a['image_id'] in ids]
            e = coco_evaluate(subset,gt,[r for r in ap if r['image_id'] in ids])
            er = coco_evaluate(subset,gt,[r for r in runtime if r['image_id'] in ids],.25)
            er_runtime = coco_evaluate(subset,gt,[r for r in runtime if r['image_id'] in ids],.25,300)
            counts = runtime_counts(er_runtime)
            hits = counts.pop('hits')
            counts['detected_gt_ids'] = sorted(hits)
            method['scopes'][scope] = dict(n_images=len(subset),standard=precision_summary(e,True),
                legacy_threshold_same_coco=precision_summary(er),runtime=counts)
        result['methods'][name] = method
        print(name,'all406',method['scopes']['all406']['standard']['map50'],flush=True)
    if 'input' in result['methods']:
        for method in result['methods'].values():
            for scope,row in method['scopes'].items():
                base = set(result['methods']['input']['scopes'][scope]['runtime']['detected_gt_ids'])
                hit = set(row['runtime']['detected_gt_ids'])
                row['runtime']['opr'] = len(base & hit)/len(base) if base else None
                row['runtime']['lost_input_gt'] = sorted(base-hit)
                row['runtime']['gained_input_gt'] = sorted(hit-base)
    (ROOT/'results/next_phase/metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name')
    parser.add_argument('--checkpoint')
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--score',nargs='+')
    args = parser.parse_args()
    if args.score: score(args.score)
    elif args.name: infer(args.name,args.checkpoint,args.limit)
    else: parser.error('需要 --name 或 --score')


if __name__=='__main__':
    main()
