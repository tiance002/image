"""NP2-R: fixed 24 train-frame scale-order diagnostic, without weight updates.

Main A: float32 INTER_LINEAR resize -> restore unpadded image -> common detector pad.
Main B: full-size float32 restoration -> identical resize -> common detector pad.
D: direct input using the same resize/pad. A_pad is a separately named historical
pipeline that restores gray borders. Symmetric uint8 variants quantize the common
640 canvas; bf16 and formal pre-resize uint8 are separately named perturbations.
Full-size B uses the established 1024/64 tiling; its context differs from A and
is explicitly not isolated by merely aligning GT coordinates.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from saferestore.data.detection_gt import ACDCDetectionDataset, ACDC_TO_COCO, letterbox_rgb
from saferestore.losses.task_detection import FrozenYoloTaskLoss
from saferestore.metrics.next_phase import CLASSES, coco_evaluate, precision_summary, runtime_counts
from saferestore.models.nafnet import build_nafnet
from scripts.train_restorer import restore_tiled
from scripts.np2r_supervision_probe import batch_forward, decomposed_total
SIZE = 640
PATHS = ('D_fp32', 'A_fp32', 'B_fp32', 'D_u8', 'A_u8', 'B_u8',
         'A_pad_historical_fp32', 'B_bf16_float', 'B_restoration_bf16_prequant_square')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    path.write_text(text + '\n', encoding='utf-8', newline='\n')
    assert json.loads(path.read_text(encoding='utf-8')) == json.loads(text)


def geometry(height, width, size=SIZE):
    scale = min(size / height, size / width)
    nw, nh = round(width * scale), round(height * scale)
    return dict(height=height, width=width, size=size, resized_width=nw, resized_height=nh,
                left=(size - nw) // 2, top=(size - nh) // 2, sx=nw / width, sy=nh / height)


def resize_float(rgb, geom):
    assert rgb.dtype == np.float32 and rgb.ndim == 3
    return cv2.resize(rgb, (geom['resized_width'], geom['resized_height']), interpolation=cv2.INTER_LINEAR)


def common_pad(resized, geom):
    assert resized.dtype == np.float32
    assert resized.shape == (geom['resized_height'], geom['resized_width'], 3)
    canvas = np.full((geom['size'], geom['size'], 3), np.float32(114 / 255), dtype=np.float32)
    top, left = geom['top'], geom['left']
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return canvas


def letterbox_float(rgb, geom=None):
    geom = geometry(*rgb.shape[:2]) if geom is None else geom
    return common_pad(resize_float(rgb, geom), geom), geom


def transform_boxes(boxes, geom):
    values = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    values[:, [0, 2]] = values[:, [0, 2]] * geom['sx'] + geom['left']
    values[:, [1, 3]] = values[:, [1, 3]] * geom['sy'] + geom['top']
    return values


def quantize_canvas(canvas):
    return (canvas * 255).round().clip(0, 255).astype(np.uint8).astype(np.float32) / np.float32(255)


def tensor_rgb(rgb, device):
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))[None].to(device).float()


def array_rgb(tensor):
    return tensor[0].float().clamp(0, 1).permute(1, 2, 0).cpu().numpy().astype(np.float32)


def size_bucket(box):
    width, height = max(0., box[2] - box[0]), max(0., box[3] - box[1])
    area = width * height
    return dict(width=width, height=height, area=area,
                coco_area='small' if area < 32 ** 2 else 'medium' if area < 96 ** 2 else 'large',
                min_side_bin='<8' if min(width, height) < 8 else '8-<16' if min(width, height) < 16
                else '16-<32' if min(width, height) < 32 else '>=32')


def anchor_centers_in_box(box, size=SIZE):
    """Actual center grid; TAL select_candidates_in_gts uses strict positive margins."""
    out = {}
    for stride in (8, 16, 32):
        axis = (np.arange(size // stride, dtype=np.float64) + .5) * stride
        xx, yy = np.meshgrid(axis, axis)
        centers = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
        margins = np.concatenate([centers - np.asarray(box[:2]), np.asarray(box[2:]) - centers], axis=1)
        strict = (margins > 1e-9).all(1)
        closed = (margins >= 0).all(1)
        out[str(stride)] = dict(strict_inside=int(strict.sum()), closed_inside=int(closed.sum()),
                               strict_centers=centers[strict].tolist())
    return out


def load_sample():
    old = ROOT / 'results/np2_external_probe/scale_order_probe.json'
    payload = json.loads(old.read_text(encoding='utf-8'))
    files = payload['sample']['files']
    assert len(files) == 24 and len(set(files)) == 24
    assert all('/train/' in name for name in files)
    return files, dict(source=str(old.relative_to(ROOT)), sha256=sha256(old),
                       n_bus=payload['sample']['n_bus'], n_train=payload['sample']['n_train'])


def cohort_gt(payload, files):
    """Explicit train diagnostic normalization; does not relax val-only production API."""
    selected = {i['id']: i for i in payload['images'] if i['file_name'] in files}
    assert len(selected) == len(files)
    images = sorted(selected.values(), key=lambda x: x['id'])
    anns = []
    by_image = defaultdict(list)
    for original in payload['annotations']:
        if original['image_id'] not in selected or original['category_id'] not in ACDC_TO_COCO:
            continue
        row = copy.deepcopy(original)
        row.pop('segmentation', None)
        row['source_annotation_id'] = int(original['id'])
        row['id'] = int(original['id']) + 1
        row['category_id'] = ACDC_TO_COCO[original['category_id']]
        anns.append(row)
        by_image[row['image_id']].append(row)
    assert len({a['id'] for a in anns}) == len(anns) and all(a['id'] > 0 for a in anns)
    return images, anns, by_image


def make_batch(record, geom):
    xyxy = transform_boxes(record['boxes'], geom) / SIZE
    xywh = xyxy.copy()
    xywh[:, :2] = (xyxy[:, :2] + xyxy[:, 2:]) / 2
    xywh[:, 2:] = xyxy[:, 2:] - xyxy[:, :2]
    ignored = transform_boxes(record['ignore_boxes'], geom) / SIZE
    return dict(batch_idx=torch.zeros(len(xywh)),
                cls=torch.tensor(record['classes'], dtype=torch.float32).reshape(-1, 1),
                bboxes=torch.from_numpy(xywh), ignore_boxes=torch.from_numpy(ignored),
                ignore_batch_idx=torch.zeros(len(ignored), dtype=torch.long))


def inverse_boxes(boxes, geom):
    boxes = boxes.clone()
    boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - geom['left']) / geom['sx']).clamp(0, geom['width'])
    boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - geom['top']) / geom['sy']).clamp(0, geom['height'])
    return boxes


def nms_rows(assigned, criterion, geom, image, threshold):
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.ops import xyxy2xywh
    boxes = assigned['pred_bboxes'] * assigned['stride_tensor']
    probabilities = assigned['pred_scores'].sigmoid()
    decoded = torch.cat([xyxy2xywh(boxes), probabilities], dim=-1).permute(0, 2, 1).contiguous()
    # NMS mutates coordinate storage; every call receives its own decoded tensor.
    result, indices = non_max_suppression(decoded.clone(), conf_thres=threshold, iou_thres=.7,
        classes=list(CLASSES), agnostic=False, max_det=300, nc=80, return_idxs=True)
    detections = result[0]
    mapped = inverse_boxes(detections[:, :4], geom)
    predictions = {}
    for box, item in zip(mapped.cpu().tolist(), detections.cpu().tolist()):
        predictions.setdefault(str(int(item[5])), []).append(dict(box=box, confidence=item[4]))
    return dict(image_id=image['id'], file_name=image['file_name'], weather=image['file_name'].split('/')[0],
                width=image['width'], height=image['height'], predictions=predictions), indices[0].cpu().tolist()


def pairwise_iou(boxes, box):
    inter = np.maximum(0, np.minimum(boxes[:, 2:], box[2:]) - np.maximum(boxes[:, :2], box[:2])).prod(1)
    area = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(1)
    other = np.maximum(0, np.asarray(box[2:]) - box[:2]).prod()
    return inter / np.maximum(area + other - inter, 1e-12)


def low_threshold_details(assigned, annotations, geom):
    boxes = (assigned['pred_bboxes'][0] * assigned['stride_tensor']).cpu().numpy()
    probabilities = assigned['pred_scores'][0].sigmoid().cpu().numpy()
    max_scores, max_labels = probabilities.max(1), probabilities.argmax(1)
    out = []
    for gt in annotations:
        if gt.get('iscrowd', 0):
            continue
        x, y, w, h = gt['bbox']
        original = [x, y, x + w, y + h]
        canvas = transform_boxes([original], geom)[0]
        ious = pairwise_iou(boxes, canvas)
        code = gt['category_id']
        candidates = (max_labels == code) & (max_scores >= .001)
        positions = np.flatnonzero(candidates)
        best = int(positions[np.argmax(ious[positions])]) if positions.size else None
        overlaps = np.flatnonzero(ious >= .5)
        target = int(overlaps[np.argmax(probabilities[overlaps, code])]) if overlaps.size else None
        out.append(dict(gt_id=gt['id'], source_gt_id=gt['source_annotation_id'], category=code,
            class_name=CLASSES[code], original_xyxy=original, detector_xyxy=canvas.tolist(),
            original_size=size_bucket(original), detector_size=size_bucket(canvas.tolist()),
            actual_center_coverage=anchor_centers_in_box(canvas.tolist()),
            low_threshold_best_class_candidate=dict(anchor_index=best,
                iou=float(ious[best]) if best is not None else 0.,
                score=float(max_scores[best]) if best is not None else None),
            highest_target_class_score_with_iou_ge_0p5=dict(anchor_index=target,
                score=float(probabilities[target, code]) if target is not None else None,
                iou=float(ious[target]) if target is not None else None,
                argmax_class=int(max_labels[target]) if target is not None else None)))
    return out


def gt_match_details(evaluator):
    details = {}
    fps = []
    for result in evaluator.evalImgs:
        if result is None:
            continue
        for gid, ignored, did in zip(result['gtIds'], result['gtIgnore'], result['gtMatches'][0]):
            if ignored:
                continue
            gt = evaluator.cocoGt.anns[int(gid)]
            dt = evaluator.cocoDt.anns[int(did)] if did else None
            if dt:
                g = gt['bbox']; d = dt['bbox']
                iou = float(pairwise_iou(np.asarray([[d[0], d[1], d[0] + d[2], d[1] + d[3]]]),
                                        [g[0], g[1], g[0] + g[2], g[1] + g[3]])[0])
            else:
                iou = None
            details[str(int(gid))] = dict(matched=bool(did), score=dt['score'] if dt else None, iou=iou,
                                         dt_id=int(did) if did else None)
        for did, matched, ignored in zip(result['dtIds'], result['dtMatches'][0], result['dtIgnore'][0]):
            if not matched and not ignored:
                dt = evaluator.cocoDt.anns[int(did)]
                fps.append(dict(image_id=dt['image_id'], category=dt['category_id'],
                                score=dt['score'], bbox=dt['bbox']))
    return details, fps


def evaluate_paths(images, annotations, predictions, direct_key="D_fp32", comparisons=None):
    outputs = {}
    for path, protocols in predictions.items():
        ap = coco_evaluate(images, annotations, protocols['ap001'], threshold=.001, max_dets=100)
        runtime = coco_evaluate(images, annotations, protocols['runtime025'], threshold=.25, max_dets=300)
        counts = runtime_counts(runtime)
        counts['hits'] = sorted(counts['hits'])
        ap_details, _ = gt_match_details(ap)
        runtime_details, fps = gt_match_details(runtime)
        outputs[path] = dict(ap=precision_summary(ap), runtime=counts,
                             ap001_gt=ap_details, runtime025_gt=runtime_details, runtime025_fp=fps)
    direct_hits = set(outputs[direct_key]['runtime']['hits'])
    direct_fp = outputs[direct_key]['runtime025_fp']
    for path, result in outputs.items():
        hits = set(result['runtime']['hits'])
        result['runtime'][f'opr_to_{direct_key}'] = len(hits & direct_hits) / max(len(direct_hits), 1)
        result['runtime'][f'gained_vs_{direct_key}'] = sorted(hits - direct_hits)
        result['runtime'][f'lost_vs_{direct_key}'] = sorted(direct_hits - hits)
        # Unmatched post-NMS FPs with no same-class direct FP IoU >= .5.
        new = []
        for fp in result['runtime025_fp']:
            x, y, w, h = fp['bbox']; box = [x, y, x + w, y + h]
            base = [d for d in direct_fp if d['image_id'] == fp['image_id'] and d['category'] == fp['category']]
            if not base or max(pairwise_iou(np.asarray([[d['bbox'][0], d['bbox'][1], d['bbox'][0] + d['bbox'][2], d['bbox'][1] + d['bbox'][3]] for d in base]), box)) < .5:
                new.append(fp)
        result['runtime'][f'new_fp_vs_{direct_key}_count'] = len(new)
        result['runtime'][f'new_fp_vs_{direct_key}'] = new
    for right, left in (comparisons if comparisons is not None else [('B_fp32', 'A_fp32'), ('B_u8', 'A_u8')]):
        if right not in outputs or left not in outputs:
            continue
        a = set(outputs[left]['runtime']['hits']); b = set(outputs[right]['runtime']['hits'])
        outputs[right]['runtime'][f'gained_vs_{left}'] = sorted(b - a)
        outputs[right]['runtime'][f'lost_vs_{left}'] = sorted(a - b)
    return outputs



def actual_rect_predictions(detector, bgr, image, device):
    """Use the real production YOLO.predict route; no square diagnostic tensor."""
    rows, shapes = {}, {}
    observed = []
    handle = detector.model.register_forward_pre_hook(lambda module, inputs: observed.append(list(inputs[0].shape)))
    try:
        for key, threshold in [('ap001', .001), ('runtime025', .25)]:
            observed.clear()
            result = detector.predict(source=bgr, imgsz=640, rect=True, conf=threshold,
                iou=.7, classes=list(CLASSES), max_det=300, agnostic_nms=False,
                half=False, device=0 if device == 'cuda' else 'cpu', verbose=False, save=False, save_txt=False)[0]
            predictions = {}
            for box, score, category in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
                predictions.setdefault(str(int(category)), []).append(dict(box=box, confidence=score))
            rows[key] = dict(image_id=image['id'], file_name=image['file_name'], weather=image['file_name'].split('/')[0],
                             width=image['width'], height=image['height'], predictions=predictions)
            shapes[key] = observed[-1] if observed else None
            assert shapes[key] is not None, 'Production detector input shape was not observed'
    finally:
        handle.remove()
    return rows, shapes


def bin_summary(per_gt, summaries, path):
    output = {}
    for space in ('original_size', 'detector_size'):
        buckets = defaultdict(lambda: dict(gt=0, matched_ap001=0, matched_runtime025=0))
        for item in per_gt:
            for dimension in ('coco_area', 'min_side_bin'):
                key = dimension + ':' + item[space][dimension]
                entry = buckets[key]; entry['gt'] += 1
                gid = str(item['gt_id'])
                entry['matched_ap001'] += int(summaries[path]['ap001_gt'][gid]['matched'])
                entry['matched_runtime025'] += int(summaries[path]['runtime025_gt'][gid]['matched'])
        output[space] = dict(buckets)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', default=[
        'v4=runs/real_v4_allweather/latest.pt', 'C0=runs/gt_next_phase_20261025_C0/latest.pt',
        'C1=runs/gt_next_phase_20261025_C1/latest.pt'])
    parser.add_argument('--out', default='results/np2_revision/scale_order_probe.json')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    output = ROOT / args.out
    assert output.resolve().is_relative_to((ROOT / 'results/np2_revision').resolve())
    if output.exists():
        raise FileExistsError(f'Preserve existing result: {output}')
    run_dir = output.parent / (output.stem + '_artifacts')
    if run_dir.exists():
        raise FileExistsError(f'Preserve existing run directory: {run_dir}')
    run_dir.mkdir(parents=True)
    files, provenance = load_sample()
    ann_path = ROOT / 'data/raw/ACDC_detection/gt_detection/instancesonly_train_gt_detection.json'
    raw = json.loads(ann_path.read_text(encoding='utf-8'))
    images, annotations, anns_by_image = cohort_gt(raw, files)
    image_by_name = {i['file_name']: i for i in images}
    dataset = ACDCDetectionDataset(ann_path, ROOT / 'data/raw/ACDC_images/rgb_anon', image_size=SIZE, seed=42)
    record_by_name = {r['file_name']: r for r in dataset.records}
    torch.set_num_threads(4); cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Explicit CUDA run requested but GPU unavailable')
    detector_path = ROOT / 'yolov8n.pt'
    loss_fn = FrozenYoloTaskLoss(detector_path, device=device)
    from ultralytics import YOLO
    production_detector = YOLO(str(detector_path))
    production_detector.model.to(device).float().eval().fuse(verbose=False)
    production_detector.model.requires_grad_(False)
    production_state = {k: v.detach().cpu().clone() for k, v in production_detector.model.state_dict().items()}
    detector_state = {k: v.detach().cpu().clone() for k, v in loss_fn.detector.state_dict().items()}
    report = dict(revision='np2r-scale-2', sample=dict(files=files, count=len(files), **provenance),
        protocol=dict(ap_conf=.001, ap_max_dets_per_class=100, runtime_conf=.25, runtime_max_dets=300,
            nms_iou=.7, class_aware=True, global_nms_max_det=300, classes=CLASSES, detection_canvas=[640,640],
            interpolation='cv2.INTER_LINEAR on RGB float32 before/after restoration',
            coordinate_transform='actual integer resize sx/sy and identical left/top; inverse same actual sx/sy',
            main_precision='FP32; TF32 off; autocast off (also retained for separately rect=True detector sidepaths)', main_padding='114/255 only at common detector preprocessing after A restoration',
            perturbations=dict(uint8='D/A/B common 640 canvas round*255 -> uint8 -> float, same location',
                historical='A_pad_historical_fp32 restores the padded gray 640 canvas',
                bf16='B_bf16_float restores full image bf16 then float resize, no quantization',
                restoration_only='B_restoration_bf16_prequant_square uses bf16 full restoration -> uint8 at full resolution -> uint8 square letterbox; detector remains square640',
                actual_production_rect='separate D_original_u8_rect / B_bf16_prequant_rect use YOLO.predict rect=True, imgsz640, FP32 detector, source original-size uint8 BGR; no native square loss attribution'),
            context='A restores one unpadded 360x640 image (NAF internal pad); B full1920x1080 uses1024/64 four tiles; context remains different',
            interpretation='fixed train diagnostic; not validation AP and not independent test; same-sequence correlation remains'),
        resources=dict(annotation_sha256=sha256(ann_path), detector_sha256=sha256(detector_path),
                       script_sha256=sha256(__file__), torch=torch.__version__, opencv=cv2.__version__, numpy=np.__version__), models={})
    write_json(run_dir / 'config.json', {k: v for k, v in report.items() if k != 'models'})
    for spec in args.models:
        name, path = spec.split('=', 1)
        checkpoint = ROOT / path
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model = build_nafnet().to(device).float().eval()
        model.load_state_dict(state['ema'], strict=True)
        model.requires_grad_(False)
        del state
        before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        predictions = {key: {'ap001': [], 'runtime025': []} for key in PATHS}
        details, losses, timing, alignments = [], defaultdict(list), [], []
        production_predictions = {key: {'ap001': [], 'runtime025': []} for key in ('D_original_u8_rect', 'B_bf16_prequant_rect')}
        production_shapes = []
        model_dir = run_dir / name; model_dir.mkdir()
        for index, file_name in enumerate(files):
            start = time.perf_counter()
            record = record_by_name[file_name]; image = image_by_name[file_name]
            bgr = cv2.imdecode(np.frombuffer(record['path'].read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(file_name)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / np.float32(255)
            geom = geometry(*rgb.shape[:2]); small = resize_float(rgb, geom)
            batch = make_batch(record, geom)
            # Verify geometry against existing uint8 letterbox GT contract, not image equality.
            _, ref_boxes, _ = letterbox_rgb((rgb * 255).round().astype(np.uint8), record['boxes'], record['ignore_boxes'], SIZE, False)
            alignments.append(float((ref_boxes - batch['bboxes']).abs().max()) if len(ref_boxes) else 0.)
            assert alignments[-1] <= 1e-6
            with torch.no_grad(), torch.autocast(device_type=device, enabled=False):
                direct = common_pad(small, geom)
                a = common_pad(array_rgb(model(tensor_rgb(small, device))), geom)
                b = letterbox_float(array_rgb(restore_tiled(model, tensor_rgb(rgb, device), tile=1024, overlap=64)), geom)[0]
                a_pad = array_rgb(model(tensor_rgb(direct, device)))
            with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == 'cuda'):
                bf16_full = array_rgb(restore_tiled(model, tensor_rgb(rgb, device), tile=1024, overlap=64))
            bf16_float = letterbox_float(bf16_full, geom)[0]
            formal_u8 = (bf16_full * 255).round().clip(0, 255).astype(np.uint8)
            formal = letterbox_rgb(formal_u8, record['boxes'], record['ignore_boxes'], SIZE, False)[0].permute(1,2,0).numpy()
            canvases = dict(D_fp32=direct, A_fp32=a, B_fp32=b,
                D_u8=quantize_canvas(direct), A_u8=quantize_canvas(a), B_u8=quantize_canvas(b),
                A_pad_historical_fp32=a_pad, B_bf16_float=bf16_float, B_restoration_bf16_prequant_square=formal)
            per_path = {}
            for key, canvas in canvases.items():
                assert canvas.dtype == np.float32 and canvas.shape == (SIZE, SIZE, 3) and np.isfinite(canvas).all()
                with torch.no_grad(), torch.autocast(device_type=device, enabled=False):
                    out = batch_forward(loss_fn, tensor_rgb(canvas, device), batch, keep_graph=False)
                    loss = float(out['native_scalar']); decomposed = float(decomposed_total(out['per_class']))
                    assert abs(loss - decomposed) < 1e-4, (name, key, file_name, loss, decomposed)
                    low = low_threshold_details(out['assigned'], anns_by_image[image['id']], geom)
                    for protocol, threshold in [('ap001', .001), ('runtime025', .25)]:
                        row, kept = nms_rows(out['assigned'], loss_fn.criterion, geom, image, threshold)
                        predictions[key][protocol].append(row)
                    per_path[key] = dict(native_loss=loss, native_parts=out['native_stats'],
                        per_class_native_components={CLASSES[c]: {term: float(value) for term, value in out['per_class'][c].items()} for c in CLASSES},
                        decomposition_abs_diff=abs(loss-decomposed), per_gt=low)
                    losses[key].append(loss)
                    del out
            rect_shapes = {}
            for key, source_bgr in [('D_original_u8_rect', bgr), ('B_bf16_prequant_rect', cv2.cvtColor(formal_u8, cv2.COLOR_RGB2BGR))]:
                rows_rect, observed_shapes = actual_rect_predictions(production_detector, source_bgr, image, device)
                for protocol in ('ap001', 'runtime025'):
                    production_predictions[key][protocol].append(rows_rect[protocol])
                rect_shapes[key] = observed_shapes
            production_shapes.append(dict(image_id=image['id'], shapes=rect_shapes))
            details.append(dict(image_id=image['id'], file_name=file_name, geometry=geom, paths=per_path))
            elapsed = time.perf_counter() - start
            timing.append(dict(image_id=image['id'], seconds_all_diagnostic_paths=elapsed))
            print(f'{name} scale {index+1}/{len(files)} {elapsed:.2f}s', flush=True)
        for key in PATHS:
            for protocol in ('ap001', 'runtime025'):
                predictions[key][protocol].sort(key=lambda x: x['image_id'])
                write_json(model_dir / key / (protocol + '.json'), {'protocol': report['protocol'], 'per_image': predictions[key][protocol]})
        summaries = evaluate_paths(images, annotations, predictions)
        for key, rows_rect in production_predictions.items():
            for protocol in ('ap001', 'runtime025'):
                rows_rect[protocol].sort(key=lambda x: x['image_id'])
                write_json(model_dir / 'actual_production_rect' / key / (protocol+'.json'), {'protocol': dict(imgsz=640,rect=True,half=False,ap_conf=.001,runtime_conf=.25,nms_iou=.7,max_det=300,ap_max_dets=100,runtime_max_dets=300), 'per_image': rows_rect[protocol]})
        production_summaries = evaluate_paths(images, annotations, production_predictions, direct_key='D_original_u8_rect', comparisons=[('B_bf16_prequant_rect','D_original_u8_rect')])
        base_gt = [gt for row in details for gt in row['paths']['D_fp32']['per_gt']]
        for key in PATHS:
            summaries[key]['mean_native_loss_components_by_class'] = {CLASSES[c]: {term: float(np.mean([row['paths'][key]['per_class_native_components'][CLASSES[c]][term] for row in details])) for term in ('cls','box','dfl','total')} for c in CLASSES}
            summaries[key]['gt_size_bins_all_classes'] = bin_summary(base_gt, summaries, key)
            for code in (5, 6, 1):
                subset = [g for g in base_gt if g['category'] == code]
                summaries[key].setdefault('gt_size_bins_by_class', {})[CLASSES[code]] = bin_summary(subset, summaries, key)
        for image_row in details:
            for key, row in image_row['paths'].items():
                for gt in row['per_gt']:
                    gt['post_nms_ap001'] = summaries[key]['ap001_gt'][str(gt['gt_id'])]
                    gt['post_nms_runtime025'] = summaries[key]['runtime025_gt'][str(gt['gt_id'])]
        production_unchanged = all(torch.equal(production_state[k],v.detach().cpu()) for k,v in production_detector.model.state_dict().items())
        assert production_unchanged
        unchanged_model = all(torch.equal(before[k], v.detach().cpu()) for k,v in model.state_dict().items())
        unchanged_detector = all(torch.equal(detector_state[k],v.detach().cpu()) for k,v in loss_fn.detector.state_dict().items())
        assert unchanged_model and unchanged_detector
        result = dict(checkpoint=path, checkpoint_sha256=sha256(checkpoint), weights_and_bn_unchanged=unchanged_model,
                      detector_weights_and_bn_unchanged=unchanged_detector, gt_geometry_max_abs_diff_normalized=max(alignments),
                      losses_mean={key:float(np.mean(values)) for key,values in losses.items()},
                      summaries=summaries, actual_production_rect=dict(config='YOLO.predict rect=True, img640, FP32 detector, uint8 original-size BGR, sameNMS/AP/runtime rules; separate from square controlled diagnostic; detectorTF32off for D/B equally', summaries=production_summaries, observed_input_shapes=production_shapes, detector_fused_setup=True, detector_weights_and_bn_unchanged_after_setup=production_unchanged), per_image=details, timings=timing)
        report['models'][name] = result
        write_json(model_dir/'result.json', result)
        del model, before
        if device == 'cuda': torch.cuda.empty_cache()
    report['all_completed'] = len(report['models']) == len(args.models)
    write_json(output, report)
    print('Completed scale diagnostic:', output, flush=True)


if __name__ == '__main__':
    main()
