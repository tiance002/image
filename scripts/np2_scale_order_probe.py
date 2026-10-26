"""NP2 诊断二：训练与推理的尺度顺序（A 先缩放后复原 vs B 先复原后缩放）。

路径 A（训练流）：全分辨率退化图 → letterbox 到 640（含 114 灰边）→ 复原网络 → 冻结检测器。
路径 B（评估流）：全分辨率退化图 → 全分辨率复原（1024/64 分块，与正式评估一致）→ letterbox 640 → 冻结检测器。

两条路径使用同一套 letterbox 参数、同一 GT、同一检测器权重与 FP32 精度、同一 NMS 规则；
唯一被隔离的变量是「缩放发生在复原之前还是之后」。参数全程不更新。
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from saferestore.data.detection_gt import ACDCDetectionDataset, letterbox_rgb
from saferestore.losses.task_detection import FrozenYoloTaskLoss, ignore_anchor_mask
from saferestore.models.nafnet import build_nafnet
from scripts.train_restorer import restore_tiled
from scripts.np2_supervision_probe import build_sample, CLASS_NAMES

SIZE = 640


def iou_xyxy(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def decode_predictions(raw, stride_list, reg_max=16, conf=0.001):
    """把检测头原始输出解码为原 640 输入坐标的 xyxy 框。"""
    from ultralytics.utils.tal import make_anchors
    boxes = raw['boxes'].permute(0, 2, 1).contiguous()
    scores = raw['scores'].permute(0, 2, 1).contiguous()
    anchor_points, stride_tensor = make_anchors(raw['feats'], stride_list, 0.5)
    from ultralytics.utils.tal import dist2bbox
    proj = torch.arange(reg_max, dtype=boxes.dtype, device=boxes.device)
    b, a, c = boxes.shape
    dist = boxes.view(b, a, 4, c // 4).softmax(3).matmul(proj)
    pred = dist2bbox(dist, anchor_points, xywh=False) * stride_tensor
    prob = scores.sigmoid()
    cls_score, cls_idx = prob.max(-1)
    keep = (cls_score >= conf)[0]
    out = []
    for box, score, label in zip(pred[0][keep], cls_score[0][keep], cls_idx[0][keep]):
        out.append(dict(box=box.tolist(), confidence=float(score), category=int(label)))
    return out


def match_gt(preds, gts, iou_threshold=0.5):
    """全局贪心一对一匹配（IoU 降序），保证同一预测不被两个 GT 重复使用。"""
    pairs = []
    for gi, gt in enumerate(gts):
        for pi, pred in enumerate(preds):
            if pred['category'] != gt['category']:
                continue
            value = iou_xyxy(pred['box'], gt['box'])
            if value >= iou_threshold:
                pairs.append((value, gi, pi))
    pairs.sort(reverse=True)
    gt_used, pred_used = {}, set()
    for value, gi, pi in pairs:
        if gi in gt_used or pi in pred_used:
            continue
        gt_used[gi] = (value, pi)
        pred_used.add(pi)
    results = []
    for gi, gt in enumerate(gts):
        if gi in gt_used:
            value, pi = gt_used[gi]
            results.append(dict(category=gt['category'], gt_box=gt['box'], best_iou=value,
                                score=preds[pi]['confidence'], pred_box=preds[pi]['box'], matched=True))
            continue
        # 未命中时仍记录同类别最高 IoU 的候选，便于看"差一点"还是"完全没框"
        best, best_iou = None, 0.0
        for pred in preds:
            if pred['category'] != gt['category']:
                continue
            value = iou_xyxy(pred['box'], gt['box'])
            if value > best_iou:
                best, best_iou = pred, value
        results.append(dict(category=gt['category'], gt_box=gt['box'], best_iou=best_iou,
                            score=best['confidence'] if best else None,
                            pred_box=best['box'] if best else None, matched=False))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', default=[
        'v4=runs/real_v4_allweather/latest.pt',
        'C0=runs/gt_next_phase_20261025_C0/latest.pt',
        'C1=runs/gt_next_phase_20261025_C1/latest.pt'])
    parser.add_argument('--n-bus', type=int, default=12)
    parser.add_argument('--n-train', type=int, default=12)
    parser.add_argument('--out', default='results/np2_external_probe/scale_order_probe.json')
    args = parser.parse_args()

    rows, _ = build_sample(ROOT / 'data/raw/ACDC_detection/gt_detection/instancesonly_train_gt_detection.json')
    bus = [r for r in rows if r['group'] == 'bus'][:args.n_bus]
    train = [r for r in rows if r['group'] == 'train'][:args.n_train]
    chosen = bus + train
    files = [r['file_name'] for r in chosen]
    print(f'尺度顺序诊断样本：bus {len(bus)} + train {len(train)} = {len(files)} 张', flush=True)

    dataset = ACDCDetectionDataset(str(ROOT / 'data/raw/ACDC_detection/gt_detection/instancesonly_train_gt_detection.json'),
                                   ROOT / 'data/raw/ACDC_images/rgb_anon', image_size=SIZE, seed=42)
    index_by_name = {r['file_name']: i for i, r in enumerate(dataset.records)}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_fn = FrozenYoloTaskLoss(ROOT / 'yolov8n.pt', device=device)

    report = dict(sample=dict(files=files, n_bus=len(bus), n_train=len(train)), models={})
    for spec in args.models:
        name, path = spec.split('=', 1)
        state = torch.load(ROOT / path, map_location='cpu', weights_only=False)
        model = build_nafnet().to(device).eval()
        model.load_state_dict(state['ema'])
        del state
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        per_gt, losses = [], defaultdict(list)
        gt_align = []
        for file_name in files:
            item = dataset[index_by_name[file_name]]
            record = dataset.records[index_by_name[file_name]]
            original = cv2.cvtColor(cv2.imdecode(np.fromfile(str(record['path']), dtype=np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            # GT 在 640 letterbox 坐标下的 xyxy（归一化 * 640）
            boxes_norm = item['bboxes'].clone()
            gt_boxes = []
            for cls_code, box in zip(item['cls'].flatten().tolist(), boxes_norm.tolist()):
                cx, cy, w, h = box
                gt_boxes.append(dict(category=int(cls_code),
                                     box=[(cx - w / 2) * SIZE, (cy - h / 2) * SIZE,
                                          (cx + w / 2) * SIZE, (cy + h / 2) * SIZE]))
            batch_a = dict(batch_idx=torch.zeros(len(gt_boxes)),
                           cls=item['cls'], bboxes=item['bboxes'],
                           ignore_boxes=item['ignore_boxes'],
                           ignore_batch_idx=torch.zeros(len(item['ignore_boxes']), dtype=torch.long))

            with torch.no_grad():
                # 路径 A：先缩放（dataset 已 letterbox）→ 复原 → 检测
                degraded = item['img'][None].to(device)
                restored_a = model(degraded)
                raw_a = loss_fn.criterion.parse_output(loss_fn.detector(restored_a.float()))
                preds_a = decode_predictions(raw_a, loss_fn.criterion.stride)
                scalar_a, stats_a = loss_fn(restored_a, batch_a)

                # 路径 B：先全分辨率复原（正式评估路径）→ 再 letterbox → 检测
                full = torch.from_numpy(original.transpose(2, 0, 1).copy())[None].float().to(device) / 255
                # 两条路径统一 FP32，确保唯一差异是「缩放与复原的先后顺序」；
                # 正式评估使用的 bf16 影响单独记录，不混进本对照。
                restored_full = restore_tiled(model, full, tile=1024, overlap=64)
                rgb_b = (restored_full[0].float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
                canvas_b, boxes_b, ignored_b = letterbox_rgb(rgb_b, record['boxes'], record['ignore_boxes'], SIZE, False)
                restored_b = canvas_b[None].to(device)
                raw_b = loss_fn.criterion.parse_output(loss_fn.detector(restored_b.float()))
                preds_b = decode_predictions(raw_b, loss_fn.criterion.stride)
                batch_b = dict(batch_idx=torch.zeros(len(gt_boxes)),
                               cls=torch.tensor([g['category'] for g in gt_boxes], dtype=torch.float32).reshape(-1, 1),
                               bboxes=boxes_b, ignore_boxes=ignored_b,
                               ignore_batch_idx=torch.zeros(len(ignored_b), dtype=torch.long))
                scalar_b, stats_b = loss_fn(restored_b, batch_b)

            # 只统计 bus/train 的 GT（以及同图内其他类作参照）
            match_a = match_gt(preds_a, gt_boxes)
            match_b = match_gt(preds_b, gt_boxes)
            # 两条路径的 GT 理论上应完全一致（同图、同尺寸、同 letterbox 参数），这里做数值校验
            if len(gt_boxes):
                cx, cy = boxes_b[:, 0], boxes_b[:, 1]
                bw, bh = boxes_b[:, 2], boxes_b[:, 3]
                xyxy_b = torch.stack([(cx - bw / 2) * SIZE, (cy - bh / 2) * SIZE,
                                      (cx + bw / 2) * SIZE, (cy + bh / 2) * SIZE], 1)
                gt_a_tensor = torch.tensor([g['box'] for g in gt_boxes], dtype=torch.float32)
                gt_align.append(float((xyxy_b - gt_a_tensor).abs().max()))
            else:
                gt_align.append(0.0)
            for index, (gt, ma, mb) in enumerate(zip(gt_boxes, match_a, match_b)):
                if gt['category'] not in (5, 6):
                    continue
                per_gt.append(dict(image=file_name, gt_index=index, category=gt['category'], name=CLASS_NAMES[gt['category']],
                                   gt_box=[round(v, 1) for v in gt['box']],
                                   A=dict(best_iou=round(ma['best_iou'], 4), score=ma['score'], matched=ma['matched']),
                                   B=dict(best_iou=round(mb['best_iou'], 4), score=mb['score'], matched=mb['matched'])))
            losses['A'].append(dict(total=float(scalar_a), cls=stats_a.get('cls_loss'), box=stats_a.get('box_loss'), dfl=stats_a.get('dfl_loss')))
            losses['B'].append(dict(total=float(scalar_b), cls=stats_b.get('cls_loss'), box=stats_b.get('box_loss'), dfl=stats_b.get('dfl_loss')))
            print(f'{name} {file_name.split("/")[-1]} A命中{sum(m["matched"] for m in match_a)}/{len(match_a)} '
                  f'B命中{sum(m["matched"] for m in match_b)}/{len(match_b)}', flush=True)

        def summary(key):
            values = [r[key] for r in per_gt]
            return dict(n=len(values), matched=sum(v['matched'] for v in values),
                        mean_best_iou=float(np.mean([v['best_iou'] for v in values])) if values else None,
                        mean_score=float(np.mean([v['score'] for v in values if v['score'] is not None])) if any(v['score'] is not None for v in values) else None,
                        zero_iou=sum(v['best_iou'] < 1e-6 for v in values))

        switched = [r for r in per_gt if r['A']['matched'] != r['B']['matched']]
        report['models'][name] = dict(checkpoint=path, n_gt=len(per_gt),
                                      gt_box_max_abs_diff_px=max(gt_align) if gt_align else None,
                                      A=summary('A'), B=summary('B'),
                                      loss_A={k: float(np.mean([x[k] for x in losses['A'] if x[k] is not None])) for k in ('cls', 'box', 'dfl')},
                                      loss_B={k: float(np.mean([x[k] for x in losses['B'] if x[k] is not None])) for k in ('cls', 'box', 'dfl')},
                                      switched_cases=switched, per_gt=per_gt,
                                      note='路径A=训练流(先letterbox640再复原)；路径B=评估流(全分辨率复原再letterbox640)；同GT/同检测器/同FP32/NMS规则')
        print(name, 'A', summary('A'), 'B', summary('B'), flush=True)
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('已写入', out)


if __name__ == '__main__':
    main()
