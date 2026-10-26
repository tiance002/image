"""NP2 诊断一：原生 TaskAlignedAssigner 在 bus/train 上的正样本监督覆盖（只读）。

原则：
- 每批仍是「该批全部 GT 一起」送进原生 TaskAlignedAssigner，不删类、不重新分配。
- 用与 ultralytics v8DetectionLoss 完全相同的中间实现（criterion.assigner / bce / bbox_loss）
  复算一次，只为把同一个 loss 按 anchor 所属 GT 类别分组；并用 loss_fn.forward 的原生标量做校验。
- 参数不更新：统计使用 no_grad；梯度诊断只读取 .grad 后立即清零。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from saferestore.data.detection_gt import ACDCDetectionDataset, collate_detection_batch
from saferestore.losses.task_detection import FrozenYoloTaskLoss, ignore_anchor_mask
from saferestore.models.nafnet import build_nafnet

CLASS_NAMES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 6: 'train', 7: 'truck'}
WEATHER_ORDER = ('fog', 'rain', 'night', 'snow')


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def interleave_by_sequence(items):
    """把同一批图按 sequence 轮转重排，使连续取样尽量落在不同序列上。"""
    groups = defaultdict(list)
    for item in items:
        groups[item['file_name'].split('/')[2]].append(item)
    keys = sorted(groups)
    for key in keys:
        groups[key].sort(key=lambda x: x['file_name'])
    out, cursor = [], 0
    while any(groups[key] for key in keys):
        key = keys[cursor % len(keys)]
        if groups[key]:
            out.append(groups[key].pop(0))
        cursor += 1
    return out


def round_robin(pool, count):
    """按天气分组、组内再按序列轮转，然后天气间 round-robin 取图。

    这样固定规则下既能均衡天气，也能尽量覆盖不同序列。
    """
    groups = {w: interleave_by_sequence([i for i in pool if i['file_name'].split('/')[0] == w])
              for w in WEATHER_ORDER}
    picked, cursor = [], {w: 0 for w in WEATHER_ORDER}
    while len(picked) < count and any(cursor[w] < len(groups[w]) for w in WEATHER_ORDER):
        for w in WEATHER_ORDER:
            if len(picked) >= count:
                break
            if cursor[w] < len(groups[w]):
                picked.append(groups[w][cursor[w]])
                cursor[w] += 1
    return picked


def build_sample(annotation_path: Path):
    """确定性抽样：24 含 bus、24 含 train（去重）、16 其他/背景；不足不用 val 补足。"""
    payload = json.loads(annotation_path.read_text(encoding='utf-8'))
    by_img = defaultdict(list)
    for annotation in payload['annotations']:
        by_img[annotation['image_id']].append(annotation)
    images = sorted(payload['images'], key=lambda x: x['file_name'])
    has_bus = [i for i in images if any(a['category_id'] == 28 for a in by_img[i['id']])]
    has_train = [i for i in images if any(a['category_id'] == 31 for a in by_img[i['id']])]
    bus_set = round_robin(has_bus, 24)
    used = {i['id'] for i in bus_set}
    train_set = round_robin([i for i in has_train if i['id'] not in used], 24)
    used |= {i['id'] for i in train_set}
    other_pool = [i for i in images if i['id'] not in used and
                  not any(a['category_id'] in (28, 31) for a in by_img[i['id']])]
    other_pool = sorted(other_pool, key=lambda x: hashlib.md5(x['file_name'].encode()).hexdigest())
    step = max(1, len(other_pool) // 16)
    other_set = other_pool[::step][:16]
    rows = []
    for label, items in (('bus', bus_set), ('train', train_set), ('other', other_set)):
        for i in items:
            counts = Counter(a['category_id'] for a in by_img[i['id']])
            rows.append(dict(group=label, image_id=i['id'], file_name=i['file_name'],
                             weather=i['file_name'].split('/')[0], width=i['width'], height=i['height'],
                             categories={str(k): v for k, v in sorted(counts.items())}))
    available = dict(bus_images=len(has_bus), train_images=len(has_train), overlap=len({i['id'] for i in has_bus} & {i['id'] for i in has_train}))
    return rows, available


def probe_batch(loss_fn, model_output, batch, ignore_mask):
    """与原生 v8DetectionLoss 同源的分解计算。返回逐类分量与分配结果。"""
    from ultralytics.utils.tal import make_anchors

    criterion = loss_fn.criterion
    raw = criterion.parse_output(model_output)
    pred_distri = raw['boxes'].permute(0, 2, 1).contiguous()
    pred_scores = raw['scores'].permute(0, 2, 1).contiguous()
    anchor_points, stride_tensor = make_anchors(raw['feats'], criterion.stride, 0.5)
    dtype = pred_scores.dtype
    bs = pred_scores.shape[0]
    imgsz = torch.tensor(raw['feats'][0].shape[2:], device=criterion.device, dtype=dtype) * criterion.stride[0]

    targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
    targets = criterion.preprocess(targets.to(criterion.device), bs, imgsz[[1, 0, 1, 0]])
    gt_labels, gt_bboxes = targets.split((1, 4), 2)
    mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
    pred_bboxes = criterion.bbox_decode(anchor_points, pred_distri)

    target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = criterion.assigner(
        pred_scores.detach().sigmoid(),
        (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
        anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)
    target_scores_sum = max(target_scores.sum(), 1)

    criterion.bce.mask = ignore_mask
    bce = criterion.bce(pred_scores, target_scores.to(dtype))
    if criterion.class_weights is not None:
        bce = bce * criterion.class_weights
    loss_cls = bce.sum() / target_scores_sum * criterion.hyp.cls
    if fg_mask.any():
        loss_box, loss_dfl = criterion.bbox_loss(pred_distri, pred_bboxes, anchor_points,
                                                 target_bboxes / stride_tensor, target_scores,
                                                 target_scores_sum, fg_mask, imgsz, stride_tensor)
    else:
        loss_box = loss_dfl = torch.zeros((), device=criterion.device)
    loss_box = loss_box * criterion.hyp.box
    loss_dfl = loss_dfl * criterion.hyp.dfl
    criterion.bce.mask = None

    return dict(target_labels=target_labels, target_scores=target_scores, fg_mask=fg_mask,
                target_gt_idx=target_gt_idx, bce=bce, target_scores_sum=target_scores_sum,
                loss_cls=loss_cls, loss_box=loss_box, loss_dfl=loss_dfl,
                total=loss_cls + loss_box + loss_dfl,
                pred_distri=pred_distri, pred_bboxes=pred_bboxes, anchor_points=anchor_points,
                target_bboxes=target_bboxes, stride_tensor=stride_tensor, imgsz=imgsz)


def gradient_probe(model, loss_fn, dataset, order, device, batch_size=1, max_batches=0):
    """按同一批原生分配，分别对 bus / train / 全体 的 loss 分量反传，读复原网梯度 L2 范数。

    每个分量单独前向一次（不 retain_graph），把显存峰值压到一次前向的量级；
    检测器是 eval 模式、无 dropout，同一输入必然得到同一分配，因此三次分量共享的分配是一致的。

    说明：反传需要保存整条前向激活，显存占用约为纯推理的数倍。这里单独用 --grad-batch
    （默认 1）构造 loader，与统计用的 --batch 解耦；TaskAlignedAssigner 是按图独立的
    （preprocess 只把 GT 补齐到批内最大值，mask_gt 屏蔽填充），因此批大小不影响单图分配结果。
    """
    loader = DataLoader(torch.utils.data.Subset(dataset, order), batch_size=batch_size, shuffle=False,
                        collate_fn=collate_detection_batch, num_workers=0)
    from ultralytics.utils.metrics import bbox_iou
    from ultralytics.utils.tal import bbox2dist, make_anchors

    for parameter in model.parameters():
        parameter.requires_grad_(True)
    result = defaultdict(float)
    criterion = loss_fn.criterion

    def component_for(tag, images, batch):
        restored = model(images)
        output = loss_fn.detector(restored.float())
        raw = criterion.parse_output(output)
        pred_distri = raw['boxes'].permute(0, 2, 1).contiguous()
        pred_scores = raw['scores'].permute(0, 2, 1).contiguous()
        anchor_points, stride_tensor = make_anchors(raw['feats'], criterion.stride, 0.5)
        imgsz = torch.tensor(raw['feats'][0].shape[2:], device=criterion.device, dtype=pred_scores.dtype) * criterion.stride[0]
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = criterion.preprocess(targets.to(criterion.device), pred_scores.shape[0], imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = criterion.bbox_decode(anchor_points, pred_distri)
        target_labels, target_bboxes, target_scores, fg_mask, _ = criterion.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)
        tss = target_scores.sum().clamp_min(1)
        # 与训练流一致地应用 crowd/rider 忽略掩码，避免负监督口径不同
        ignored = batch.get('ignore_boxes', images.new_empty((0, 4))).to(criterion.device)
        ignored = ignored * torch.tensor([images.shape[3], images.shape[2], images.shape[3], images.shape[2]],
                                         device=criterion.device, dtype=ignored.dtype)
        ignore_idx = batch.get('ignore_batch_idx', torch.empty(0, dtype=torch.long)).to(criterion.device)
        criterion.bce.mask = ignore_anchor_mask(anchor_points * stride_tensor,
                                                pred_bboxes.detach() * stride_tensor,
                                                ignored, ignore_idx, len(images))
        bce = criterion.bce(pred_scores, target_scores)
        if criterion.class_weights is not None:
            bce = bce * criterion.class_weights
        criterion.bce.mask = None
        weight = target_scores.sum(-1)[fg_mask]
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        box_per = (1.0 - iou) * weight / tss * criterion.hyp.box
        target_ltrb = bbox2dist(anchor_points, target_bboxes, criterion.reg_max - 1)
        dfl_per = criterion.bbox_loss.dfl_loss(
            pred_distri[fg_mask].view(-1, criterion.reg_max), target_ltrb[fg_mask]) * weight / tss * criterion.hyp.dfl
        labels_fg = target_labels[fg_mask].long()
        if tag == 'all':
            return bce.sum() / tss * criterion.hyp.cls + box_per.sum() + dfl_per.sum()
        code = 5 if tag == 'bus' else 6
        selected = labels_fg == code
        return bce[..., code].sum() / tss * criterion.hyp.cls + box_per[selected].sum() + dfl_per[selected].sum()

    for index, batch in enumerate(loader):
        if max_batches and index >= max_batches:
            break
        images = batch['img'].to(device)
        for tag in ('all', 'bus', 'train'):
            model.zero_grad(set_to_none=True)
            value = component_for(tag, images, batch)
            value.backward()
            total = 0.0
            for parameter in model.parameters():
                if parameter.grad is not None:
                    total += float(parameter.grad.detach().pow(2).sum())
            result[tag] += total ** 0.5
            model.zero_grad(set_to_none=True)
            del value
            if device == 'cuda':
                torch.cuda.empty_cache()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return dict(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', default=[
        'v4=runs/real_v4_allweather/latest.pt',
        'C0=runs/gt_next_phase_20261025_C0/latest.pt',
        'C1=runs/gt_next_phase_20261025_C1/latest.pt'])
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--out', default='results/np2_external_probe/supervision_probe.json')
    parser.add_argument('--max-batches', type=int, default=0, help='仅用于冒烟验证，0 表示全部')
    parser.add_argument('--grad', action='store_true', help='额外统计 bus/train/全体 对复原网络的梯度范数')
    parser.add_argument('--grad-batch', type=int, default=1, help='反传用的批大小（显存约为推理的数倍）')
    args = parser.parse_args()

    annotation_path = ROOT / 'data/raw/ACDC_detection/gt_detection/instancesonly_train_gt_detection.json'
    rows, available = build_sample(annotation_path)
    manifest = dict(annotation=str(annotation_path.relative_to(ROOT)),
                    annotation_sha256=sha256_file(annotation_path),
                    rule='bus 24 + train 24（去掉与 bus 重叠）+ 其他/背景 16；按天气 round-robin，组内路径字典序；对照按路径 md5 等距抽样；不足不用 val 补足',
                    available=available,
                    counts=dict(total=len(rows), by_group=dict(Counter(r['group'] for r in rows)),
                                by_weather=dict(Counter(r['weather'] for r in rows))),
                    images=rows)
    print('抽样：', manifest['counts'], '可用池：', available, flush=True)

    dataset = ACDCDetectionDataset(str(annotation_path), ROOT / 'data/raw/ACDC_images/rgb_anon', image_size=640, seed=42)
    index_by_name = {r['file_name']: i for i, r in enumerate(dataset.records)}
    order = [index_by_name[r['file_name']] for r in rows]
    loader = DataLoader(torch.utils.data.Subset(dataset, order), batch_size=args.batch, shuffle=False,
                        collate_fn=collate_detection_batch, num_workers=0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_fn = FrozenYoloTaskLoss(ROOT / 'yolov8n.pt', device=device)
    report = dict(manifest=manifest, models={})

    for spec in args.models:
        name, path = spec.split('=', 1)
        checkpoint = ROOT / path
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        checkpoint_iter = int(state.get('iter', -1))
        model = build_nafnet().to(device).eval()
        model.load_state_dict(state['ema'])
        del state
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        agg_gt, class_weight_total = [], defaultdict(float)
        cls_pos = defaultdict(float); cls_neg = defaultdict(float)
        box_sum = defaultdict(float); dfl_sum = defaultdict(float)
        native_total, reproduced_total = [], []
        ignored_fraction, gt_total = [], 0

        for batch_index, batch in enumerate(loader):
            if args.max_batches and batch_index >= args.max_batches:
                break
            images = batch['img'].to(device)
            with torch.no_grad():
                restored = model(images)
                native_scalar, native_stats = loss_fn(restored, batch)
                output = loss_fn.detector(restored.float())
                probe = probe_batch(loss_fn, output, batch, loss_fn.criterion.bce.mask)
            native_total.append(float(native_scalar))
            reproduced_total.append(float(probe['total']))
            ignored_fraction.append(float(native_stats.get('ignored_anchor_fraction', 0.0)))
            gt_total += int(native_stats.get('gt_count', 0))

            fg = probe['fg_mask']; labels = probe['target_labels'].long()
            tss = float(probe['target_scores_sum'])
            cls = batch['cls'].flatten().long()
            batch_idx = batch['batch_idx'].long()
            for b in range(len(batch['file_names'])):
                classes_b = cls[batch_idx == b].tolist()
                for local, code in enumerate(classes_b):
                    # target_gt_idx 是「图内」GT 序号（preprocess 把每图 GT 补齐到批内最大值后，
                    # 再按图取前 n_gt 个有效行），所以这里必须用 local，不能用批内全局序号。
                    picked = fg[b] & (probe['target_gt_idx'][b] == local)
                    n_pos = int(picked.sum())
                    weight = float(probe['target_scores'][b][picked].sum()) if n_pos else 0.0
                    agg_gt.append(dict(image=batch['file_names'][b], category=int(code),
                                       positives=n_pos, weight=weight))

            # 逐类 target_scores 权重（正样本侧）
            for b in range(fg.shape[0]):
                if fg[b].any():
                    for code, w in zip(labels[b][fg[b]].tolist(), probe['target_scores'][b][fg[b]].sum(-1).tolist()):
                        class_weight_total[code] += float(w)
            # 分类正负拆分 + 归因
            bce = probe['bce']
            pos = fg.unsqueeze(-1) & (probe['target_scores'] > 0)
            for code in range(bce.shape[-1]):
                total_c = float(bce[..., code].sum())
                if total_c == 0:
                    continue
                p = float(bce[..., code].masked_select(pos[..., code]).sum())
                cls_pos[code] += p / tss
                cls_neg[code] += (total_c - p) / tss
            # box / dfl 按 GT 类别归因（同一批分配、同一批 anchor）
            if fg.any():
                from ultralytics.utils.metrics import bbox_iou
                weight = probe['target_scores'].sum(-1)[fg]
                iou = bbox_iou(probe['pred_bboxes'][fg], probe['target_bboxes'][fg], xywh=False, CIoU=True)
                box_per = ((1.0 - iou) * weight) / tss
                # DFL：复用原生 BboxLoss 的逐 anchor 计算
                from ultralytics.utils.tal import bbox2dist
                target_ltrb = bbox2dist(probe['anchor_points'], probe['target_bboxes'], loss_fn.criterion.reg_max - 1)
                dfl_raw = loss_fn.criterion.bbox_loss.dfl_loss(
                    probe['pred_distri'][fg].view(-1, loss_fn.criterion.reg_max),
                    target_ltrb[fg]) * weight / tss
                labels_fg = labels[fg]
                for code in labels_fg.unique().tolist():
                    mask = labels_fg == code
                    box_sum[code] += float(box_per[mask].sum())
                    dfl_sum[code] += float(dfl_raw[mask].sum())

        if args.grad:
            grad = gradient_probe(model, loss_fn, dataset, order, device, args.grad_batch, args.max_batches)
            print(name, '梯度范数', {k: round(v, 4) for k, v in grad.items()}, flush=True)
        else:
            grad = None

        del model
        if device == 'cuda':
            torch.cuda.empty_cache()

        by_class = defaultdict(lambda: dict(gt=0, positives=0, zero=0, weight=0.0))
        for row in agg_gt:
            entry = by_class[row['category']]
            entry['gt'] += 1
            entry['positives'] += row['positives']
            entry['zero'] += int(row['positives'] == 0)
            entry['weight'] += row['weight']
        for code, entry in by_class.items():
            entry['name'] = CLASS_NAMES[code]
            entry['mean_positives'] = entry['positives'] / entry['gt'] if entry['gt'] else 0.0
            entry['zero_ratio'] = entry['zero'] / entry['gt'] if entry['gt'] else 0.0
            entry['cls_pos_contribution'] = cls_pos[code]
            entry['cls_neg_contribution'] = cls_neg[code]
            entry['box_contribution'] = box_sum[code]
            entry['dfl_contribution'] = dfl_sum[code]
            entry['target_weight_total'] = class_weight_total[code]
        report['models'][name] = dict(
            checkpoint=path, checkpoint_sha256=sha256_file(checkpoint), iter=checkpoint_iter,
            gt_count_total=gt_total, ignored_anchor_fraction_mean=float(np.mean(ignored_fraction)),
            native_loss_total_sum=float(np.sum(native_total)),
            reproduced_loss_total_sum=float(np.sum(reproduced_total)),
            reproduction_abs_diff=float(abs(np.sum(native_total) - np.sum(reproduced_total))),
            per_class={str(k): v for k, v in sorted(by_class.items())}, per_gt=agg_gt,
            grad_norm=grad,
            notes='正 anchor 来自单次原生联合分配；cls_pos/cls_neg/box/dfl 为同一 loss 的按类别分解，未重新分配')
        print(name, {CLASS_NAMES[k]: (v['gt'], v['positives'], round(v['mean_positives'], 2), round(v['zero_ratio'], 3))
                     for k, v in sorted(by_class.items())}, 'loss误差', report['models'][name]['reproduction_abs_diff'], flush=True)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('已写入', out)


if __name__ == '__main__':
    main()
