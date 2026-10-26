"""NP2-R 诊断一（修订版）：原生 TaskAlignedAssigner 在 bus/train 上的正样本监督覆盖。

本文件是 `np2_supervision_probe.py` 的修订版，修正了三处已定位缺陷：

A. 坐标尺度不一致：`pred_bboxes`/`anchor_points` 是**网格坐标**（bbox_decode 直接产出，
   未乘 stride），而 assigner 返回的 `target_bboxes` 是**像素坐标**。原生 BboxLoss 在调用前
   会做 `target_bboxes / stride_tensor`。旧版直接 `bbox_iou(pred_bboxes, target_bboxes)`，
   等于拿「按 stride 缩放过的框」和「像素框」算 IoU —— 不同 stride 层被缩放的比例不同，
   IoU 与后续 bbox2dist 全部失真。

B. 权重形状广播：原生是 `weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)` → [N,1]，
   与 `bbox_iou` 返回的 [N,1] 对齐。旧版用 [N]，`[N,1] * [N]` 广播成 [N,N]，
   相当于把每个 anchor 的权重乘上了全部 anchor 的 IoU 列，box/DFL 与梯度都被放大。
   `DFLoss.__call__` 返回的也是 [N,1]（keepdim），同样受影响。

C. 校验范围不足：旧版只比较「全部批次加总」的原生损失，既不校验逐批一致性，
   也没有校验手动分解与梯度。本版对**每一批**都校验：
   - 七类分量之和 == 原生总损失（同 gain、同 target_scores_sum、同 batch 口径）；
   - 七类梯度向量之和 == 原生总损失的梯度向量（比对向量本身，不只比范数）。

其余原则不变：
- 每批仍是「该批全部 GT 一起」送进原生 TaskAlignedAssigner，不删类、不重新分配；
- 参数不更新：统计使用 no_grad，梯度只读取 .grad 后立即清零；
- 固定 64 张样本与三个检查点，不重新挑图。
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

from saferestore.data.detection_gt import ACDCDetectionDataset, collate_detection_batch, ACDC_TO_COCO
from saferestore.losses.task_detection import FrozenYoloTaskLoss
from saferestore.models.nafnet import build_nafnet

CLASS_NAMES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 6: 'train', 7: 'truck'}
SUPERVISED = (0, 1, 2, 3, 5, 6, 7)
WEATHER_ORDER = ('fog', 'rain', 'night', 'snow')


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()



def state_tensor_sha256(module):
    """Hash every parameter and buffer (including BN state), independent of .grad."""
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().contiguous().cpu()
        digest.update(name.encode('utf-8'))
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode('ascii'))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def build_gt_metadata(annotation_path, dataset):
    """Map raw annotation IDs in the exact unmodified Dataset GT order."""
    payload = json.loads(annotation_path.read_text(encoding='utf-8'))
    grouped = defaultdict(list)
    for annotation in payload['annotations']:
        if not annotation.get('iscrowd', 0) and not annotation.get('ignore', 0) and int(annotation['category_id']) in ACDC_TO_COCO:
            grouped[annotation['image_id']].append(annotation)
    result = {}
    for record in dataset.records:
        values = []
        for local, annotation in enumerate(grouped[record['image_id']]):
            x, y, width, height = map(float, annotation['bbox'])
            box = [max(0.0, x), max(0.0, y), min(float(record['width']), x + width), min(float(record['height']), y + height)]
            assert ACDC_TO_COCO[int(annotation['category_id'])] == record['classes'][local]
            assert np.allclose(box, record['boxes'][local], atol=0, rtol=0)
            values.append(dict(source_annotation_id=int(annotation['id']), image_id=int(record['image_id']),
                               local_gt_index=local, original_box_xyxy=box,
                               original_width=box[2] - box[0], original_height=box[3] - box[1]))
        assert len(values) == len(record['classes'])
        result[record['file_name']] = values
    return result


def write_report(path, report, first=False):
    content = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if first:
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(content)
    else:
        path.write_text(content, encoding='utf-8', newline='\n')
    assert path.read_text(encoding='utf-8') == content


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
    """按天气分组、组内再按序列轮转，然后天气间 round-robin 取图。"""
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
    available = dict(bus_images=len(has_bus), train_images=len(has_train),
                     overlap=len({i['id'] for i in has_bus} & {i['id'] for i in has_train}))
    return rows, available


def assign_targets(criterion, raw, batch):
    """按原生 v8DetectionLoss 的调用方式取分配结果，返回中间张量（坐标语义见模块说明）。"""
    from ultralytics.utils.tal import make_anchors

    pred_distri = raw['boxes'].permute(0, 2, 1).contiguous()
    pred_scores = raw['scores'].permute(0, 2, 1).contiguous()
    anchor_points, stride_tensor = make_anchors(raw['feats'], criterion.stride, 0.5)
    imgsz = torch.tensor(raw['feats'][0].shape[2:], device=criterion.device,
                         dtype=pred_scores.dtype) * criterion.stride[0]
    targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
    targets = criterion.preprocess(targets.to(criterion.device), pred_scores.shape[0], imgsz[[1, 0, 1, 0]])
    gt_labels, gt_bboxes = targets.split((1, 4), 2)
    mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
    pred_bboxes = criterion.bbox_decode(anchor_points, pred_distri)
    target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = criterion.assigner(
        pred_scores.detach().sigmoid(),
        (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
        anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)
    return dict(pred_distri=pred_distri, pred_scores=pred_scores, anchor_points=anchor_points,
                stride_tensor=stride_tensor, imgsz=imgsz, pred_bboxes=pred_bboxes,
                target_labels=target_labels, target_bboxes=target_bboxes,
                target_scores=target_scores, fg_mask=fg_mask, target_gt_idx=target_gt_idx,
                gt_labels=gt_labels, gt_bboxes=gt_bboxes)


def class_components(criterion, assigned, bce):
    """把同一份原生分配上的 loss 拆成逐类可微分量。

    坐标系：`pred_bboxes`/`anchor_points` 是网格坐标，`target_bboxes` 是像素坐标；
    这里统一到网格坐标（除以 stride），与原生 `BboxLoss` 的 `target_bboxes / stride_tensor` 一致。
    形状：`weight` 必须是 [N,1]；`bbox_iou` 与 `DFLoss` 都返回 [N,1]。
    """
    from ultralytics.utils.metrics import bbox_iou
    from ultralytics.utils.tal import bbox2dist

    pred_distri = assigned['pred_distri']
    pred_bboxes = assigned['pred_bboxes']
    anchor_points = assigned['anchor_points']
    target_bboxes = assigned['target_bboxes']
    target_scores = assigned['target_scores']
    fg_mask = assigned['fg_mask']
    target_labels = assigned['target_labels']
    stride_tensor = assigned['stride_tensor']

    tss = target_scores.sum().clamp_min(1)
    cls_by_class = bce.sum(dim=(0, 1)) / tss * criterion.hyp.cls  # [nc]，逐类分类损失
    zero = pred_distri.new_zeros(())
    box_per = dfl_per = labels_fg = None
    if bool(fg_mask.any()):
        target_grid = target_bboxes / stride_tensor
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)  # [N,1]，必须是 [N,1]
        iou = bbox_iou(pred_bboxes[fg_mask], target_grid[fg_mask], xywh=False, CIoU=True)  # [N,1]
        box_per = ((1.0 - iou) * weight) / tss * criterion.hyp.box  # [N,1]
        target_ltrb = bbox2dist(anchor_points, target_grid, criterion.reg_max - 1)
        dfl_per = criterion.bbox_loss.dfl_loss(
            pred_distri[fg_mask].view(-1, criterion.reg_max), target_ltrb[fg_mask]
        ) * weight / tss * criterion.hyp.dfl  # DFLoss 返回 [N,1]
        labels_fg = target_labels[fg_mask].long()

    per_class = {}
    for code in range(target_scores.shape[-1]):
        if box_per is None:
            box_c, dfl_c = zero, zero
        else:
            selected = labels_fg == code
            box_c = box_per[selected].sum() if bool(selected.any()) else zero
            dfl_c = dfl_per[selected].sum() if bool(selected.any()) else zero
        per_class[code] = dict(cls=cls_by_class[code], box=box_c, dfl=dfl_c,
                               total=cls_by_class[code] + box_c + dfl_c)
    return per_class


def batch_forward(loss_fn, restored, batch, keep_graph):
    """一次模型前向，返回原生标量/逐项、逐类分解与分配结果。

    原生损失直接走 `loss_fn`（内部即 ultralytics 原生 `criterion`），
    分解用同一份 restored 再做一次检测器前向；检测器 eval、无 dropout，
    两次前向的输出与分配逐位相同（下面 per-batch 校验会验证这一点）。
    """
    criterion = loss_fn.criterion
    context = torch.enable_grad() if keep_graph else torch.no_grad()
    with context:
        native_scalar, native_stats = loss_fn(restored, batch)
        output = loss_fn.detector(restored.float())
        raw = criterion.parse_output(output)
        assigned = assign_targets(criterion, raw, batch)
        bce = criterion.bce(assigned['pred_scores'], assigned['target_scores'].to(assigned['pred_scores'].dtype))
        if criterion.class_weights is not None:
            bce = bce * criterion.class_weights
        per_class = class_components(criterion, assigned, bce)
    return dict(native_scalar=native_scalar, native_stats=native_stats, raw=raw,
                assigned=assigned, bce=bce, per_class=per_class)


def decomposed_total(per_class):
    """七类（SUPERVISED）分量之和；与原生口径一致：同 gain、同 tss、同 batch。"""
    return sum(per_class[code]['total'] for code in SUPERVISED)


def grad_vector(parameters):
    """按参数顺序拼成一维 float32 向量（仅用于比较向量本身，不参与训练）。"""
    pieces = []
    for parameter in parameters:
        if parameter.grad is None:
            pieces.append(torch.zeros(parameter.numel(), device=parameter.device))
        else:
            pieces.append(parameter.grad.detach().reshape(-1))
    return torch.cat(pieces).to(torch.float32)


def compare_vectors(a, b):
    """比较两个梯度向量：相对 L2 误差、余弦相似度、最大绝对差。"""
    diff = a - b
    norm_a = float(a.norm())
    norm_b = float(b.norm())
    denominator = max(norm_b, 1e-12)
    cosine = float(torch.dot(a, b) / (norm_a * norm_b)) if norm_a > 0 and norm_b > 0 else 1.0
    return dict(norm_sum=norm_a, norm_native=norm_b,
                relative_l2_error=float(diff.norm()) / denominator,
                cosine=cosine, max_abs_diff=float(diff.abs().max()))


def gradient_check(model, loss_fn, dataset, order, device, batch_size, max_batches, gt_metadata=None):
    """逐批验证「七类梯度向量之和 == 原生总损失梯度向量」。

    每个分量都从同一次前向的同一份分配上反传（retain_graph=True），
    因此不存在「不同次前向分配不同」的混淆。
    """
    loader = DataLoader(torch.utils.data.Subset(dataset, order), batch_size=batch_size, shuffle=False,
                        collate_fn=collate_detection_batch, num_workers=0)
    parameters = [p for p in model.parameters() if p.requires_grad]
    per_batch, by_class_norm = [], defaultdict(list)
    by_class_ratio = defaultdict(list)
    for index, batch in enumerate(loader):
        if max_batches and index >= max_batches:
            break
        images = batch['img'].to(device)
        model.zero_grad(set_to_none=True)
        out = batch_forward(loss_fn, model(images), batch, keep_graph=True)
        # Preserve historical forward/assignment, but make backward convolution FP32.
        # TF32 backward is not sufficiently additive for the vector identity check.
        with torch.backends.cudnn.flags(enabled=torch.backends.cudnn.enabled,
                benchmark=torch.backends.cudnn.benchmark, benchmark_limit=torch.backends.cudnn.benchmark_limit,
                deterministic=torch.backends.cudnn.deterministic, allow_tf32=False):
            native = out['native_scalar']
            native.backward(retain_graph=True)
            native_vector = grad_vector(parameters)
            sum_vector = torch.zeros_like(native_vector)
            norms = {}
            for code in SUPERVISED:
                model.zero_grad(set_to_none=True)
                term = out['per_class'][code]['total']
                if term.requires_grad:
                    term.backward(retain_graph=True)
                    current = grad_vector(parameters)
                else:
                    current = torch.zeros_like(native_vector)
                norms[code] = float(current.norm())
                sum_vector += current
        model.zero_grad(set_to_none=True)
        comparison = compare_vectors(sum_vector, native_vector)
        decomposed = float(decomposed_total(out['per_class']))
        difference = abs(float(native) - decomposed)
        ratios = {code: value / max(comparison['norm_native'], 1e-12) for code, value in norms.items()}
        per_batch.append(dict(batch=index, images=len(batch['file_names']),
                              file_names=list(batch['file_names']),
                              gt_ids=([item['source_annotation_id'] for name in batch['file_names'] for item in gt_metadata[name]] if gt_metadata else None),
                              native_loss=float(native), decomposed_loss=decomposed,
                              loss_abs_diff=difference,
                              grad=comparison, class_grad_norm=norms,
                              class_grad_norm_ratio_nonadditive=ratios))
        if not np.isfinite(difference) or difference >= 1e-4 or not np.isfinite(comparison['relative_l2_error']) or comparison['relative_l2_error'] >= 1e-4:
            raise RuntimeError(f'gradient batch {index} failed consistency: {per_batch[-1]}')
        for code, value in norms.items():
            by_class_norm[code].append(value)
            by_class_ratio[code].append(ratios[code])
        if (index + 1) % 8 == 0:
            print('gradient checked', index + 1, '/', len(loader), flush=True)
        del out, native, native_vector, sum_vector, current, term
        if device == 'cuda':
            torch.cuda.empty_cache()

    losses = [r['loss_abs_diff'] for r in per_batch]
    rel = [r['grad']['relative_l2_error'] for r in per_batch]
    return dict(batch_size=batch_size, batches=len(per_batch),
                loss_max_abs_diff=max(losses) if losses else None,
                grad_relative_l2_max=max(rel) if rel else None,
                grad_cosine_min=min(r['grad']['cosine'] for r in per_batch) if per_batch else None,
                all_pass=bool(losses and max(losses) < 1e-4 and max(rel) < 1e-4),
                native_mean_norm=float(np.mean([r['grad']['norm_native'] for r in per_batch])) if per_batch else None,
                native_sum_of_batch_norms=float(sum(r['grad']['norm_native'] for r in per_batch)),
                per_class_mean_norm={CLASS_NAMES[c]: float(np.mean(v)) for c, v in by_class_norm.items()},
                per_class_sum_of_batch_norms={CLASS_NAMES[c]: float(sum(v)) for c, v in by_class_norm.items()},
                per_class_mean_norm_ratio_nonadditive={CLASS_NAMES[c]: float(np.mean(v)) for c, v in by_class_ratio.items()},
                precision=dict(forward_cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32), backward_cudnn_allow_tf32=False, matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32), note='Forward uses historical precision; only backward convolution TF32 disabled. Tolerance unchanged at1e-4; diagnosis evidence supervision_tf32_diagnosis.json.'),
                interpretation='Gradient norm ratios are nonadditive; mean/sum of per-batch norms are not a joint-dataset gradient norm.',
                per_batch=per_batch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', default=[
        'v4=runs/real_v4_allweather/latest.pt',
        'C0=runs/gt_next_phase_20261025_C0/latest.pt',
        'C1=runs/gt_next_phase_20261025_C1/latest.pt'])
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--out', default='results/np2_revision/supervision_probe.json')
    parser.add_argument('--max-batches', type=int, default=0, help='仅用于冒烟验证，0 表示全部')
    parser.add_argument('--grad', action='store_true', help='额外做逐批梯度向量求和校验')
    parser.add_argument('--grad-batch', type=int, default=1)
    parser.add_argument('--grad-max-batches', type=int, default=0, help='0 表示与 --max-batches 一致')
    args = parser.parse_args()
    original_precision = dict(cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32), matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32), cudnn_benchmark=bool(torch.backends.cudnn.benchmark), cudnn_deterministic=bool(torch.backends.cudnn.deterministic))
    torch.backends.cuda.matmul.allow_tf32 = False

    report_path = (ROOT / args.out).resolve()
    if not report_path.is_relative_to((ROOT / 'results/np2_revision').resolve()):
        raise ValueError('NP2-R outputs must stay under results/np2_revision')
    if report_path.exists():
        raise FileExistsError(f'Preserve existing evidence: {report_path}')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_path = ROOT / 'data/raw/ACDC_detection/gt_detection/instancesonly_train_gt_detection.json'
    rows, available = build_sample(annotation_path)
    legacy_path = ROOT / 'results/np2_external_probe/supervision_probe.json'
    legacy = json.loads(legacy_path.read_text(encoding='utf-8'))
    if rows != legacy['manifest']['images']:
        raise RuntimeError('The fixed 64-image manifest differs from the original NP2 sample')
    manifest = dict(annotation=str(annotation_path.relative_to(ROOT)),
                    annotation_sha256=sha256_file(annotation_path),
                    rule='bus 24 + train 24（去掉与 bus 重叠）+ 其他/背景 16；按天气 round-robin，组内路径字典序；对照按路径 md5 等距抽样；不足不用 val 补足',
                    available=available,
                    revision='np2r-1：修正网格/像素坐标混用、weight=[N] 广播、缺少逐批与梯度校验',
                    counts=dict(total=len(rows), by_group=dict(Counter(r['group'] for r in rows)),
                                by_weather=dict(Counter(r['weather'] for r in rows))),
                    images=rows,
                    legacy_manifest_exact_match=True,
                    legacy_evidence_sha256=sha256_file(legacy_path),
                    loss_units='Per batch mean native scalar: gained cls/box/DFL, shared target_scores_sum. Dataset totals sum batches (not an image mean).',
                    legacy_cls_units='Legacy cls_pos/cls_neg were BCE/target_scores_sum without hyp.cls gain; saved explicitly alongside gained fields.',
                    command_options=vars(args), precision=dict(original_flags=original_precision, forward_cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32), backward_cudnn_allow_tf32=False, matmul_allow_tf32=False, backward_only_tf32_disabled=True, validation_tolerance=1e-4, numerical_diagnosis='results/np2_revision/supervision_tf32_diagnosis.json'))
    print('抽样：', manifest['counts'], '可用池：', available, flush=True)

    dataset = ACDCDetectionDataset(str(annotation_path), ROOT / 'data/raw/ACDC_images/rgb_anon',
                                   image_size=640, seed=42)
    gt_metadata = build_gt_metadata(annotation_path, dataset)
    index_by_name = {r['file_name']: i for i, r in enumerate(dataset.records)}
    order = [index_by_name[r['file_name']] for r in rows]
    loader = DataLoader(torch.utils.data.Subset(dataset, order), batch_size=args.batch, shuffle=False,
                        collate_fn=collate_detection_batch, num_workers=0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_fn = FrozenYoloTaskLoss(ROOT / 'yolov8n.pt', device=device)
    detector_before = state_tensor_sha256(loss_fn.detector)
    detector_file_before = sha256_file(ROOT / 'yolov8n.pt')
    report = dict(status='running', manifest=manifest, models={},
                  detector_sha256=detector_file_before, detector_tensor_sha256=detector_before,
                  no_optimizer_created=True, no_optimizer_step=True)
    write_report(report_path, report, first=True)

    for spec in args.models:
        name, path = spec.split('=', 1)
        checkpoint = ROOT / path
        expected_checkpoint = legacy['models'][name]
        checkpoint_hash_before = sha256_file(checkpoint)
        if path != expected_checkpoint['checkpoint'] or checkpoint_hash_before != expected_checkpoint['checkpoint_sha256']:
            raise RuntimeError(f'Checkpoint differs from original NP2: {name}')
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        checkpoint_iter = int(state.get('iter', -1))
        model = build_nafnet().to(device).eval()
        model.load_state_dict(state['ema'])
        del state
        model_before = state_tensor_sha256(model)
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        agg_gt, class_weight_total = [], defaultdict(float)
        cls_pos = defaultdict(float); cls_neg = defaultdict(float)
        box_sum = defaultdict(float); dfl_sum = defaultdict(float)
        ignored_fraction, gt_total = [], 0
        batches, max_loss_diff = [], 0.0

        for batch_index, batch in enumerate(loader):
            if args.max_batches and batch_index >= args.max_batches:
                break
            images = batch['img'].to(device)
            out = batch_forward(loss_fn, model(images), batch, keep_graph=False)
            assigned, probe_bce = out['assigned'], out['bce']
            native_scalar, native_stats = out['native_scalar'], out['native_stats']
            decomposed = float(decomposed_total(out['per_class']))
            diff = abs(float(native_scalar) - decomposed)
            max_loss_diff = max(max_loss_diff, diff)
            if not np.isfinite(diff) or diff >= 1e-4:
                raise RuntimeError(f'{name} statistics batch {batch_index} failed loss consistency: {diff}')
            batches.append(dict(batch=batch_index, images=len(batch['file_names']),
                                file_names=list(batch['file_names']),
                                gt_ids=[item['source_annotation_id'] for file_name in batch['file_names'] for item in gt_metadata[file_name]],
                                native_loss=float(native_scalar), decomposed_loss=decomposed,
                                abs_diff=diff,
                                parts={k: float(native_stats.get(f'{k}_loss', float('nan')))
                                       for k in ('box', 'cls', 'dfl')}))
            ignored_fraction.append(float(native_stats.get('ignored_anchor_fraction', 0.0)))
            gt_total += int(native_stats.get('gt_count', 0))

            fg = assigned['fg_mask']
            labels = assigned['target_labels'].long()
            tss = float(assigned['target_scores'].sum().clamp_min(1))

            # 逐 GT 正 anchor 数（沿用旧统计，target_gt_idx 是「图内」序号）
            cls = batch['cls'].flatten().long()
            batch_idx = batch['batch_idx'].long()
            for b in range(len(batch['file_names'])):
                classes_b = cls[batch_idx == b].tolist()
                for local, code in enumerate(classes_b):
                    picked = fg[b] & (assigned['target_gt_idx'][b] == local)
                    n_pos = int(picked.sum())
                    weight = float(assigned['target_scores'][b][picked].sum()) if n_pos else 0.0
                    meta = dict(gt_metadata[batch['file_names'][b]][local])
                    pixel_box = assigned['gt_bboxes'][b, local]
                    centers = assigned['anchor_points'] * assigned['stride_tensor']
                    inside = ((centers - pixel_box[:2]).amin(-1) > 1e-9) & ((pixel_box[2:] - centers).amin(-1) > 1e-9)
                    strides = assigned['stride_tensor'].flatten()
                    ignore_mask = loss_fn.criterion.bce.mask[b]
                    meta.update(detector_box_xyxy=pixel_box.detach().cpu().tolist(),
                                detector_width=float(pixel_box[2] - pixel_box[0]), detector_height=float(pixel_box[3] - pixel_box[1]),
                                anchor_centers_inside_by_stride={str(int(stride)): int((inside & (strides == stride)).sum()) for stride in strides.unique().tolist()},
                                candidate_centers_in_ignore=int((inside & ignore_mask).sum()),
                                positive_anchors_in_ignore=int((picked & ignore_mask).sum()))
                    agg_gt.append(dict(image=batch['file_names'][b], category=int(code),
                                       positives=n_pos, weight=weight, **meta))

            for b in range(fg.shape[0]):
                if fg[b].any():
                    for code, w in zip(labels[b][fg[b]].tolist(),
                                       assigned['target_scores'][b][fg[b]].sum(-1).tolist()):
                        class_weight_total[code] += float(w)

            pos = fg.unsqueeze(-1) & (assigned['target_scores'] > 0)
            for code in range(probe_bce.shape[-1]):
                total_c = float(probe_bce[..., code].sum())
                if total_c == 0:
                    continue
                p = float(probe_bce[..., code].masked_select(pos[..., code]).sum())
                cls_pos[code] += p / tss
                cls_neg[code] += (total_c - p) / tss

            # 逐类 box/DFL 直接取修正后分解的标量值
            for code in SUPERVISED:
                box_sum[code] += float(out['per_class'][code]['box'])
                dfl_sum[code] += float(out['per_class'][code]['dfl'])

        if args.grad:
            limit = args.grad_max_batches or args.max_batches
            for parameter in model.parameters():
                parameter.requires_grad_(True)
            gradient = gradient_check(model, loss_fn, dataset, order, device, args.grad_batch, limit, gt_metadata)
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            print(name, '逐批梯度校验：loss 最大偏差', gradient['loss_max_abs_diff'],
                  '梯度相对误差最大', gradient['grad_relative_l2_max'],
                  '余弦最小', gradient['grad_cosine_min'], flush=True)
        else:
            gradient = None

        model_after = state_tensor_sha256(model)
        detector_after = state_tensor_sha256(loss_fn.detector)
        checkpoint_hash_after = sha256_file(checkpoint)
        detector_file_after = sha256_file(ROOT / 'yolov8n.pt')
        invariants = dict(model_tensor_before=model_before, model_tensor_after=model_after,
                          detector_tensor_before=detector_before, detector_tensor_after=detector_after,
                          checkpoint_file_before=checkpoint_hash_before, checkpoint_file_after=checkpoint_hash_after,
                          detector_file_before=detector_file_before, detector_file_after=detector_file_after,
                          model_unchanged=model_before == model_after,
                          detector_including_bn_unchanged=detector_before == detector_after,
                          checkpoint_file_unchanged=checkpoint_hash_before == checkpoint_hash_after,
                          detector_file_unchanged=detector_file_before == detector_file_after,
                          detector_has_no_grad=all(p.grad is None for p in loss_fn.detector.parameters()),
                          optimizer_created=False, optimizer_steps=0)
        if not all(invariants[k] for k in ('model_unchanged', 'detector_including_bn_unchanged', 'checkpoint_file_unchanged', 'detector_file_unchanged', 'detector_has_no_grad')):
            raise RuntimeError(f'Frozen parameter/file invariant violated: {invariants}')
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
        for code in SUPERVISED:
            by_class[code]
        for code, entry in by_class.items():
            entry['name'] = CLASS_NAMES[code]
            entry['mean_positives'] = entry['positives'] / entry['gt'] if entry['gt'] else 0.0
            entry['zero_ratio'] = entry['zero'] / entry['gt'] if entry['gt'] else 0.0
            entry['cls_pos_ungained_legacy_units'] = cls_pos[code]
            entry['cls_neg_ungained_legacy_units'] = cls_neg[code]
            entry['cls_pos_contribution'] = cls_pos[code] * float(loss_fn.criterion.hyp.cls)
            entry['cls_neg_contribution'] = cls_neg[code] * float(loss_fn.criterion.hyp.cls)
            entry['box_contribution'] = box_sum[code]
            entry['dfl_contribution'] = dfl_sum[code]
            entry['target_weight_total'] = class_weight_total[code]

        gain = {key: float(getattr(loss_fn.criterion.hyp, key)) for key in ('cls', 'box', 'dfl')}
        reported_sum = sum(entry['cls_pos_contribution'] + entry['cls_neg_contribution'] + entry['box_contribution'] + entry['dfl_contribution'] for entry in by_class.values())
        native_sum = float(sum(b['native_loss'] for b in batches))
        if abs(reported_sum - native_sum) >= 1e-4:
            raise RuntimeError(f'Reported gained class terms do not sum to native: {reported_sum} vs {native_sum}')
        old_coverage = [(r['image'], r['category'], r['positives']) for r in expected_checkpoint['per_gt']]
        new_coverage = [(r['image'], r['category'], r['positives']) for r in agg_gt]
        report['models'][name] = dict(
            invariants=invariants, gain=gain,
            reported_class_component_total=reported_sum,
            reported_component_abs_diff=abs(reported_sum - native_sum),
            legacy_positive_coverage_exact_match=new_coverage == old_coverage,
            checkpoint=path, checkpoint_sha256=sha256_file(checkpoint), iter=checkpoint_iter,
            gt_count_total=gt_total, ignored_anchor_fraction_mean=float(np.mean(ignored_fraction)),
            native_loss_total_sum=float(sum(b['native_loss'] for b in batches)),
            decomposed_loss_total_sum=float(sum(b['decomposed_loss'] for b in batches)),
            batch_consistency=dict(batches=len(batches),
                                   max_abs_diff=max_loss_diff,
                                   all_batches_pass=bool(batches and max_loss_diff < 1e-4),
                                   per_batch=batches),
            per_class={str(k): v for k, v in sorted(by_class.items())}, per_gt=agg_gt,
            gradient=gradient,
            notes=('正 anchor 来自单次原生联合分配；cls_pos/cls_neg/box/dfl 为同一 loss 的按类别分解，未重新分配。'
                   'target_weight_total 只描述「这 64 张诊断样本中有多少监督」，不等于整个训练期的监督量。'
                   '梯度只报告范数与向量一致性，不把范数比例当作可加和的贡献率。'))
        write_report(report_path, report)
        print(name, {CLASS_NAMES[k]: (v['gt'], v['positives'], round(v['mean_positives'], 2), round(v['zero_ratio'], 3))
                     for k, v in sorted(by_class.items())}, '逐批 loss 最大偏差', max_loss_diff, flush=True)

    report['status'] = 'completed'
    report['completed_models'] = list(report['models'])
    write_report(report_path, report)
    print('已写入并验证', report_path)


if __name__ == '__main__':
    main()
