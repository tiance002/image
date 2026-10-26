"""NP2-R：逐类损失分解的针对性测试（CPU 即可运行）。

每个用例都拿**原生 BboxLoss / IgnoreRegionBCE 的输出**当基准，而不是拿自己和自己比。
同时把旧版的两处写法（网格/像素混用、weight=[N] 广播）作为「反例」一起跑，
要求反例必须失败——否则说明测试没有覆盖到缺陷。

覆盖用例：
- zero_positive          零正 anchor
- single_positive        单正 anchor
- multi_positive         多正 anchor、跨两个类别
- multi_stride           8/16/32 三个 stride 层混合（坐标尺度缺陷主要在这里暴露）
- weight_shape           权重形状 [N] vs [N,1]
- crowd_rider_mask       crowd/rider 忽略掩码下的正负监督
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from saferestore.losses.task_detection import FrozenYoloTaskLoss, ignore_anchor_mask
from scripts.np2r_supervision_probe import SUPERVISED, class_components

TOL = 1e-5  # 相对容差：float32 下逐类分别求和与整张量一次性求和会有 ~1e-7 量级的顺序误差


def close(a, b, tol=TOL):
    """相对容差比较：大小量级不同（如 cls 可达 1e5）时不能用绝对容差。"""
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def fake_anchors(levels, device='cpu'):
    """按给定 (stride, h, w) 构造 anchor 与 stride 张量，等价于 make_anchors 的输出。"""
    from ultralytics.utils.tal import make_anchors
    feats = [torch.zeros(1, 64, h, w, device=device) for _, h, w in levels]
    strides = torch.tensor([s for s, _, _ in levels], device=device, dtype=torch.float32)
    return make_anchors(feats, strides, 0.5)


def build_case(criterion, levels, positive_specs, seed=0):
    """positive_specs: [(anchor_index, class_code, dx, dy)]，dx/dy 是相对 anchor 中心的像素偏移。"""
    torch.manual_seed(seed)
    anchor_points, stride_tensor = fake_anchors(levels)
    num_anchors = anchor_points.shape[0]
    pred_distri = torch.randn(1, num_anchors, criterion.reg_max * 4) * 0.5
    pred_bboxes = criterion.bbox_decode(anchor_points, pred_distri)

    target_scores = torch.zeros(1, num_anchors, criterion.nc)
    target_labels = torch.zeros(1, num_anchors, dtype=torch.float32)
    target_bboxes = torch.zeros(1, num_anchors, 4)
    fg_mask = torch.zeros(1, num_anchors, dtype=torch.bool)
    for index, code, dx, dy in positive_specs:
        centre = anchor_points[index] * stride_tensor[index]
        width, height = 40.0, 24.0
        target_bboxes[0, index] = torch.tensor([
            centre[0] - width / 2 + dx, centre[1] - height / 2 + dy,
            centre[0] + width / 2 + dx, centre[1] + height / 2 + dy])
        target_scores[0, index, code] = 1.0
        target_labels[0, index] = code
        fg_mask[0, index] = True
    return dict(anchor_points=anchor_points, stride_tensor=stride_tensor, pred_distri=pred_distri,
                pred_bboxes=pred_bboxes, target_scores=target_scores, target_labels=target_labels,
                target_bboxes=target_bboxes, fg_mask=fg_mask)


def native_bbox_loss(criterion, case, bce=None):
    """原生 BboxLoss + 原生 BCE，作为唯一基准。"""
    tss = case['target_scores'].sum().clamp_min(1)
    if not bool(case['fg_mask'].any()):
        box = dfl = torch.zeros(())
    else:
        box, dfl = criterion.bbox_loss(
            case['pred_distri'], case['pred_bboxes'], case['anchor_points'],
            case['target_bboxes'] / case['stride_tensor'],  # ← 原生就在这里统一尺度
            case['target_scores'], tss, case['fg_mask'],
            torch.tensor([640.0, 640.0]), case['stride_tensor'])
    box = box * criterion.hyp.box
    dfl = dfl * criterion.hyp.dfl
    cls = None
    if bce is not None:
        cls = bce.sum() / tss * criterion.hyp.cls
    return dict(box=float(box), dfl=float(dfl), cls=None if cls is None else float(cls), tss=float(tss))


def weighted_bce(criterion, logits, target_scores):
    """原生口径：IgnoreRegionBCE 之后乘 class_weights（只有 7 类有监督）。"""
    values = criterion.bce(logits, target_scores)
    if criterion.class_weights is not None:
        values = values * criterion.class_weights
    return values


def my_totals(criterion, case, bce):
    assigned = dict(pred_distri=case['pred_distri'], pred_bboxes=case['pred_bboxes'],
                    anchor_points=case['anchor_points'], target_bboxes=case['target_bboxes'],
                    target_scores=case['target_scores'], fg_mask=case['fg_mask'],
                    target_labels=case['target_labels'], stride_tensor=case['stride_tensor'])
    per_class = class_components(criterion, assigned, bce)
    return dict(
        box=float(sum(per_class[c]['box'] for c in SUPERVISED)),
        dfl=float(sum(per_class[c]['dfl'] for c in SUPERVISED)),
        cls=float(sum(per_class[c]['cls'] for c in SUPERVISED)),
        total=float(sum(per_class[c]['total'] for c in SUPERVISED)),
        per_class=per_class)


def legacy_totals(criterion, case, bce, use_weight_vector=False):
    """旧实现的反例：不除 stride、weight 用 [N]。"""
    from ultralytics.utils.metrics import bbox_iou
    from ultralytics.utils.tal import bbox2dist
    fg = case['fg_mask']
    tss = case['target_scores'].sum().clamp_min(1)
    if not bool(fg.any()):
        return dict(box=0.0, dfl=0.0)
    weight = case['target_scores'].sum(-1)[fg]
    if not use_weight_vector:
        weight = weight.unsqueeze(-1)
    iou = bbox_iou(case['pred_bboxes'][fg], case['target_bboxes'][fg], xywh=False, CIoU=True)
    box = float((((1.0 - iou) * weight) / tss * criterion.hyp.box).sum())
    target_ltrb = bbox2dist(case['anchor_points'], case['target_bboxes'], criterion.reg_max - 1)
    dfl = criterion.bbox_loss.dfl_loss(
        case['pred_distri'][fg].view(-1, criterion.reg_max), target_ltrb[fg]) * weight / tss * criterion.hyp.dfl
    return dict(box=box, dfl=float(dfl.sum()))


def main():
    device = 'cpu'
    loss_fn = FrozenYoloTaskLoss(ROOT / 'yolov8n.pt', device=device)
    criterion = loss_fn.criterion
    results, failures = [], []

    def record(name, ok, detail):
        results.append(dict(case=name, ok=bool(ok), detail=detail))
        if not ok:
            failures.append(name)
        print(('PASS' if ok else 'FAIL'), name, detail, flush=True)

    # ---- 1. 零正样本 ----
    case = build_case(criterion, [(8, 80, 80)], [])
    bce = weighted_bce(criterion, case['pred_distri'].new_zeros(1, case['anchor_points'].shape[0], criterion.nc),
                       case['target_scores'])
    base = native_bbox_loss(criterion, case, bce)
    mine = my_totals(criterion, case, bce)
    record('zero_positive', mine['box'] == 0 and mine['dfl'] == 0 and close(mine['cls'], base['cls']),
           dict(native=base['cls'], mine=mine['cls']))

    # ---- 2. 单正样本 ----
    case = build_case(criterion, [(8, 80, 80)], [(1234, 5, 0.0, 0.0)])
    scores = torch.randn(1, case['anchor_points'].shape[0], criterion.nc)
    bce = weighted_bce(criterion, scores, case['target_scores'])
    base, mine = native_bbox_loss(criterion, case, bce), my_totals(criterion, case, bce)
    ok = (close(mine['box'], base['box']) and close(mine['dfl'], base['dfl'])
          and close(mine['cls'], base['cls']))
    record('single_positive', ok, dict(native=(base['box'], base['dfl']), mine=(mine['box'], mine['dfl'])))

    # ---- 3. 多正样本、跨类别 ----
    specs = [(100, 5, 0.0, 0.0), (2001, 5, 3.0, -2.0), (3999, 6, -1.0, 4.0), (5000, 6, 0.0, 0.0), (6100, 0, 2.0, 2.0)]
    case = build_case(criterion, [(8, 80, 80)], specs)
    scores = torch.randn(1, case['anchor_points'].shape[0], criterion.nc)
    bce = weighted_bce(criterion, scores, case['target_scores'])
    base, mine = native_bbox_loss(criterion, case, bce), my_totals(criterion, case, bce)
    ok = (close(mine['box'], base['box']) and close(mine['dfl'], base['dfl'])
          and close(mine['cls'], base['cls']))
    record('multi_positive', ok, dict(native=(base['box'], base['dfl']), mine=(mine['box'], mine['dfl'])))

    # ---- 4. 多 stride（8/16/32 混合）----
    levels = [(8, 80, 80), (16, 40, 40), (32, 20, 20)]
    specs = [(500, 5, 0.0, 0.0), (7000, 6, 0.0, 0.0), (8200, 5, 5.0, 0.0), (2000, 0, 0.0, 0.0)]
    case = build_case(criterion, levels, specs)
    scores = torch.randn(1, case['anchor_points'].shape[0], criterion.nc)
    bce = weighted_bce(criterion, scores, case['target_scores'])
    base, mine = native_bbox_loss(criterion, case, bce), my_totals(criterion, case, bce)
    ok = (close(mine['box'], base['box']) and close(mine['dfl'], base['dfl']))
    record('multi_stride', ok, dict(native=(base['box'], base['dfl']), mine=(mine['box'], mine['dfl'])))

    # ---- 5. 两个反例必须失败：不除 stride / weight=[N] ----
    legacy_scale = legacy_totals(criterion, case, bce)
    legacy_shape = legacy_totals(criterion, case, bce, use_weight_vector=True)
    record('legacy_no_stride_should_mismatch', abs(legacy_scale['box'] - base['box']) > 1e-3,
           dict(native=base['box'], legacy=legacy_scale['box']))
    record('legacy_weight_broadcast_should_mismatch', abs(legacy_shape['box'] - base['box']) > 1e-3,
           dict(native=base['box'], legacy=legacy_shape['box']))

    # ---- 6. crowd/rider 忽略掩码 ----
    case = build_case(criterion, [(8, 80, 80)], [(1200, 5, 0.0, 0.0), (1210, 5, 0.0, 0.0), (1300, 0, 0.0, 0.0)])
    scores = torch.randn(1, case['anchor_points'].shape[0], criterion.nc)
    # 忽略区域按 anchor 中心构造，确保确定性地覆盖 1210（正样本）与 1300（负样本）
    centers = case['anchor_points'] * case['stride_tensor']

    def box_around(index, radius=6.0):
        centre = centers[index]
        return [centre[0] - radius, centre[1] - radius, centre[0] + radius, centre[1] + radius]

    regions = torch.tensor([box_around(1210), box_around(1300)])
    predicted = case['pred_bboxes'] * case['stride_tensor']
    mask = ignore_anchor_mask(case['anchor_points'] * case['stride_tensor'], predicted, regions,
                              torch.zeros(len(regions), dtype=torch.long), 1)
    criterion.bce.mask = mask
    bce_masked = weighted_bce(criterion, scores, case['target_scores'])
    criterion.bce.mask = None
    base = native_bbox_loss(criterion, case, bce_masked)
    mine = my_totals(criterion, case, bce_masked)
    positives = case['target_scores'] > 0
    kept_positive = float(bce_masked.masked_select(positives).sum()) > 0
    negative_zeroed = bool((bce_masked[0, mask[0]][(case['target_scores'][0, mask[0]] == 0)].abs().sum() == 0))
    ok = (close(mine['cls'], base['cls']) and kept_positive and negative_zeroed)
    record('crowd_rider_mask', ok,
           dict(masked_anchors=int(mask.sum()), kept_positive=kept_positive,
                negative_zeroed=negative_zeroed, native_cls=base['cls'], mine_cls=mine['cls']))

    print()
    print('汇总：', sum(r['ok'] for r in results), '/', len(results), '通过')
    if failures:
        raise SystemExit(f'以下用例未通过：{failures}')


if __name__ == '__main__':
    main()
