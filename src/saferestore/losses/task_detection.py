"""冻结本地 YOLOv8，复用原生匹配、分类、box 和 DFL 进行图像梯度监督。"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from saferestore.data.detection_gt import SUPERVISED_COCO_CLASSES


def ignore_anchor_mask(anchor_points, predicted_boxes, ignore_boxes, ignore_batch_idx, batch_size):
    """保守忽略与 crowd/rider 区域有交集的预测或中心落入区域的 anchor。

    这是训练负样本掩码，不是 COCO crowd 评估规则；原生正样本仍保留。
    坐标均为输入像素，predicted_boxes 为 B,N,4，anchor_points 为 N,2。
    """
    masks = torch.zeros((batch_size, len(anchor_points)), dtype=torch.bool, device=anchor_points.device)
    for index in range(batch_size):
        regions = ignore_boxes[ignore_batch_idx == index]
        if not len(regions):
            continue
        center_inside = ((anchor_points[:, None] >= regions[None, :, :2]) &
                         (anchor_points[:, None] <= regions[None, :, 2:])).all(-1)
        intersection = (torch.minimum(predicted_boxes[index, :, None, 2:], regions[None, :, 2:]) -
                        torch.maximum(predicted_boxes[index, :, None, :2], regions[None, :, :2])).clamp_min(0)
        overlap = intersection.prod(-1) > 0
        masks[index] = (center_inside | overlap).any(-1)
    return masks


class IgnoreRegionBCE(nn.Module):
    """原生 BCE，仅去除忽略区域中的负监督；已匹配正监督保持。"""

    def __init__(self):
        super().__init__()
        self.base = nn.BCEWithLogitsLoss(reduction="none")
        self.mask = None

    def forward(self, logits, targets):
        values = self.base(logits, targets)
        if self.mask is not None:
            values = values * (~self.mask[..., None] | (targets > 0)).to(values.dtype)
        return values


class FrozenYoloTaskLoss(nn.Module):
    """专用独立检测器实例；生产检测器和原始权重均不改动。"""

    def __init__(self, weights, device="cpu"):
        super().__init__()
        from ultralytics import YOLO
        from ultralytics.cfg import DEFAULT_CFG_DICT
        from ultralytics.utils.loss import v8DetectionLoss

        path = Path(weights)
        if not path.is_file():
            raise FileNotFoundError(f"缺少本地 YOLO 权重，禁止自动下载：{path}")
        self.detector = YOLO(str(path)).model.to(device).float().eval()
        if self.detector.model[-1].nc != 80 or self.detector.model[-1].end2end:
            raise ValueError("仅支持 COCO80 非 end2end YOLOv8 检测头")
        self.detector.requires_grad_(False)
        options = dict(DEFAULT_CFG_DICT)
        options.update(vars(self.detector.args) if isinstance(self.detector.args, SimpleNamespace) else self.detector.args)
        self.detector.args = SimpleNamespace(**options)
        self.criterion = v8DetectionLoss(self.detector)
        self.criterion.class_weights = torch.zeros((1, 1, 80), device=device)
        self.criterion.class_weights[..., list(SUPERVISED_COCO_CLASSES)] = 1
        self.criterion.bce = IgnoreRegionBCE()
        self.last_stats = {}

    def train(self, mode=True):
        # 外层 model.train() 不能使冻结检测器 BN 更新。
        super().train(False)
        self.detector.eval()
        return self

    def forward(self, images, batch):
        from ultralytics.utils.tal import make_anchors

        if images.ndim != 4 or images.shape[1] != 3 or images.shape[2] % 32 or images.shape[3] % 32:
            raise ValueError("检测输入必须为 RGB BCHW，边长为32倍数")
        self.detector.eval()
        native_batch = {key: batch[key].to(images.device) for key in ("batch_idx", "cls", "bboxes")}
        classes = native_batch["cls"].flatten()
        if len(classes) and not torch.isin(classes, classes.new_tensor(SUPERVISED_COCO_CLASSES)).all():
            raise ValueError("GT 含未支持类别或错误 COCO ID")
        # 必须保留此学生前向计算图。
        output = self.detector(images.float())
        raw = self.criterion.parse_output(output)
        if not isinstance(raw, dict) or not {"boxes", "scores", "feats"}.issubset(raw):
            raise RuntimeError("当前 Ultralytics 输出与已审计原生损失 API 不兼容")
        with torch.no_grad():
            anchor_points, stride = make_anchors(raw["feats"], self.criterion.stride, 0.5)
            predicted = self.criterion.bbox_decode(anchor_points, raw["boxes"].permute(0, 2, 1).contiguous()) * stride
            ignored = batch.get("ignore_boxes", images.new_empty((0, 4))).to(images.device)
            ignored = ignored * images.new_tensor([images.shape[3], images.shape[2], images.shape[3], images.shape[2]])
            indices = batch.get("ignore_batch_idx", images.new_empty((0,), dtype=torch.long)).to(images.device)
            mask = ignore_anchor_mask(anchor_points * stride, predicted, ignored, indices, len(images))
            self.criterion.bce.mask = mask
        native_values, detached = self.criterion(output, native_batch)
        # 原生返回各项乘 batch_size；取每图均值，microbatch变化不改变外部权重含义。
        scalar = native_values.sum() / len(images)
        stats = {key: float(value) for key, value in detached.items()}
        stats.update(total=float(scalar.detach()), ignored_anchor_fraction=float(mask.float().mean()),
                     gt_count=len(classes))
        self.last_stats = stats
        return scalar, stats
