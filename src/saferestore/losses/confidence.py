"""在空间对齐的 YOLOv8 原始输出上单侧保持高置信度检测。"""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch


SUPPORTED_CLASSES = (0, 1, 2, 3, 5, 6, 7)


def detector_confidence_loss(
    student_raw: Sequence[torch.Tensor],
    teacher_raw: Sequence[torch.Tensor],
    *,
    nc: int = 80,
    classes: Sequence[int] = SUPPORTED_CLASSES,
    threshold: float = 0.25,
    topk: int = 16,
    return_stats: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """仅惩罚相对于已分离梯度的输入教师分支的置信度下降。

    输入为逐尺度对应的原始输出 ``[B, 4 * reg_max + nc, H, W]``。
    最后 ``nc`` 个通道为类别 logits；回归通道不参与计算。
    教师分支必须使用与复原学生图像相同视野的退化输入，确保锚点空间对应。

    对每张图像的每个类别，从全部尺度中选取至多 ``topk`` 个教师锚点，
    其 sigmoid 分数须 >= ``threshold``。先对所选锚点的概率下降平方求均值，
    再对存在所选锚点的类别求均值，最后对批次求均值。
    没有所选锚点的图像贡献可微的零值。学生置信度提高不受惩罚，
    教师分支始终不接收梯度。

    可选的已分离梯度统计包括所选锚点总数、有监督的图像/类别对数、
    有监督的图像数，以及按传入类别顺序排列的各类别所选锚点数和图像/类别对数。
    统计保持张量形式，以避免设备同步。
    """
    if isinstance(nc, bool) or not isinstance(nc, int) or nc < 1:
        raise ValueError("nc must be a positive integer")
    if isinstance(topk, bool) or not isinstance(topk, int) or topk < 1:
        raise ValueError("topk must be a positive integer")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and between 0 and 1")
    class_ids = tuple(classes)
    if not class_ids or len(set(class_ids)) != len(class_ids):
        raise ValueError("classes must be nonempty and unique")
    if any(isinstance(c, bool) or not isinstance(c, int) or not 0 <= c < nc for c in class_ids):
        raise ValueError("classes must contain integer indices in [0, nc)")
    if not isinstance(student_raw, (list, tuple)) or not isinstance(teacher_raw, (list, tuple)):
        raise ValueError("raw outputs must be lists or tuples of per-scale tensors")
    if not student_raw or len(student_raw) != len(teacher_raw):
        raise ValueError("student and teacher must have the same nonzero number of scales")

    student_logits, teacher_logits = [], []
    first = student_raw[0]
    if not isinstance(first, torch.Tensor) or first.ndim != 4:
        raise ValueError("raw scales must be four-dimensional tensors")
    batch, channels = first.shape[:2]
    device = first.device
    if batch < 1 or channels <= nc or (channels - nc) % 4:
        raise ValueError("raw channels must be 4 * reg_max + nc with a nonempty batch")
    for student, teacher in zip(student_raw, teacher_raw):
        if not isinstance(student, torch.Tensor) or not isinstance(teacher, torch.Tensor):
            raise ValueError("raw scales must be tensors")
        if student.ndim != 4 or student.shape != teacher.shape:
            raise ValueError("corresponding student/teacher scales must have matching BCHW shapes")
        if student.shape[:2] != (batch, channels) or min(student.shape[-2:]) < 1:
            raise ValueError("scales must share batch/channels and have nonempty spatial dimensions")
        if student.device != device or teacher.device != device:
            raise ValueError("all raw scales must be on the same device")
        if not student.is_floating_point() or not teacher.is_floating_point():
            raise ValueError("raw logits must be floating-point tensors")
        # 用 float32 计算 sigmoid 与归约，保持 AMP 下的数值稳定性。
        student_logits.append(student[:, -nc:, :, :][:, class_ids].float().flatten(2))
        teacher_logits.append(teacher.detach()[:, -nc:, :, :][:, class_ids].float().flatten(2))

    student_scores = torch.cat(student_logits, dim=2).sigmoid()
    teacher_scores = torch.cat(teacher_logits, dim=2).sigmoid()
    scores, indices = teacher_scores.topk(min(topk, teacher_scores.shape[2]), dim=2)
    selected = scores >= threshold
    counts = selected.sum(dim=2)
    deficits = (scores - student_scores.gather(2, indices)).clamp_min(0).square()
    per_class = (deficits * selected).sum(dim=2) / counts.clamp_min(1)
    represented = counts > 0
    per_image = per_class.sum(dim=1) / represented.sum(dim=1).clamp_min(1)
    loss = per_image.mean()
    if not return_stats:
        return loss
    stats = {
        "selected_anchors": counts.sum().detach(),
        "represented_class_pairs": represented.sum().detach(),
        "represented_images": represented.any(dim=1).sum().detach(),
        "selected_per_class": counts.sum(dim=0).detach(),
        "class_pairs_per_class": represented.sum(dim=0).detach(),
    }
    return loss, stats
