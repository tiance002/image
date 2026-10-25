"""检测器特征一致性损失。

设计动机
--------
项目原计划用 ACDC 的 GT 框做"检测一致性损失"，但实测发现 ACDC 的
``rgb_ref_anon`` 清晰参考帧与原退化帧**并非同一时刻**（同一路段、不同视角，
梯度相关系数约 0.000），GT 框无法迁移到清晰帧上。因此改走不需要框的路线：

    让复原图在冻结检测器的特征空间里，靠近"清晰图"的特征分布。

这正是任务驱动复原（task-driven restoration）的标准做法：检测器的中间层
特征本身就是"对检测有用的结构"的编码，用它做监督比像素损失更贴近下游任务。

实现要点
--------
1. 只挂 Detect 头的**输入层**（由 ``model.model[-1].f`` 自动探测），
   即 P3/P4/P5 三个尺度的颈部输出，避免依赖硬编码层号。
2. 用余弦距离而非 L2：特征幅值会随图像整体亮度漂移，余弦只约束方向，
   不会把"亮度对齐"这种与检测无关的事也强加给网络。
3. 检测器全程 ``eval()`` + ``requires_grad_(False)``，梯度只回传到输入图像。
4. 目标特征在 ``torch.no_grad()`` 下计算并 detach，避免目标侧跟着动。
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from saferestore.losses.confidence import detector_confidence_loss


def prepare_detector_view(
    restored: torch.Tensor, target: torch.Tensor, size: int = 256, mode: str = "crop"
) -> tuple[torch.Tensor, torch.Tensor]:
    """在同一计算预算下选择中心裁剪或保留完整训练块的缩小视野。"""
    if restored.shape != target.shape:
        raise ValueError("复原图与目标图尺寸必须一致")
    if mode not in {"crop", "resize"} or size < 0:
        raise ValueError("检测器视野模式必须为 crop/resize，尺寸不能为负")
    if size == 0 or max(restored.shape[-2:]) <= size:
        return restored, target
    if mode == "resize":
        # 训练块为正方形；两侧使用相同缩小规则，避免目标结构错位。
        return (
            F.interpolate(restored, size=(size, size), mode="area"),
            F.interpolate(target, size=(size, size), mode="area"),
        )
    height, width = restored.shape[-2:]
    top, left = max((height - size) // 2, 0), max((width - size) // 2, 0)
    return (
        restored[:, :, top : top + size, left : left + size],
        target[:, :, top : top + size, left : left + size],
    )


def feature_magnitude_loss(features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """补充余弦距离忽略的通道幅度信息；目标侧不接收梯度。"""
    # 在均方值内加 epsilon，使零向量的反向传播也有限。
    log_rms = 0.5 * (features.float().square().mean(dim=1) + 1e-6).log()
    target_log_rms = 0.5 * (target.detach().float().square().mean(dim=1) + 1e-6).log()
    return F.smooth_l1_loss(log_rms, target_log_rms)


class DetectorFeatureLoss(nn.Module):
    """冻结检测器特征空间上的一致性损失。

    参数
    ----
    weights:
        YOLO 权重路径（本项目使用本地 ``yolov8n.pt``，不联网下载）。
    layer_indices:
        要挂 hook 的层号；为 ``None`` 时自动从 Detect 头的输入索引推断。
    """

    def __init__(
        self,
        weights: str,
        layer_indices: tuple[int, ...] | None = None,
        layer_weights: tuple[float, ...] | None = None,
        magnitude_weight: float = 0.0,
        confidence_monitor: bool = False,
        confidence_grad: bool = False,
        confidence_threshold: float = 0.25,
        confidence_topk: int = 16,
    ) -> None:
        super().__init__()
        if not torch.isfinite(torch.tensor(magnitude_weight)) or magnitude_weight < 0:
            raise ValueError("幅度损失权重必须为有限非负数")
        self.magnitude_weight = float(magnitude_weight)
        self.confidence_monitor = confidence_monitor
        self.confidence_grad = confidence_grad
        self.confidence_threshold = confidence_threshold
        self.confidence_topk = confidence_topk
        self.last_confidence_loss: torch.Tensor | None = None
        self.last_confidence_stats: dict[str, torch.Tensor] = {}
        # 缺失本地权重时立即退出，禁止 YOLO 按文件名自动联网下载。
        if not Path(weights).is_file():
            raise FileNotFoundError(f"检测器本地权重不存在（禁止自动下载）：{weights}")
        from ultralytics import YOLO

        yolo = YOLO(str(weights))
        self.detector = yolo.model.float().eval()
        for param in self.detector.parameters():
            param.requires_grad_(False)

        if layer_indices is None:
            # Detect 头（最后一层）的 .f 就是它用到的三个颈部输出层号
            head = self.detector.model[-1]
            layer_indices = tuple(int(i) for i in head.f)
        self.layer_indices = tuple(layer_indices)

        # 尺度权重默认为等权（保持 v2 的行为不变）。顺序与 layer_indices 一致，
        # 即 (P3, P4, P5) —— P3 分辨率最高、负责小目标。
        if layer_weights is None:
            self.layer_weights = tuple(1.0 for _ in self.layer_indices)
        else:
            if len(layer_weights) != len(self.layer_indices):
                raise ValueError(
                    f"尺度权重个数 {len(layer_weights)} 与特征层数 "
                    f"{len(self.layer_indices)} 不一致"
                )
            self.layer_weights = tuple(float(w) for w in layer_weights)

        self._features: dict[int, torch.Tensor] = {}
        self._handles = [
            self.detector.model[i].register_forward_hook(self._make_hook(i))
            for i in self.layer_indices
        ]

    def _make_hook(self, index: int):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, (list, tuple)) else output
            self._features[index] = tensor

        return hook

    def forward(
        self, restored: torch.Tensor, clean: torch.Tensor,
        confidence_reference: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """restored / clean 均为 (B,3,H,W)、取值范围 [0,1]、RGB 顺序。"""
        # 目标侧不建图，省显存也避免目标漂移
        with torch.no_grad():
            self._features.clear()
            self.detector(clean.detach())
            targets = {i: f.detach().clone() for i, f in self._features.items()}

        teacher_raw = None
        if self.confidence_monitor:
            if confidence_reference is None or confidence_reference.shape != restored.shape:
                raise ValueError("置信度教师必须是与复原图视野一致的原始退化输入")
            with torch.no_grad():
                result = self.detector(confidence_reference.detach())
                teacher_raw = [item.detach().clone() for item in self._raw_maps(result)]

        self._features.clear()
        result = self.detector(restored)
        outputs = dict(self._features)
        self.last_confidence_loss = None
        self.last_confidence_stats = {}
        if teacher_raw is not None:
            # 对照组也计算同一监测值，但不保留新增损失的反传图。
            with torch.set_grad_enabled(torch.is_grad_enabled() and self.confidence_grad):
                self.last_confidence_loss, self.last_confidence_stats = detector_confidence_loss(
                    self._raw_maps(result), teacher_raw,
                    nc=int(self.detector.model[-1].nc), threshold=self.confidence_threshold,
                    topk=self.confidence_topk, return_stats=True,
                )

        losses = []
        weight_sum = 0.0
        for index, layer_weight in zip(self.layer_indices, self.layer_weights):
            feat = outputs[index]
            target = targets[index]
            # 逐样本在通道维上算余弦相似度，再取平均：
            # 不用全局平均，避免批内样本互相"借"方向。
            cos = torch.nn.functional.cosine_similarity(
                feat.flatten(2), target.flatten(2), dim=1
            )
            losses.append(layer_weight * (1.0 - cos).mean())
            weight_sum += layer_weight
        # 除以权重和，保证调整尺度配比时总损失量级不变，
        # 否则等于顺带改了特征损失的总体强度，实验无法归因。
        loss = torch.stack(losses).sum() / max(weight_sum, 1e-6)
        if self.magnitude_weight > 0:
            # 只在最低分辨率层增加约束，不再执行额外的检测器前向。
            index = self.layer_indices[-1]
            loss = loss + self.magnitude_weight * feature_magnitude_loss(outputs[index], targets[index])
        return loss

    @staticmethod
    def _raw_maps(result) -> list[torch.Tensor]:
        """兼容本机dict与旧版list输出；只读取sigmoid之前的分类logits。"""
        if isinstance(result, tuple) and len(result) == 2:
            raw = result[1]
            if isinstance(raw, list):
                return raw
            if isinstance(raw, dict) and "boxes" in raw and "scores" in raw:
                # 本机版本已把三个尺度展平拼接，保留相同锚点顺序即可逐点对应。
                return [torch.cat((raw["boxes"], raw["scores"]), dim=1).unsqueeze(2)]
        raise RuntimeError("YOLO输出不含预期的原始分类logits，停止训练")

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
