"""使用本地假检测器验证置信度保护与既有特征损失的集成。"""

import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from saferestore.losses.confidence import detector_confidence_loss
from saferestore.losses.perceptual import DetectorFeatureLoss


def test_modern_dict_and_legacy_scales_preserve_logits_and_loss():
    generator = torch.Generator().manual_seed(123)
    teacher = [torch.randn(2, 144, 2, 3, generator=generator),
               torch.randn(2, 144, 1, 2, generator=generator)]
    student = []
    for item in teacher:
        output = item.clone()
        output[:, 64:] -= 0.3
        student.append(output.requires_grad_())
    flattened = torch.cat([item.flatten(2) for item in student], dim=2)
    modern = DetectorFeatureLoss._raw_maps((None, {
        "boxes": flattened[:, :64], "scores": flattened[:, 64:],
    }))
    legacy = DetectorFeatureLoss._raw_maps((None, student))
    assert torch.equal(modern[0].flatten(2), flattened)
    legacy_loss, legacy_stats = detector_confidence_loss(
        legacy, teacher, return_stats=True
    )
    teacher_flat = torch.cat([item.flatten(2) for item in teacher], dim=2)
    modern_loss, modern_stats = detector_confidence_loss(
        modern, [teacher_flat.unsqueeze(2)], return_stats=True
    )
    assert legacy_loss.item() > 0
    assert torch.equal(legacy_loss, modern_loss)
    for name in legacy_stats:
        assert torch.equal(legacy_stats[name], modern_stats[name])


class FakeHead(nn.Module):
    nc = 80
    f = [0]


class FakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        neck = nn.Conv2d(3, 4, kernel_size=1)
        with torch.no_grad():
            neck.weight.copy_(torch.arange(12).reshape(4, 3, 1, 1).float() / 12)
            neck.bias.copy_(torch.tensor([0.4, -0.2, 0.1, 0.3]))
        self.model = nn.ModuleList([neck, FakeHead()])
        self.calls = 0

    def forward(self, image):
        self.calls += 1
        features = self.model[0](image)
        logits = features.mean(dim=1, keepdim=True).expand(-1, 80, -1, -1)
        boxes = torch.zeros(image.shape[0], 64, image.shape[2] * image.shape[3])
        return None, {"boxes": boxes, "scores": logits.flatten(2)}


def make_feature_loss(monkeypatch, tmp_path, *, monitor, confidence_grad=False):
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(
        YOLO=lambda _weights: SimpleNamespace(model=FakeDetector())
    ))
    weights = tmp_path / "local.pt"
    weights.touch()
    return DetectorFeatureLoss(
        str(weights), confidence_monitor=monitor, confidence_grad=confidence_grad
    )


@pytest.mark.parametrize("confidence_grad", [False, True])
def test_monitor_keeps_feature_loss_and_gradient_and_detaches_teacher(
    monkeypatch, tmp_path, confidence_grad
):
    generator = torch.Generator().manual_seed(42)
    image = torch.rand(2, 3, 3, 4, generator=generator)
    clean = torch.rand(2, 3, 3, 4, generator=generator).requires_grad_()
    reference = torch.ones_like(image).requires_grad_()
    baseline_image = image.clone().requires_grad_()
    monitored_image = image.clone().requires_grad_()
    baseline = make_feature_loss(monkeypatch, tmp_path, monitor=False)
    monitored = make_feature_loss(
        monkeypatch, tmp_path, monitor=True, confidence_grad=confidence_grad
    )
    expected = baseline(baseline_image, clean)
    actual = monitored(monitored_image, clean, confidence_reference=reference)
    expected_gradient, = torch.autograd.grad(expected, baseline_image)
    actual_gradient, = torch.autograd.grad(actual, monitored_image, retain_graph=True)
    assert torch.equal(expected, actual)
    assert torch.equal(expected_gradient, actual_gradient)
    assert baseline.detector.calls == 2
    assert monitored.detector.calls == 3
    assert monitored.last_confidence_loss.item() > 0
    assert monitored.last_confidence_loss.requires_grad is confidence_grad
    assert monitored.last_confidence_stats["class_pairs_per_class"].tolist() == [2] * 7
    if confidence_grad:
        confidence_gradient, = torch.autograd.grad(
            monitored.last_confidence_loss, monitored_image
        )
        assert torch.isfinite(confidence_gradient).all()
        assert confidence_gradient.sum() < 0
    assert clean.grad is None
    assert reference.grad is None
    assert all(parameter.grad is None for parameter in monitored.detector.parameters())
    baseline.close()
    monitored.close()
