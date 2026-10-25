import pytest
import torch

from saferestore.losses.confidence import detector_confidence_loss


def raw(scores, *, nc=8, requires_grad=False):
    """Create a raw map with four regression bins and explicit class probabilities."""
    probabilities = torch.as_tensor(scores, dtype=torch.float32)
    if probabilities.ndim == 2:
        probabilities = probabilities.unsqueeze(0)
    logits = torch.logit(probabilities).unsqueeze(2)
    regression = torch.zeros(logits.shape[0], 4, 1, logits.shape[-1])
    return torch.cat((regression, logits), dim=1).requires_grad_(requires_grad)


def sample(*, car=0.1, bus=0.1, anchors=1):
    scores = torch.full((8, anchors), 0.1)
    scores[2] = car
    scores[5] = bus
    return scores


@pytest.mark.parametrize("student_score", [0.8, 0.9])
def test_equal_or_raised_confidence_is_free(student_score):
    teacher = raw(sample(bus=0.8), requires_grad=True)
    student = raw(sample(bus=student_score), requires_grad=True)
    loss = detector_confidence_loss([student], [teacher], nc=8)
    assert loss.item() == 0
    loss.backward()
    assert torch.count_nonzero(student.grad) == 0
    assert teacher.grad is None


def test_lower_confidence_penalty_and_gradient_raise_student_without_teacher_gradient():
    teacher = raw(sample(bus=0.8), requires_grad=True)
    student = raw(sample(bus=0.4), requires_grad=True)
    loss = detector_confidence_loss([student], [teacher], nc=8)
    assert loss.item() == pytest.approx(0.16)
    loss.backward()
    assert student.grad[0, 4 + 5, 0, 0] < 0
    assert teacher.grad is None
    assert torch.count_nonzero(student.grad[:, :4]) == 0


def test_no_positives_returns_differentiable_zero_and_finite_gradients():
    student = raw(sample(), requires_grad=True)
    loss, stats = detector_confidence_loss([student], [raw(sample())], nc=8, return_stats=True)
    assert loss.item() == 0
    assert stats["selected_anchors"].item() == 0
    assert stats["represented_images"].item() == 0
    loss.backward()
    assert torch.isfinite(student.grad).all()
    assert torch.count_nonzero(student.grad) == 0


def test_class_balance_prevents_many_cars_dominating_one_bus():
    teacher_scores = sample(car=0.8, anchors=20)
    teacher_scores[5, 0] = 0.8
    student_scores = teacher_scores.clone()
    student_scores[2] = 0.6
    student_scores[5, 0] = 0.4
    loss, stats = detector_confidence_loss(
        [raw(student_scores)], [raw(teacher_scores)], nc=8, topk=16, return_stats=True
    )
    assert loss.item() == pytest.approx((0.2**2 + 0.4**2) / 2)
    assert stats["selected_anchors"].item() == 17
    assert stats["selected_per_class"].tolist() == [0, 0, 16, 0, 1, 0, 0]
    assert stats["class_pairs_per_class"].tolist() == [0, 0, 1, 0, 1, 0, 0]


def test_balances_classes_per_image_then_batch_and_ignores_empty_classes():
    teacher_scores = torch.stack([sample(bus=0.8), sample(car=0.8, bus=0.8)])
    student_scores = teacher_scores.clone()
    student_scores[0, 5] = 0.4
    student_scores[1, 2] = 0.6
    loss = detector_confidence_loss([raw(student_scores)], [raw(teacher_scores)], nc=8)
    assert loss.item() == pytest.approx((0.16 + 0.04 / 2) / 2)


def test_global_topk_across_scales_uses_class_channels_and_matching_anchors():
    teacher_a, teacher_b = raw(sample(bus=0.6)), raw(sample(bus=0.9))
    student_a = raw(sample(bus=0.1), requires_grad=True)
    student_b = raw(sample(bus=0.8), requires_grad=True)
    teacher_a[:, :4] = 100  # Regression confidence must never enter selection.
    loss, stats = detector_confidence_loss(
        [student_a, student_b], [teacher_a, teacher_b], nc=8, topk=1, return_stats=True
    )
    assert loss.item() == pytest.approx(0.01)
    assert stats["selected_anchors"].item() == 1
    loss.backward()
    assert torch.count_nonzero(student_a.grad) == 0
    assert student_b.grad[0, 9, 0, 0] < 0


def test_threshold_is_inclusive_and_unsupported_classes_are_ignored():
    scores = sample(bus=0.25)
    scores[4] = 0.9  # Airplane is outside supported road classes.
    loss, stats = detector_confidence_loss([raw(scores)], [raw(scores)], nc=8, return_stats=True)
    assert loss.item() == 0
    assert stats["selected_anchors"].item() == 1


@pytest.mark.parametrize("kwargs", [
    {"nc": 0}, {"nc": True}, {"topk": 0}, {"topk": 1.5},
    {"threshold": -0.1}, {"threshold": float("nan")},
    {"classes": ()}, {"classes": (2, 2)}, {"classes": (8,)}, {"classes": (1.5,)},
])
def test_invalid_configuration_rejected(kwargs):
    options = {"nc": 8, **kwargs}
    with pytest.raises(ValueError):
        detector_confidence_loss([raw(sample())], [raw(sample())], **options)


@pytest.mark.parametrize("student,teacher", [
    ([], []), ([torch.zeros(1, 12, 1, 1)], []),
    ([torch.zeros(1, 12, 1, 1)], [torch.zeros(1, 12, 2, 1)]),
    ([torch.zeros(1, 11, 1, 1)], [torch.zeros(1, 11, 1, 1)]),
    ([torch.zeros(1, 12, 1)], [torch.zeros(1, 12, 1)]),
    ([torch.zeros(1, 12, 0, 1)], [torch.zeros(1, 12, 0, 1)]),
    ([torch.zeros(1, 12, 1, 1, dtype=torch.int64)],
     [torch.zeros(1, 12, 1, 1, dtype=torch.int64)]),
])
def test_invalid_layout_rejected(student, teacher):
    with pytest.raises(ValueError):
        detector_confidence_loss(student, teacher, nc=8)
