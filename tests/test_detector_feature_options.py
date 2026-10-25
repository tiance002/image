import torch
import torch.nn.functional as F

from saferestore.losses.perceptual import feature_magnitude_loss, prepare_detector_view


def test_default_crop_is_exactly_the_legacy_center_view():
    image = torch.arange(320 * 320, dtype=torch.float32).reshape(1, 1, 320, 320)
    output, target = prepare_detector_view(image, image.clone())
    assert torch.equal(output, image[:, :, 32:288, 32:288])
    assert torch.equal(output, target)


def test_resize_keeps_border_context_alignment_and_gradient():
    image = torch.zeros(1, 3, 320, 320)
    image[:, :, :16] = 1
    image.requires_grad_()
    cropped, _ = prepare_detector_view(image, image.detach())
    resized, target = prepare_detector_view(image, image.detach(), mode="resize")
    assert cropped.sum() == 0
    assert resized.shape == (1, 3, 256, 256)
    assert resized.sum() > 0
    assert torch.equal(resized, target)
    resized.sum().backward()
    assert torch.isfinite(image.grad).all()
    assert image.grad[:, :, :16].sum() > 0


def test_magnitude_detects_positive_scaling_that_cosine_ignores():
    target = torch.ones(1, 4, 3, 3, requires_grad=True)
    output = torch.full((1, 4, 3, 3), 2.0, requires_grad=True)
    assert (1 - F.cosine_similarity(output, target, dim=1)).abs().max() < 1e-6
    loss = feature_magnitude_loss(output, target)
    assert loss > 0
    loss.backward()
    assert output.grad.abs().sum() > 0
    assert target.grad is None


def test_identical_and_zero_features_have_finite_zero_magnitude_loss():
    for value in (0.0, 1.0):
        features = torch.full((1, 4, 3, 3), value, requires_grad=True)
        loss = feature_magnitude_loss(features, features.detach())
        assert loss == 0
        loss.backward()
        assert torch.isfinite(features.grad).all()
