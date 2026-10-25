import torch

from saferestore.losses import SafeRestorationLoss
from saferestore.models import FidelityPreservingRestorer


def test_loss_is_finite_and_differentiable() -> None:
    model = FidelityPreservingRestorer(base_channels=8)
    criterion = SafeRestorationLoss(
        {
            "pixel": 1.0,
            "edge": 0.2,
            "ssim": 0.1,
            "uncertainty": 0.05,
            "uncertainty_regularization": 0.01,
            "degradation": 0.1,
        }
    )
    weather = torch.rand(1, 3, 32, 32)
    clean = torch.rand(1, 3, 32, 32)
    outputs = model(weather)
    loss, components = criterion(
        outputs,
        clean,
        weather_label=torch.tensor([2]),
        severity=torch.tensor([0.85]),
    )

    assert torch.isfinite(loss)
    assert set(components) == {
        "total",
        "pixel",
        "edge",
        "ssim",
        "uncertainty",
        "uncertainty_mean",
        "degradation",
    }
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
