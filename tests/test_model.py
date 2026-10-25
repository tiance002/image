import torch

from saferestore.models import FidelityPreservingRestorer


def test_model_output_contract() -> None:
    model = FidelityPreservingRestorer(base_channels=8)
    image = torch.rand(2, 3, 64, 96)
    outputs = model(image)

    assert outputs["restored"].shape == image.shape
    assert outputs["conservative"].shape == image.shape
    assert outputs["uncertainty"].shape == (2, 1, 64, 96)
    assert outputs["degradation"].shape == (2, 4)
    assert outputs["restoration_strength"].shape == (2, 1, 1, 1)
    assert torch.all((outputs["restored"] >= 0) & (outputs["restored"] <= 1))
    assert torch.all((outputs["uncertainty"] >= 0) & (outputs["uncertainty"] <= 1))


def test_degradation_gate_bounds_restoration_strength() -> None:
    model = FidelityPreservingRestorer(
        base_channels=8,
        use_degradation_gate=True,
        minimum_restoration_strength=0.25,
    )
    outputs = model(torch.rand(2, 3, 32, 48))
    strength = outputs["restoration_strength"]
    assert torch.all(strength >= 0.25)
    assert torch.all(strength <= 1.0)
