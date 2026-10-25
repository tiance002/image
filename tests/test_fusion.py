import pytest
import torch

from saferestore.models import calibrated_conservative_fusion


def test_zero_strength_returns_direct_restoration() -> None:
    restored = torch.rand(2, 3, 8, 8)
    weather = torch.rand(2, 3, 8, 8)
    uncertainty = torch.rand(2, 1, 8, 8)
    fused, weight = calibrated_conservative_fusion(
        restored,
        weather,
        uncertainty,
        torch.tensor([0.3, 0.8]),
        severity_threshold=0.5,
        severity_temperature=0.05,
        uncertainty_threshold=0.2,
        strength=0.0,
        power=1.0,
    )
    assert torch.equal(fused, restored)
    assert torch.count_nonzero(weight) == 0


def test_low_severity_receives_more_input_preservation() -> None:
    restored = torch.zeros(2, 3, 4, 4)
    weather = torch.ones_like(restored)
    uncertainty = torch.full((2, 1, 4, 4), 0.8)
    _, weight = calibrated_conservative_fusion(
        restored,
        weather,
        uncertainty,
        torch.tensor([0.3, 0.8]),
        severity_threshold=0.5,
        severity_temperature=0.05,
        uncertainty_threshold=0.2,
        strength=1.0,
        power=1.0,
    )
    assert weight[0].mean() > weight[1].mean()


def test_invalid_fusion_parameters_are_rejected() -> None:
    image = torch.zeros(1, 3, 2, 2)
    uncertainty = torch.zeros(1, 1, 2, 2)
    with pytest.raises(ValueError):
        calibrated_conservative_fusion(
            image,
            image,
            uncertainty,
            torch.tensor([0.5]),
            severity_threshold=0.5,
            severity_temperature=0.0,
            uncertainty_threshold=0.2,
            strength=1.0,
            power=1.0,
        )
