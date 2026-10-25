import pytest
import torch

from scripts.restore_directory import select_restoration


def outputs_fixture() -> dict[str, torch.Tensor]:
    return {
        "restored": torch.full((1, 3, 2, 2), 0.8),
        "conservative": torch.full((1, 3, 2, 2), 0.6),
        "uncertainty": torch.ones((1, 1, 2, 2)),
        "degradation": torch.tensor([[0.0, 0.0, 0.1]]),
    }


def test_select_restoration_supports_all_frozen_methods() -> None:
    outputs = outputs_fixture()
    weather = torch.full((1, 3, 2, 2), 0.2)
    parameters = {
        "severity_threshold": 0.5,
        "severity_temperature": 0.1,
        "uncertainty_threshold": 0.0,
        "strength": 1.0,
        "power": 1.0,
    }

    assert torch.equal(select_restoration(outputs, weather, "restored"), outputs["restored"])
    assert torch.equal(
        select_restoration(outputs, weather, "conservative"), outputs["conservative"]
    )
    calibrated = select_restoration(outputs, weather, "calibrated", parameters)
    assert torch.all(calibrated < outputs["restored"])
    assert torch.all(calibrated > weather)


def test_select_restoration_requires_calibration() -> None:
    with pytest.raises(ValueError, match="requires fusion parameters"):
        select_restoration(outputs_fixture(), torch.zeros((1, 3, 2, 2)), "calibrated")
