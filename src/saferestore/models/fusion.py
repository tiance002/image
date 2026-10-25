"""Validation-calibrated conservative fusion for restoration outputs."""

from __future__ import annotations

import torch


def calibrated_conservative_fusion(
    restored: torch.Tensor,
    weather: torch.Tensor,
    uncertainty: torch.Tensor,
    predicted_severity: torch.Tensor,
    *,
    severity_threshold: float,
    severity_temperature: float,
    uncertainty_threshold: float,
    strength: float,
    power: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Blend toward the input only for low-severity, high-risk regions.

    Returns the fused image and its per-pixel input-preservation weight.
    """
    if severity_temperature <= 0.0:
        raise ValueError("severity_temperature must be positive.")
    if not 0.0 <= uncertainty_threshold < 1.0:
        raise ValueError("uncertainty_threshold must be in [0, 1).")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1].")
    if power <= 0.0:
        raise ValueError("power must be positive.")

    severity = predicted_severity.reshape(-1, 1, 1, 1)
    image_gate = torch.sigmoid((severity_threshold - severity) / severity_temperature)
    pixel_gate = torch.clamp(
        (uncertainty - uncertainty_threshold) / (1.0 - uncertainty_threshold),
        0.0,
        1.0,
    ).pow(power)
    weight = torch.clamp(strength * image_gate * pixel_gate, 0.0, 1.0)
    fused = (1.0 - weight) * restored + weight * weather
    return fused, weight
