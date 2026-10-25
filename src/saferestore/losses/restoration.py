"""Losses for fidelity-preserving restoration."""

from __future__ import annotations

import torch
from torch import nn

from saferestore.models.blocks import SobelMagnitude


class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-3) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((prediction - target).square() + self.epsilon**2).mean()


def ssim_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Differentiable local SSIM loss using an 11x11 uniform window."""
    kernel_size = 11
    padding = kernel_size // 2
    mu_x = torch.nn.functional.avg_pool2d(prediction, kernel_size, 1, padding)
    mu_y = torch.nn.functional.avg_pool2d(target, kernel_size, 1, padding)
    sigma_x = torch.nn.functional.avg_pool2d(prediction.square(), kernel_size, 1, padding) - mu_x.square()
    sigma_y = torch.nn.functional.avg_pool2d(target.square(), kernel_size, 1, padding) - mu_y.square()
    sigma_xy = torch.nn.functional.avg_pool2d(prediction * target, kernel_size, 1, padding) - mu_x * mu_y
    c1 = 0.01**2
    c2 = 0.03**2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2) + 1e-8
    )
    return 1.0 - score.mean()


class SafeRestorationLoss(nn.Module):
    """Baseline restoration, edge and uncertainty objectives."""

    def __init__(self, weights: dict[str, float]) -> None:
        super().__init__()
        self.weights = weights
        self.pixel_loss = CharbonnierLoss()
        self.sobel = SobelMagnitude()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        clean: torch.Tensor,
        weather_label: torch.Tensor | None = None,
        severity: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        restored = outputs["restored"]
        uncertainty = outputs["uncertainty"]
        pixel = self.pixel_loss(restored, clean)
        edge = torch.nn.functional.l1_loss(self.sobel(restored), self.sobel(clean))
        structural = ssim_loss(restored, clean)

        with torch.no_grad():
            error = (restored - clean).abs().mean(dim=1, keepdim=True)
            uncertainty_target = 1.0 - torch.exp(-5.0 * error)
        uncertainty_loss = torch.nn.functional.binary_cross_entropy(
            uncertainty.clamp(1e-5, 1.0 - 1e-5), uncertainty_target
        )
        uncertainty_regularization = uncertainty.mean()
        degradation_loss = torch.zeros((), device=clean.device, dtype=clean.dtype)
        if weather_label is not None and severity is not None:
            weather_label = weather_label.to(device=clean.device).reshape(-1)
            severity = severity.to(device=clean.device, dtype=clean.dtype).reshape(-1)
            degradation_target = torch.stack(
                [
                    ((weather_label == 0) | (weather_label == 2)).to(clean.dtype),
                    ((weather_label == 1) | (weather_label == 2)).to(clean.dtype),
                    severity,
                    (weather_label == 2).to(clean.dtype),
                ],
                dim=1,
            )
            degradation_loss = torch.nn.functional.binary_cross_entropy(
                outputs["degradation"].clamp(1e-5, 1.0 - 1e-5),
                degradation_target,
            )

        total = (
            self.weights.get("pixel", 1.0) * pixel
            + self.weights.get("edge", 0.0) * edge
            + self.weights.get("ssim", 0.0) * structural
            + self.weights.get("uncertainty", 0.0) * uncertainty_loss
            + self.weights.get("uncertainty_regularization", 0.0)
            * uncertainty_regularization
            + self.weights.get("degradation", 0.0) * degradation_loss
        )
        components = {
            "total": total.detach(),
            "pixel": pixel.detach(),
            "edge": edge.detach(),
            "ssim": structural.detach(),
            "uncertainty": uncertainty_loss.detach(),
            "uncertainty_mean": uncertainty_regularization.detach(),
            "degradation": degradation_loss.detach(),
        }
        return total, components
