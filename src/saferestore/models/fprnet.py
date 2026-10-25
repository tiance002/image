"""Fidelity-preserving rain/fog restoration baseline."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import ConditionalChannelGate, ConvBlock, SobelMagnitude


class DegradationEncoder(nn.Module):
    """Predict rain/fog presence probabilities and relative intensities."""

    def __init__(self, condition_dim: int = 4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, condition_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.features(image).flatten(1)
        return torch.sigmoid(self.head(features))


class FidelityPreservingRestorer(nn.Module):
    """Compact dual-branch U-Net with degradation conditioning and uncertainty."""

    def __init__(
        self,
        base_channels: int = 32,
        degradation_dim: int = 4,
        uncertainty_floor: float = 0.02,
        use_degradation_gate: bool = False,
        minimum_restoration_strength: float = 0.2,
    ) -> None:
        super().__init__()
        if not 0.0 <= minimum_restoration_strength <= 1.0:
            raise ValueError("minimum_restoration_strength must be in [0, 1].")
        c = base_channels
        self.uncertainty_floor = uncertainty_floor
        self.use_degradation_gate = use_degradation_gate
        self.minimum_restoration_strength = minimum_restoration_strength
        self.degradation = DegradationEncoder(degradation_dim)
        self.sobel = SobelMagnitude()

        self.enc1 = ConvBlock(3, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.bottleneck = ConvBlock(c * 2, c * 4)
        self.pool = nn.MaxPool2d(2)
        self.condition_gate = ConditionalChannelGate(c * 4, degradation_dim)

        self.structure = nn.Sequential(
            ConvBlock(1, c),
            nn.Conv2d(c, c, 3, padding=1),
        )
        self.dec2 = ConvBlock(c * 4 + c * 2, c * 2)
        self.dec1 = ConvBlock(c * 2 + c + c, c)
        self.restoration_head = nn.Conv2d(c, 3, 3, padding=1)
        self.uncertainty_head = nn.Sequential(
            ConvBlock(c + 3, c),
            nn.Conv2d(c, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, weather: torch.Tensor) -> dict[str, torch.Tensor]:
        degradation = self.degradation(weather)
        enc1 = self.enc1(weather)
        enc2 = self.enc2(self.pool(enc1))
        bottleneck = self.bottleneck(self.pool(enc2))
        bottleneck = self.condition_gate(bottleneck, degradation)

        dec2 = F.interpolate(bottleneck, size=enc2.shape[-2:], mode="bilinear", align_corners=False)
        dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))

        dec1 = F.interpolate(dec2, size=enc1.shape[-2:], mode="bilinear", align_corners=False)
        structure = self.structure(self.sobel(weather))
        dec1 = self.dec1(torch.cat([dec1, enc1, structure], dim=1))

        residual = torch.tanh(self.restoration_head(dec1)) * 0.5
        if self.use_degradation_gate:
            predicted_severity = degradation[:, 2:3, None, None]
            restoration_strength = self.minimum_restoration_strength + (
                1.0 - self.minimum_restoration_strength
            ) * predicted_severity
            residual = residual * restoration_strength
        else:
            restoration_strength = torch.ones(
                weather.shape[0], 1, 1, 1, device=weather.device, dtype=weather.dtype
            )
        restored = torch.clamp(weather + residual, 0.0, 1.0)
        uncertainty = self.uncertainty_head(torch.cat([dec1, (restored - weather).abs()], dim=1))
        uncertainty = self.uncertainty_floor + (1.0 - self.uncertainty_floor) * uncertainty
        conservative = (1.0 - uncertainty) * restored + uncertainty * weather
        return {
            "restored": restored,
            "uncertainty": uncertainty,
            "conservative": conservative,
            "degradation": degradation,
            "restoration_strength": restoration_strength,
        }
