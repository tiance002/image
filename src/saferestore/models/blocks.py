"""Reusable neural-network blocks."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Two convolutional layers with GroupNorm and SiLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups != 0:
            groups -= 1
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class SobelMagnitude(nn.Module):
    """Compute a differentiable grayscale Sobel gradient magnitude."""

    def __init__(self) -> None:
        super().__init__()
        kernel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        kernel_y = kernel_x.transpose(2, 3).contiguous()
        self.register_buffer("kernel_x", kernel_x)
        self.register_buffer("kernel_y", kernel_y)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        gray = (
            0.299 * image[:, 0:1]
            + 0.587 * image[:, 1:2]
            + 0.114 * image[:, 2:3]
        )
        grad_x = torch.nn.functional.conv2d(gray, self.kernel_x, padding=1)
        grad_y = torch.nn.functional.conv2d(gray, self.kernel_y, padding=1)
        return torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)


class ConditionalChannelGate(nn.Module):
    """Modulate feature channels from a degradation descriptor."""

    def __init__(self, channels: int, condition_dim: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(condition_dim, channels * 2),
            nn.Tanh(),
        )

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, bias = self.projection(condition).chunk(2, dim=1)
        scale = scale[:, :, None, None]
        bias = bias[:, :, None, None]
        return features * (1.0 + 0.25 * scale) + 0.25 * bias

