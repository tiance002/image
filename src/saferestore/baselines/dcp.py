"""Dark Channel Prior dehazing baseline.

This compact implementation is intended as an interpretable classical baseline,
not as a claim of reproducing every detail of a specific optimized release.
"""

from __future__ import annotations

import cv2
import numpy as np


def dark_channel(image: np.ndarray, patch_size: int = 15) -> np.ndarray:
    """Return the minimum RGB channel followed by local erosion."""
    minimum = image.min(axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    return cv2.erode(minimum, kernel)


def estimate_atmosphere(image: np.ndarray, dark: np.ndarray, top_fraction: float = 0.001) -> np.ndarray:
    """Estimate atmospheric light from the brightest dark-channel pixels."""
    flat_dark = dark.reshape(-1)
    flat_image = image.reshape(-1, 3)
    count = max(1, int(flat_dark.size * top_fraction))
    indices = np.argpartition(flat_dark, -count)[-count:]
    candidate_pixels = flat_image[indices]
    return candidate_pixels[np.argmax(candidate_pixels.sum(axis=1))]


def guided_filter(
    guidance: np.ndarray,
    source: np.ndarray,
    radius: int = 40,
    epsilon: float = 1e-3,
) -> np.ndarray:
    """Refine a transmission map using the grayscale guided filter."""
    kernel = (radius, radius)
    mean_i = cv2.boxFilter(guidance, -1, kernel)
    mean_p = cv2.boxFilter(source, -1, kernel)
    corr_i = cv2.boxFilter(guidance * guidance, -1, kernel)
    corr_ip = cv2.boxFilter(guidance * source, -1, kernel)
    variance_i = corr_i - mean_i * mean_i
    covariance_ip = corr_ip - mean_i * mean_p
    coefficient_a = covariance_ip / (variance_i + epsilon)
    coefficient_b = mean_p - coefficient_a * mean_i
    mean_a = cv2.boxFilter(coefficient_a, -1, kernel)
    mean_b = cv2.boxFilter(coefficient_b, -1, kernel)
    return mean_a * guidance + mean_b


def dark_channel_dehaze(
    image_rgb: np.ndarray,
    patch_size: int = 15,
    omega: float = 0.95,
    min_transmission: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Dehaze an RGB uint8 image and return restored image and transmission."""
    image = image_rgb.astype(np.float32) / 255.0
    dark = dark_channel(image, patch_size)
    atmosphere = np.clip(estimate_atmosphere(image, dark), 1e-3, 1.0)
    normalized = image / atmosphere[None, None, :]
    raw_transmission = 1.0 - omega * dark_channel(normalized, patch_size)
    guidance = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    transmission = guided_filter(guidance, raw_transmission)
    transmission = np.clip(transmission, min_transmission, 1.0)
    restored = (image - atmosphere[None, None, :]) / transmission[:, :, None] + atmosphere
    restored = np.clip(restored * 255.0, 0, 255).astype(np.uint8)
    return restored, transmission.astype(np.float32)

