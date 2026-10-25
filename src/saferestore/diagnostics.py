"""Image-change diagnostics for explaining conservative restoration decisions."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ImageChangeDiagnostics:
    """No-reference modification measurements relative to the input image."""

    heatmap: np.ndarray
    mean_absolute_change: float
    p95_absolute_change: float
    significant_change_ratio: float
    gradient_cosine_similarity: float


def compute_image_change_diagnostics(
    input_rgb: np.ndarray,
    output_rgb: np.ndarray,
    significant_threshold: float = 20.0 / 255.0,
) -> ImageChangeDiagnostics:
    """Measure and visualize output changes without treating input as ground truth."""
    input_array = np.asarray(input_rgb)
    output_array = np.asarray(output_rgb)
    if input_array.shape != output_array.shape or input_array.ndim != 3:
        raise ValueError("Input and output must be same-shaped HWC RGB images.")
    if input_array.shape[2] != 3:
        raise ValueError("Input and output must contain three RGB channels.")

    input_float = np.clip(input_array, 0, 255).astype(np.float32) / 255.0
    output_float = np.clip(output_array, 0, 255).astype(np.float32) / 255.0
    change = np.mean(np.abs(output_float - input_float), axis=2)
    heatmap_gray = np.clip(change * 4.0 * 255.0, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_INFERNO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    input_gray = cv2.cvtColor((input_float * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    output_gray = cv2.cvtColor((output_float * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    input_x = cv2.Sobel(input_gray, cv2.CV_32F, 1, 0, ksize=3)
    input_y = cv2.Sobel(input_gray, cv2.CV_32F, 0, 1, ksize=3)
    output_x = cv2.Sobel(output_gray, cv2.CV_32F, 1, 0, ksize=3)
    output_y = cv2.Sobel(output_gray, cv2.CV_32F, 0, 1, ksize=3)
    input_gradient = cv2.magnitude(input_x, input_y).ravel()
    output_gradient = cv2.magnitude(output_x, output_y).ravel()
    denominator = float(np.linalg.norm(input_gradient) * np.linalg.norm(output_gradient))
    if denominator <= 1e-12:
        gradient_similarity = 1.0 if np.allclose(input_gradient, output_gradient) else 0.0
    else:
        gradient_similarity = float(np.dot(input_gradient, output_gradient) / denominator)

    return ImageChangeDiagnostics(
        heatmap=heatmap,
        mean_absolute_change=float(change.mean()),
        p95_absolute_change=float(np.quantile(change, 0.95)),
        significant_change_ratio=float(np.mean(change >= significant_threshold)),
        gradient_cosine_similarity=float(np.clip(gradient_similarity, 0.0, 1.0)),
    )
