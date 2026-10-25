"""Small, dependency-light detection metric helpers."""

from __future__ import annotations

import numpy as np


def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Return IoU for two ``xyxy`` boxes."""
    left = max(float(box_a[0]), float(box_b[0]))
    top = max(float(box_a[1]), float(box_b[1]))
    right = min(float(box_a[2]), float(box_b[2]))
    bottom = min(float(box_a[3]), float(box_b[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def average_precision_101(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute COCO-style 101-point interpolated AP for one IoU threshold."""
    if recall.size == 0:
        return 0.0
    precision_envelope = np.maximum.accumulate(precision[::-1])[::-1]
    samples = np.linspace(0.0, 1.0, 101)
    values = [
        float(precision_envelope[recall >= threshold].max())
        if np.any(recall >= threshold)
        else 0.0
        for threshold in samples
    ]
    return float(np.mean(values))

