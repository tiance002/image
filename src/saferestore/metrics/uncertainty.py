"""Dependency-light uncertainty calibration and selective-risk metrics."""

from __future__ import annotations

import numpy as np


def _paired_flatten(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first_flat = np.asarray(first, dtype=np.float64).reshape(-1)
    second_flat = np.asarray(second, dtype=np.float64).reshape(-1)
    if first_flat.shape != second_flat.shape:
        raise ValueError("Metric inputs must contain the same number of values.")
    if first_flat.size == 0:
        raise ValueError("Metric inputs must not be empty.")
    valid = np.isfinite(first_flat) & np.isfinite(second_flat)
    if not np.any(valid):
        raise ValueError("Metric inputs contain no finite paired values.")
    return first_flat[valid], second_flat[valid]


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks, including correct handling of ties."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def spearman_correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Compute Spearman correlation without requiring SciPy.

    A constant input has no ranking information and returns ``0.0`` instead of
    propagating a NaN into experiment summaries.
    """
    first_flat, second_flat = _paired_flatten(first, second)
    first_rank = _average_ranks(first_flat)
    second_rank = _average_ranks(second_flat)
    first_centered = first_rank - first_rank.mean()
    second_centered = second_rank - second_rank.mean()
    denominator = np.sqrt(np.sum(first_centered**2) * np.sum(second_centered**2))
    if denominator <= np.finfo(np.float64).eps:
        return 0.0
    return float(np.sum(first_centered * second_centered) / denominator)


def calibration_bins(
    uncertainty: np.ndarray,
    absolute_error: np.ndarray,
    *,
    num_bins: int = 10,
    error_scale: float = 5.0,
) -> list[dict[str, float | int]]:
    """Bin uncertainty against the training target derived from absolute error."""
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2.")
    uncertainty_flat, error_flat = _paired_flatten(uncertainty, absolute_error)
    uncertainty_flat = np.clip(uncertainty_flat, 0.0, 1.0)
    target = 1.0 - np.exp(-error_scale * np.clip(error_flat, 0.0, None))
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    indices = np.minimum(np.digitize(uncertainty_flat, edges[1:-1]), num_bins - 1)
    rows: list[dict[str, float | int]] = []
    for index in range(num_bins):
        mask = indices == index
        count = int(mask.sum())
        mean_uncertainty = float(uncertainty_flat[mask].mean()) if count else 0.0
        mean_target = float(target[mask].mean()) if count else 0.0
        rows.append(
            {
                "bin": index,
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": count,
                "mean_uncertainty": mean_uncertainty,
                "mean_target": mean_target,
                "absolute_gap": abs(mean_uncertainty - mean_target) if count else 0.0,
            }
        )
    return rows


def expected_calibration_error(rows: list[dict[str, float | int]]) -> float:
    """Return count-weighted calibration gap for precomputed bins."""
    total = sum(int(row["count"]) for row in rows)
    if total == 0:
        raise ValueError("Calibration bins contain no samples.")
    return float(
        sum(int(row["count"]) * float(row["absolute_gap"]) for row in rows) / total
    )


def risk_coverage_curve(
    uncertainty: np.ndarray,
    absolute_error: np.ndarray,
    *,
    num_points: int = 20,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute mean error when progressively retaining low-uncertainty pixels.

    Lower AURC is better. The final point always represents 100% coverage.
    """
    if num_points < 2:
        raise ValueError("num_points must be at least 2.")
    uncertainty_flat, error_flat = _paired_flatten(uncertainty, absolute_error)
    order = np.argsort(uncertainty_flat, kind="mergesort")
    sorted_error = np.clip(error_flat[order], 0.0, None)
    coverages = np.linspace(1.0 / num_points, 1.0, num_points)
    counts = np.maximum(1, np.ceil(coverages * sorted_error.size).astype(int))
    cumulative_error = np.cumsum(sorted_error, dtype=np.float64)
    risks = cumulative_error[counts - 1] / counts
    area = float(np.trapezoid(risks, coverages))
    return coverages, risks, area


def risk_separation_ratio(
    uncertainty: np.ndarray,
    absolute_error: np.ndarray,
    *,
    fraction: float = 0.2,
) -> float:
    """Compare mean error in the most- and least-uncertain pixel subsets."""
    if not 0.0 < fraction <= 0.5:
        raise ValueError("fraction must be in (0, 0.5].")
    uncertainty_flat, error_flat = _paired_flatten(uncertainty, absolute_error)
    order = np.argsort(uncertainty_flat, kind="mergesort")
    count = max(1, int(np.ceil(fraction * order.size)))
    low_error = float(np.mean(error_flat[order[:count]]))
    high_error = float(np.mean(error_flat[order[-count:]]))
    if low_error <= np.finfo(np.float64).eps:
        return float("inf") if high_error > 0 else 1.0
    return high_error / low_error
