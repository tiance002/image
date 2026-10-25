"""Deterministic statistical summaries shared by experiment reports."""

from __future__ import annotations

import numpy as np


def bootstrap_mean_interval(
    values: list[float] | np.ndarray,
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 42,
) -> dict[str, float | int]:
    """Return a percentile bootstrap confidence interval for the sample mean."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a non-empty one-dimensional sequence")
    if not np.isfinite(array).all():
        raise ValueError("values must all be finite")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(resamples, array.size))
    bootstrap_means = array[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "sample_count": int(array.size),
        "mean": float(array.mean()),
        "ci_lower": float(np.quantile(bootstrap_means, alpha)),
        "ci_upper": float(np.quantile(bootstrap_means, 1.0 - alpha)),
        "confidence": confidence,
        "resamples": resamples,
        "seed": seed,
    }


def paired_bootstrap_difference(
    candidate: list[float] | np.ndarray,
    reference: list[float] | np.ndarray,
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 42,
) -> dict[str, float | int]:
    """Bootstrap the paired mean difference ``candidate - reference``."""
    candidate_array = np.asarray(candidate, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    if candidate_array.shape != reference_array.shape:
        raise ValueError("candidate and reference must have the same shape")
    return bootstrap_mean_interval(
        candidate_array - reference_array,
        confidence=confidence,
        resamples=resamples,
        seed=seed,
    )
