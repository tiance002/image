import numpy as np
import pytest

from saferestore.statistics import bootstrap_mean_interval, paired_bootstrap_difference


def test_bootstrap_constant_interval_is_exact() -> None:
    result = bootstrap_mean_interval([2.5, 2.5, 2.5], resamples=100, seed=7)
    assert result["mean"] == pytest.approx(2.5)
    assert result["ci_lower"] == pytest.approx(2.5)
    assert result["ci_upper"] == pytest.approx(2.5)


def test_paired_bootstrap_difference_is_deterministic() -> None:
    candidate = np.asarray([2.0, 4.0, 6.0, 8.0])
    reference = np.asarray([1.0, 2.0, 3.0, 4.0])
    first = paired_bootstrap_difference(candidate, reference, resamples=500, seed=11)
    second = paired_bootstrap_difference(candidate, reference, resamples=500, seed=11)
    assert first == second
    assert first["mean"] == pytest.approx(2.5)
    assert first["ci_lower"] > 0.0


def test_bootstrap_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        bootstrap_mean_interval([1.0, float("nan")])
