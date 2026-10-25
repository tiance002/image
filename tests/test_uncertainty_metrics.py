import numpy as np

from saferestore.metrics import (
    calibration_bins,
    expected_calibration_error,
    risk_coverage_curve,
    risk_separation_ratio,
    spearman_correlation,
)


def test_spearman_handles_order_ties_and_constants() -> None:
    values = np.array([0.1, 0.2, 0.2, 0.9])
    assert np.isclose(spearman_correlation(values, values), 1.0)
    assert np.isclose(spearman_correlation(values, -values), -1.0)
    assert spearman_correlation(np.ones(4), values) == 0.0


def test_calibration_error_is_zero_for_matching_target() -> None:
    error = np.array([0.01, 0.05, 0.1, 0.2])
    target = 1.0 - np.exp(-5.0 * error)
    rows = calibration_bins(target, error, num_bins=4)
    assert np.isclose(expected_calibration_error(rows), 0.0)


def test_risk_curve_and_separation_follow_ranked_error() -> None:
    uncertainty = np.linspace(0.0, 1.0, 100)
    error = uncertainty.copy()
    coverages, risks, area = risk_coverage_curve(uncertainty, error, num_points=10)
    assert coverages[-1] == 1.0
    assert np.all(np.diff(risks) >= 0.0)
    assert 0.0 < area < error.mean()
    assert risk_separation_ratio(uncertainty, error) > 5.0
