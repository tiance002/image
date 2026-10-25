import numpy as np

from saferestore.diagnostics import compute_image_change_diagnostics


def test_change_diagnostics_reports_no_change_for_identical_images() -> None:
    image = np.full((24, 32, 3), 120, dtype=np.uint8)

    result = compute_image_change_diagnostics(image, image.copy())

    assert result.heatmap.shape == image.shape
    assert result.mean_absolute_change == 0.0
    assert result.p95_absolute_change == 0.0
    assert result.significant_change_ratio == 0.0
    assert result.gradient_cosine_similarity == 1.0


def test_change_diagnostics_detects_local_structural_change() -> None:
    input_image = np.zeros((32, 32, 3), dtype=np.uint8)
    input_image[:, 8:10] = 255
    output_image = input_image.copy()
    output_image[:, 20:24] = 255

    result = compute_image_change_diagnostics(input_image, output_image)

    assert result.mean_absolute_change > 0.1
    assert result.p95_absolute_change == 1.0
    assert result.significant_change_ratio > 0.1
    assert 0.0 < result.gradient_cosine_similarity < 1.0
