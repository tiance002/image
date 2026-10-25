import numpy as np

from saferestore.baselines import dark_channel_dehaze


def test_dcp_output_contract() -> None:
    image = np.full((32, 48, 3), 180, dtype=np.uint8)
    image[8:24, 12:36] = np.array([80, 100, 120], dtype=np.uint8)
    restored, transmission = dark_channel_dehaze(image, patch_size=5)

    assert restored.shape == image.shape
    assert restored.dtype == np.uint8
    assert transmission.shape == image.shape[:2]
    assert np.all(np.isfinite(transmission))
    assert np.all((transmission >= 0.1) & (transmission <= 1.0))

