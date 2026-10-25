"""验证合成RGB、BGR标定、OpenCV编码与训练RGB加载之间的颜色契约。"""

import random

import cv2
import numpy as np
import pytest

from scripts import build_real_scene_pairs as scene
from scripts import build_real_weather_pairs as weather
from scripts.train_restorer import imread_rgb


@pytest.mark.parametrize("size", [None, (48, 32)])
def test_scene_jpeg_roundtrip_preserves_rgb_and_training_loader(tmp_path, size):
    rgb = np.zeros((64, 96, 3), np.uint8)
    rgb[:, :48] = [230, 40, 10]
    rgb[:, 48:] = [10, 40, 230]
    path = tmp_path / "中文目标.jpg"
    assert scene.imwrite(path, rgb, size=size)
    loaded = imread_rgb(path) * 255
    expected_size = (64, 96) if size is None else (32, 48)
    assert loaded.shape[:2] == expected_size
    assert np.abs(loaded[8, 8] - np.array([230, 40, 10])).max() < 5
    assert np.abs(loaded[8, -8] - np.array([10, 40, 230])).max() < 5


@pytest.mark.parametrize("key", ["air", "mean_c", "std_c"])
def test_legacy_bgr_calibration_converted_once_without_mutation(key):
    values = [0.51, 0.70, 0.93]
    result = weather.calibration_rgb({key: values}, key)
    np.testing.assert_allclose(result, [0.93, 0.70, 0.51])
    assert values == [0.51, 0.70, 0.93]


def test_fog_uses_rgb_airlight_from_actual_bgr_calibration_contract():
    clear_rgb = np.zeros((16, 24, 3), np.float32)
    result = weather.synth_fog(clear_rgb, {"air": [0.51, 0.70, 0.93], "strength": 0.4}, random.Random(42))
    assert (result[:, :, 0] > result[:, :, 1]).all()
    assert (result[:, :, 1] > result[:, :, 2]).all()


def test_rain_reinhard_target_uses_rgb_statistics(monkeypatch):
    recorded = {}

    def transfer(image, mean, std, weight):
        recorded["mean"] = mean
        recorded["std"] = std
        return image

    monkeypatch.setattr(weather, "reinhard_transfer", transfer)
    monkeypatch.setattr(weather, "wet_road", lambda image, rng: image)
    monkeypatch.setattr(weather, "lens_droplets", lambda image, rng: image)
    monkeypatch.setattr(weather, "render_rain_layer", lambda h, w, rng: np.zeros((h, w), np.float32))
    image = np.full((16, 24, 3), 0.5, np.float32)
    weather.synth_rain(image, {"mean_c": [0.2, 0.4, 0.8], "std_c": [0.1, 0.2, 0.3]}, random.Random(42))
    # 同一个标量抖动不改变各通道比例。
    np.testing.assert_allclose(recorded["mean"] / recorded["mean"][2], [4, 2, 1])
    np.testing.assert_allclose(recorded["std"] / recorded["std"][2], [3, 2, 1])


def test_weather_writer_bgr_contract_and_rgb_loader(tmp_path):
    path = tmp_path / "weather.jpg"
    bgr = np.full((32, 32, 3), [10, 40, 230], np.uint8)
    assert weather.imwrite(path, bgr)
    np.testing.assert_allclose(imread_rgb(path)[8, 8] * 255, [230, 40, 10], atol=3)


def test_weather_builder_refuses_existing_split_before_any_write(tmp_path, monkeypatch):
    monkeypatch.setattr(weather, "OUT", tmp_path)
    (tmp_path / "clean/train").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        weather.build("train", 3, 42)


def test_calibration_builder_preserves_historical_file(tmp_path, monkeypatch):
    path = tmp_path / "calibration.json"
    path.write_text("history", encoding="utf-8")
    monkeypatch.setattr(weather, "CALIB", path)
    with pytest.raises(FileExistsError):
        weather.build_calibration()
    assert path.read_text(encoding="utf-8") == "history"
