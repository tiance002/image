import numpy as np

from scripts.summarize_acdc_benchmark import (
    bootstrap_indices,
    bootstrap_map50,
    dataset_map50,
    group_by_weather,
    paired_bootstrap_map50,
    validate_method_payloads,
)


def record(file_name: str, prediction_box: list[float]) -> dict:
    return {
        "file_name": file_name,
        "weather": file_name.split("/")[0],
        "ground_truth": {"2": [[0, 0, 10, 10]]},
        "predictions": {"2": [{"confidence": 0.9, "box": prediction_box}]},
        "true_positives": int(prediction_box == [0, 0, 10, 10]),
        "false_positives": int(prediction_box != [0, 0, 10, 10]),
        "false_negatives": int(prediction_box != [0, 0, 10, 10]),
    }


def test_dataset_map50_and_bootstrap_distinguish_good_method() -> None:
    good = [record("a.png", [0, 0, 10, 10]), record("b.png", [0, 0, 10, 10])]
    bad = [record("a.png", [20, 20, 30, 30]), record("b.png", [20, 20, 30, 30])]
    indices = bootstrap_indices(2, 20, 7)

    assert dataset_map50(good) == 1.0
    assert dataset_map50(bad) == 0.0
    assert bootstrap_map50(good, indices)["ci_lower"] == 1.0
    difference = paired_bootstrap_map50(good, bad, indices)
    assert difference["observed_difference"] == 1.0
    assert difference["ci_lower"] == 1.0


def test_validate_method_payloads_rejects_image_mismatch() -> None:
    base = {
        "annotation_sha256": "same",
        "per_image": [{"file_name": "a.png"}],
    }
    payloads = {
        "weather_input": base,
        "dcp": {**base, "per_image": [{"file_name": "b.png"}]},
    }

    try:
        validate_method_payloads(payloads)
    except ValueError as error:
        assert "Image order differs" in str(error)
    else:
        raise AssertionError("Expected image mismatch to fail")


def test_validate_method_payloads_rejects_detector_config_mismatch() -> None:
    base = {
        "annotation_sha256": "annotation",
        "detector_config_sha256": "config-a",
        "iou_threshold": 0.5,
        "confidence_threshold": 0.25,
        "per_image": [{"file_name": "a.png"}],
    }
    payloads = {
        "weather_input": base,
        "dcp": {**base, "detector_config_sha256": "config-b"},
    }

    try:
        validate_method_payloads(payloads)
    except ValueError as error:
        assert "detector_config_sha256" in str(error)
    else:
        raise AssertionError("Expected detector config mismatch to fail")


def test_bootstrap_indices_are_deterministic() -> None:
    assert np.array_equal(bootstrap_indices(3, 4, 42), bootstrap_indices(3, 4, 42))


def test_group_by_weather_preserves_record_order() -> None:
    records = [
        record("rain/val/b.png", [0, 0, 10, 10]),
        record("fog/val/a.png", [0, 0, 10, 10]),
        record("rain/val/c.png", [0, 0, 10, 10]),
    ]

    grouped = group_by_weather(records)

    assert list(grouped) == ["fog", "rain"]
    assert [item["file_name"] for item in grouped["rain"]] == [
        "rain/val/b.png",
        "rain/val/c.png",
    ]
