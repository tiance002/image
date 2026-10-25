import json
import zipfile
from pathlib import Path

import numpy as np

from saferestore.artifacts import (
    create_batch_result_bundle,
    create_image_result_bundle,
    read_image_result_summary,
    summarize_image_runs,
)


def test_create_image_result_bundle_saves_outputs_metrics_and_hashes(
    tmp_path: Path,
) -> None:
    images = {
        "input": np.zeros((8, 10, 3), dtype=np.uint8),
        "uncertainty": np.full((8, 10, 3), 127, dtype=np.uint8),
    }

    bundle = create_image_result_bundle(
        tmp_path,
        "saferestore_image_test",
        images,
        {"safety": {"accepted": False, "score": 0.5}},
    )

    with zipfile.ZipFile(bundle) as archive:
        assert set(archive.namelist()) == {
            "input.png",
            "uncertainty.png",
            "metrics.json",
            "manifest.json",
        }
        metrics = json.loads(archive.read("metrics.json"))
        manifest = json.loads(archive.read("manifest.json"))
    assert metrics["safety"]["accepted"] is False
    assert manifest["experiment_id"] == "saferestore_image_test"
    assert len(manifest["files"]) == 3
    assert all(len(item["sha256"]) == 64 for item in manifest["files"])


def test_create_image_result_bundle_rejects_unsafe_experiment_id(tmp_path: Path) -> None:
    try:
        create_image_result_bundle(
            tmp_path,
            "../outside",
            {"input": np.zeros((2, 2, 3), dtype=np.uint8)},
            {},
        )
    except ValueError as error:
        assert "experiment_id" in str(error)
    else:
        raise AssertionError("Expected unsafe experiment id to be rejected.")


def test_summarize_image_runs_skips_invalid_metrics(tmp_path: Path) -> None:
    valid = tmp_path / "saferestore_image_valid"
    valid.mkdir()
    (valid / "metrics.json").write_text(
        json.dumps(
            {
                "degradation": {
                    "rain_probability": 0.2,
                    "fog_probability": 0.8,
                    "severity": 0.6,
                },
                "safety": {"score": 0.85, "accepted_restoration": False},
                "detections": {"counts": {"car": 2, "person": 1}},
                "timing_ms": {"pipeline_total": 123.45},
            }
        ),
        encoding="utf-8",
    )
    broken = tmp_path / "saferestore_image_broken"
    broken.mkdir()
    (broken / "metrics.json").write_text("not json", encoding="utf-8")

    rows = summarize_image_runs(tmp_path)

    assert rows == [
        ["saferestore_image_valid", 0.2, 0.8, 0.6, 0.85, "回退输入", 3, 123.5]
    ]


def test_create_batch_bundle_contains_item_bundles_and_summary(tmp_path: Path) -> None:
    item_bundle = create_image_result_bundle(
        tmp_path,
        "saferestore_image_batch_item",
        {"input": np.zeros((4, 4, 3), dtype=np.uint8)},
        {
            "degradation": {
                "rain_probability": 0.2,
                "fog_probability": 0.8,
                "severity": 0.6,
            },
            "safety": {"score": 0.7, "accepted_restoration": False},
            "detections": {"counts": {"car": 2}},
            "timing_ms": {"pipeline_total": 120.0},
        },
    )
    row = read_image_result_summary(item_bundle, "road.jpg")

    batch_bundle, summary_csv = create_batch_result_bundle(
        tmp_path,
        "saferestore_batch_test",
        [item_bundle],
        [row],
    )

    assert row["decision"] == "input_fallback"
    assert row["detected_objects"] == 2
    assert "road.jpg" in summary_csv.read_text(encoding="utf-8-sig")
    with zipfile.ZipFile(batch_bundle) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("manifest.json"))
    assert summary_csv.name in names
    assert f"items/{item_bundle.name}" in names
    assert manifest["item_count"] == 1
