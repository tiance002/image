import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "report_d0", Path(__file__).resolve().parents[1] / "scripts/report_d0.py"
)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def test_macro_report_exposes_rare_class_loss_and_excludes_train():
    boxes = {str(c): [[c * 20, 0, c * 20 + 10, 10]] for c in report.CLASSES}
    predictions = {c: [{"box": b[0], "confidence": 0.9}] for c, b in boxes.items()}
    base = {"x": {"ground_truth": boxes, "predictions": predictions}}
    method = {"x": {"ground_truth": boxes, "predictions": {c: p for c, p in predictions.items() if c != "6"}}}
    result = report.compare(base, method, ["x"])
    assert result["method"]["map50"] == pytest.approx(6 / 7)
    assert result["method"]["map50_without_train"] == pytest.approx(1)
    assert result["method"]["per_class"]["train"]["ground_truth"] == 1
    assert result["class_redline_pass"] is False
    assert result["opr"] == pytest.approx(6 / 7)
    assert result["target_changes"]["train"] == {"kept": 0, "lost": 1, "gained": 0}


def test_pairing_fails_on_missing_image_or_changed_ground_truth():
    row = {"ground_truth": {"2": [[0, 0, 10, 10]]}, "predictions": {}}
    with pytest.raises(ValueError, match="缺少配对"):
        report.compare({"x": row}, {}, ["x"])
    other = {**row, "ground_truth": {"2": []}}
    with pytest.raises(ValueError, match="真值不一致"):
        report.compare({"x": row}, {"x": other}, ["x"])


def test_opr_does_not_count_one_prediction_as_two_targets():
    targets = [[0, 0, 10, 10], [1, 0, 11, 10]]
    predictions = [{"box": [0, 0, 10, 10], "confidence": 0.9}]
    assert report.detected_targets(targets, predictions) == {0}


def test_missing_detector_weights_fail_before_download(tmp_path):
    from saferestore.losses.perceptual import DetectorFeatureLoss

    with pytest.raises(FileNotFoundError, match="禁止自动下载"):
        DetectorFeatureLoss(str(tmp_path / "absent.pt"))


def test_duplicate_image_ids_are_rejected(tmp_path):
    import json

    path = tmp_path / "duplicates.json"
    path.write_text(json.dumps([{"image_id": "x"}, {"image_id": "x"}]))
    with pytest.raises(ValueError, match="重复文件名"):
        report.load_table(path)
