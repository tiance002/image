import json

from PIL import Image

from scripts.evaluate_coco_detection import (
    build_per_image_records,
    load_dataset,
    match_image_records,
    xywh_to_xyxy,
)


def test_xywh_to_xyxy() -> None:
    assert xywh_to_xyxy([10, 20, 30, 40]).tolist() == [10, 20, 40, 60]


def test_load_dataset_filters_unsupported_and_missing(tmp_path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (64, 48)).save(images_dir / "present.png")
    annotation_path = tmp_path / "instances.json"
    annotation_path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "present.png", "width": 64, "height": 48},
                    {"id": 2, "file_name": "missing.png", "width": 64, "height": 48},
                ],
                "categories": [
                    {"id": 10, "name": "car"},
                    {"id": 11, "name": "rider"},
                ],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 10, "bbox": [1, 2, 20, 10]},
                    {"id": 2, "image_id": 1, "category_id": 11, "bbox": [2, 3, 10, 8]},
                ],
            }
        ),
        encoding="utf-8",
    )

    images, _, ground_truth, inspection = load_dataset(images_dir, annotation_path)

    assert len(images) == 1
    assert len(ground_truth[2][1]) == 1
    assert inspection["class_counts"] == {"car": 1}
    assert inspection["ignored_annotations"] == 1
    assert len(inspection["missing_images"]) == 1


def test_match_image_records_counts_true_false_and_missed() -> None:
    ground_truth = {"2": [[0, 0, 10, 10], [20, 20, 30, 30]]}
    predictions = {
        "2": [
            {"confidence": 0.9, "box": [0, 0, 10, 10]},
            {"confidence": 0.8, "box": [40, 40, 50, 50]},
        ]
    }

    metrics = match_image_records(ground_truth, predictions, 0.5)

    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["f1"] == 0.5


def test_build_per_image_records_serializes_boxes() -> None:
    images = [{"id": 7, "file_name": "fog/val/scene.png"}]
    predictions = {
        2: [
            {
                "image_id": 7,
                "confidence": 0.9,
                "box": xywh_to_xyxy([0, 0, 10, 10]),
            }
        ]
    }
    ground_truth = {2: {7: [xywh_to_xyxy([0, 0, 10, 10])]}}

    records = build_per_image_records(images, predictions, ground_truth, 0.5)

    assert records[0]["weather"] == "fog"
    assert records[0]["true_positives"] == 1
    assert records[0]["ground_truth"]["2"] == [[0.0, 0.0, 10.0, 10.0]]
