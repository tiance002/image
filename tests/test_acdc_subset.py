from scripts.build_acdc_subset import build_subset, image_condition


def test_build_subset_filters_images_and_annotations() -> None:
    payload = {
        "info": {"name": "fixture"},
        "categories": [{"id": 1, "name": "car"}],
        "images": [
            {"id": 1, "file_name": "fog/train/a.png"},
            {"id": 2, "file_name": "rain/train/b.png"},
            {"id": 3, "file_name": "snow/train/c.png"},
        ],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 1, "bbox": [0, 0, 1, 1]},
            {"id": 11, "image_id": 2, "category_id": 1, "bbox": [0, 0, 1, 1]},
            {"id": 12, "image_id": 3, "category_id": 1, "bbox": [0, 0, 1, 1]},
        ],
    }

    subset, summary = build_subset(payload, {"fog", "rain"})

    assert [image["id"] for image in subset["images"]] == [1, 2]
    assert [annotation["id"] for annotation in subset["annotations"]] == [10, 11]
    assert summary["condition_counts"] == {"fog": 1, "rain": 1}
    assert subset["info"] == payload["info"]


def test_image_condition_accepts_windows_separators() -> None:
    assert image_condition({"file_name": "rain\\val\\scene.png"}) == "rain"
