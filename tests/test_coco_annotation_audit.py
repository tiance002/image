import json

from PIL import Image

from scripts.audit_coco_annotations import audit_annotation_file, audit_directory


def make_coco(path, *, invalid: bool = False) -> None:
    path.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "fog/val/scene.png", "width": 64, "height": 48}],
                "categories": [{"id": 2, "name": "car"}],
                "annotations": [
                    {
                        "id": 3,
                        "image_id": 99 if invalid else 1,
                        "category_id": 2,
                        "bbox": [1, 2, -1 if invalid else 20, 10],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_audit_annotation_file_checks_images_and_structure(tmp_path) -> None:
    annotation_path = tmp_path / "instances.json"
    make_coco(annotation_path)
    images_dir = tmp_path / "images"
    image_path = images_dir / "fog" / "val" / "scene.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (64, 48)).save(image_path)

    report = audit_annotation_file(annotation_path, images_dir)

    assert report["structural_errors"] == 0
    assert report["images_found"] == 1
    assert report["weather_counts"] == {"fog": 1}
    assert report["split_counts"] == {"val": 1}
    assert report["category_counts"] == {"car": 1}


def test_audit_directory_finds_duplicate_payload_and_invalid_box(tmp_path) -> None:
    first = tmp_path / "a.json"
    second = tmp_path / "nested" / "b.json"
    second.parent.mkdir()
    make_coco(first, invalid=True)
    second.write_bytes(first.read_bytes())

    report = audit_directory(tmp_path)

    assert report["annotation_file_count"] == 2
    assert report["total_structural_errors"] == 4
    assert len(report["duplicate_payload_groups"]) == 1
