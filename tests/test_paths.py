import json

import pytest
from PIL import Image

from saferestore.paths import find_coco_images, find_images, mirrored_output_path


def test_find_images_is_recursive_and_deterministic(tmp_path) -> None:
    root = tmp_path / "images"
    (root / "rain" / "val").mkdir(parents=True)
    (root / "fog" / "val").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(root / "rain" / "val" / "b.png")
    Image.new("RGB", (8, 8)).save(root / "fog" / "val" / "a.jpg")
    (root / "notes.txt").write_text("not an image", encoding="utf-8")

    paths = find_images(root)

    assert [path.relative_to(root).as_posix() for path in paths] == [
        "fog/val/a.jpg",
        "rain/val/b.png",
    ]


def test_mirrored_output_path_preserves_hierarchy(tmp_path) -> None:
    input_root = tmp_path / "input"
    input_path = input_root / "fog" / "train" / "scene.jpg"
    output_root = tmp_path / "output"

    assert mirrored_output_path(input_path, input_root, output_root) == (
        output_root / "fog" / "train" / "scene.jpg"
    )
    assert mirrored_output_path(input_path, input_root, output_root, suffix=".png") == (
        output_root / "fog" / "train" / "scene.png"
    )
    assert mirrored_output_path(input_path.resolve(), input_root, output_root) == (
        output_root / "fog" / "train" / "scene.jpg"
    )


def test_find_coco_images_preserves_declared_subset_and_order(tmp_path) -> None:
    root = tmp_path / "images"
    for relative in ("fog/val/a.png", "rain/val/b.png", "snow/test/ignored.png"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8)).save(path)
    annotations = tmp_path / "instances.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 2, "file_name": "rain/val/b.png"},
                    {"id": 1, "file_name": "fog/val/a.png"},
                ]
            }
        ),
        encoding="utf-8",
    )

    paths = find_coco_images(root, annotations)

    assert [path.relative_to(root).as_posix() for path in paths] == [
        "rain/val/b.png",
        "fog/val/a.png",
    ]


def test_find_coco_images_rejects_missing_or_unsafe_paths(tmp_path) -> None:
    root = tmp_path / "images"
    root.mkdir()
    annotations = tmp_path / "instances.json"
    annotations.write_text(
        json.dumps({"images": [{"id": 1, "file_name": "rain/val/missing.png"}]}),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="coverage is incomplete"):
        find_coco_images(root, annotations)

    annotations.write_text(
        json.dumps({"images": [{"id": 1, "file_name": "../escape.png"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsafe COCO image path"):
        find_coco_images(root, annotations)
