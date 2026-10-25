import json
from pathlib import Path

from PIL import Image

from scripts.run_acdc_benchmark import METHODS, build_commands, inspect_image_coverage


def test_inspect_image_coverage_uses_coco_relative_paths(tmp_path) -> None:
    images_dir = tmp_path / "images"
    image_path = images_dir / "fog" / "val" / "scene.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(image_path)
    annotations = tmp_path / "instances.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "fog/val/scene.png"},
                    {"id": 2, "file_name": "rain/val/missing.png"},
                ]
            }
        ),
        encoding="utf-8",
    )

    coverage = inspect_image_coverage(images_dir, annotations)

    assert coverage["declared"] == 2
    assert coverage["found"] == 1
    assert coverage["missing"] == 1


def test_build_commands_covers_five_frozen_methods() -> None:
    commands = build_commands(
        python="python",
        images_dir=Path("images"),
        annotations=Path("instances.json"),
        output_dir=Path("runs/acdc"),
        config=Path("config.yaml"),
        checkpoint=Path("best.pt"),
        calibration=Path("calibration.json"),
        detector_config=Path("detector.yaml"),
        methods=list(METHODS),
        limit=10,
    )

    assert len(commands) == 10
    assert {item["method"] for item in commands if item["stage"] == "evaluate"} == set(METHODS)
    calibrated = next(
        item
        for item in commands
        if item["stage"] == "process" and item["method"] == "calibrated_fusion"
    )
    assert "--fusion-calibration" in calibrated["command"]
    assert all(
        "--annotations" in item["command"]
        for item in commands
        if item["stage"] == "process"
    )
    assert all(
        ["--limit", "10"] == item["command"][-2:]
        for item in commands
        if item["stage"] == "process"
    )
    assert all(
        ["--limit", "10"] == item["command"][-2:]
        for item in commands
        if item["stage"] == "evaluate"
    )
    report = next(item for item in commands if item["stage"] == "report")
    assert report["method"] == "five_method_summary"
