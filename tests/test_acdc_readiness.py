import importlib.util
import json
import sys
import zipfile
from pathlib import Path


def load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_acdc_benchmark.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("prepare_acdc_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_annotations(path: Path, file_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": index, "file_name": name}
                    for index, name in enumerate(file_names)
                ]
            }
        ),
        encoding="utf-8",
    )


def test_find_and_inspect_valid_archive(tmp_path):
    module = load_script_module()
    archive = tmp_path / module.ARCHIVE_NAME
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("rgb_anon/fog/val/sample.png", b"image")

    found = module.find_archive(None, [tmp_path])
    inspected = module.inspect_archive(found)

    assert found == archive
    assert inspected["valid_zip"] is True
    assert inspected["image_member_count"] == 1


def test_report_waits_when_rgb_archive_and_images_are_missing(tmp_path, monkeypatch):
    module = load_script_module()
    annotations = tmp_path / "annotations.json"
    write_annotations(annotations, ["fog/val/a.png"])
    prerequisites = []
    for index in range(4):
        path = tmp_path / f"required-{index}.bin"
        path.write_bytes(b"ok")
        prerequisites.append(path)
    monkeypatch.setattr(module, "REQUIRED_BENCHMARK_FILES", tuple(prerequisites))

    report = module.build_report(
        archive_path=None,
        search_dirs=[tmp_path],
        images_dir=tmp_path / "images",
        annotations=annotations,
    )

    assert report["status"] == "waiting_for_rgb_archive"
    assert report["coverage"] is None


def test_report_is_ready_only_with_complete_declared_coverage(tmp_path, monkeypatch):
    module = load_script_module()
    annotations = tmp_path / "annotations.json"
    names = ["fog/val/a.png", "rain/val/b.png"]
    write_annotations(annotations, names)
    images_dir = tmp_path / "images"
    for name in names:
        path = images_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")
    prerequisites = []
    for index in range(4):
        path = tmp_path / f"required-{index}.bin"
        path.write_bytes(b"ok")
        prerequisites.append(path)
    monkeypatch.setattr(module, "REQUIRED_BENCHMARK_FILES", tuple(prerequisites))

    report = module.build_report(
        archive_path=None,
        search_dirs=[tmp_path],
        images_dir=images_dir,
        annotations=annotations,
    )

    assert report["status"] == "ready_for_benchmark"
    assert report["coverage"] == {
        "declared": 2,
        "found": 2,
        "missing": 0,
        "missing_examples": [],
    }
