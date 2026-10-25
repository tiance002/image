import zipfile

import pytest

from scripts.import_public_dataset import import_archive


def test_import_archive_records_inventory_and_hash(tmp_path) -> None:
    archive_path = tmp_path / "sample.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("images/sample.jpg", b"not-a-real-image")
        archive.writestr("annotations/sample.xml", "<annotation />")
    destination = tmp_path / "dataset"

    result = import_archive("rtts_roboflow", archive_path, destination)

    assert result["archive_sha256"]
    assert result["inventory"]["image_count"] == 1
    assert result["inventory"]["xml_annotation_count"] == 1
    assert (destination / "IMPORT_RECORD.json").is_file()


def test_import_archive_rejects_path_traversal(tmp_path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.txt", "unsafe")

    with pytest.raises(ValueError, match="Unsafe archive member"):
        import_archive("acdc", archive_path, tmp_path / "dataset")


def test_import_archive_refuses_nonempty_destination(tmp_path) -> None:
    archive_path = tmp_path / "sample.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("sample.json", "{}")
    destination = tmp_path / "dataset"
    destination.mkdir()
    (destination / "existing.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        import_archive("acdc", archive_path, destination)
