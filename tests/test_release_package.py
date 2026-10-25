import json
import zipfile
from pathlib import Path

from scripts.create_release_package import create_package


def test_release_package_uses_portable_root_and_excludes_caches(tmp_path: Path) -> None:
    project = tmp_path / "unicode-project"
    (project / "data" / "raw").mkdir(parents=True)
    (project / "data" / "archives").mkdir(parents=True)
    (project / "src" / "__pycache__").mkdir(parents=True)
    (project / "runs").mkdir()
    (project / "app.py").write_text("print('ok')", encoding="utf-8")
    (project / "README.md").write_text("readme", encoding="utf-8")
    (project / "PROJECT_DELIVERY_GUIDE_ZH.md").write_text("guide", encoding="utf-8")
    (project / "data" / "raw" / "sample.jpg").write_bytes(b"image")
    (project / "data" / "archives" / "duplicate.zip").write_bytes(b"archive")
    (project / "src" / "module.py").write_text("VALUE = 1", encoding="utf-8")
    (project / "src" / "__pycache__" / "module.pyc").write_bytes(b"cache")
    (project / "runs" / "model.pt").write_bytes(b"weights")

    package = create_package(project, tmp_path / "release", "2026-10-12")

    with zipfile.ZipFile(package) as archive:
        names = set(archive.namelist())
        assert archive.testzip() is None
    assert "SafeRestore-RF/app.py" in names
    assert "SafeRestore-RF/PROJECT_DELIVERY_GUIDE_ZH.md" in names
    assert "SafeRestore-RF/data/raw/sample.jpg" in names
    assert not any("archives" in name for name in names)
    assert not any("__pycache__" in name for name in names)
    metadata = json.loads(package.with_suffix(".package.json").read_text(encoding="utf-8"))
    assert metadata["project_folder"] == "SafeRestore-RF"
    assert metadata["file_count"] == len(names)
