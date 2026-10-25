"""Create a portable ZIP64 delivery package with datasets and checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ARCHIVE_ROOT = Path("SafeRestore-RF")
INCLUDE_PATHS = (
    ".vscode",
    "configs",
    "data/derived",
    "data/factorial",
    "data/manifests",
    "data/processed",
    "data/raw",
    "docs",
    "runs",
    "scripts",
    "src",
    "tests",
    ".gitignore",
    "app.py",
    "pyproject.toml",
    "README.md",
    "yolov8n.pt",
    "PROJECT_DELIVERY_GUIDE_ZH.md",
)
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".cache", ".tmp"}


def iter_package_files(project_root: Path) -> list[Path]:
    """Return the sorted, explicit delivery whitelist without caches."""
    files: list[Path] = []
    for relative in INCLUDE_PATHS:
        source = project_root / relative
        if not source.exists():
            continue
        candidates = [source] if source.is_file() else source.rglob("*")
        for path in candidates:
            if not path.is_file():
                continue
            project_relative = path.relative_to(project_root)
            if any(part in EXCLUDED_PARTS for part in project_relative.parts):
                continue
            if path.suffix.lower() == ".pyc":
                continue
            files.append(path)
    return sorted(set(files), key=lambda path: path.relative_to(project_root).as_posix())


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a large file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def create_package(project_root: Path, output_directory: Path, date_stamp: str) -> Path:
    """Create the complete package and sidecar checksum/metadata files."""
    output_directory.mkdir(parents=True, exist_ok=True)
    package_base = f"SafeRestore-RF_full_project_with_datasets_{date_stamp}"
    package_path = output_directory / f"{package_base}.zip"
    checksum_path = output_directory / f"{package_base}.sha256.txt"
    metadata_path = output_directory / f"{package_base}.package.json"
    if package_path.exists():
        raise FileExistsError(f"Package already exists and will not be overwritten: {package_path}")

    files = iter_package_files(project_root)
    total_bytes = sum(path.stat().st_size for path in files)
    written_bytes = 0
    print(f"Packaging {len(files)} files ({total_bytes / 1024**3:.3f} GiB)...", flush=True)
    try:
        with ZipFile(
            package_path,
            "w",
            compression=ZIP_DEFLATED,
            compresslevel=1,
            allowZip64=True,
        ) as archive:
            for index, path in enumerate(files, start=1):
                relative = path.relative_to(project_root)
                archive_name = (ARCHIVE_ROOT / relative).as_posix()
                archive.write(path, arcname=archive_name)
                written_bytes += path.stat().st_size
                if index % 250 == 0 or index == len(files):
                    percent = 100.0 * written_bytes / max(total_bytes, 1)
                    print(
                        f"[{index}/{len(files)}] source bytes processed: {percent:.1f}%",
                        flush=True,
                    )
    except Exception:
        package_path.unlink(missing_ok=True)
        raise

    print("Computing SHA-256...", flush=True)
    digest = sha256_file(package_path)
    checksum_path.write_text(f"{digest}  {package_path.name}\n", encoding="utf-8")
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "package": package_path.name,
        "package_bytes": package_path.stat().st_size,
        "sha256": digest,
        "project_folder": ARCHIVE_ROOT.as_posix(),
        "file_count": len(files),
        "source_bytes": total_bytes,
        "includes_extracted_datasets": True,
        "excludes_duplicate_download_archives": True,
        "excluded_path": f"{ARCHIVE_ROOT.as_posix()}/data/archives",
        "license_notice": (
            "ACDC is restricted to non-commercial use under its official license."
        ),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "package": str(package_path),
                "size_gib": round(package_path.stat().st_size / 1024**3, 3),
                "sha256": digest,
                "checksum": str(checksum_path),
                "metadata": str(metadata_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return package_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=datetime.now().astimezone().strftime("%Y-%m-%d"))
    parser.add_argument("--output-directory", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output_directory = args.output_directory or project_root.parent / "release_packages"
    create_package(project_root, output_directory.resolve(), args.date)


if __name__ == "__main__":
    main()
