"""Safely import a manually downloaded public dataset archive with provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_METADATA = {
    "acdc": {
        "destination": ROOT / "data" / "raw" / "ACDC",
        "source": "https://acdc.vision.ee.ethz.ch/download",
        "license_note": "Non-commercial use after accepting the official ACDC license.",
    },
    "rtts_roboflow": {
        "destination": ROOT / "data" / "raw" / "RTTS_roboflow",
        "source": "https://universe.roboflow.com/sophie-aj5fi/rtts-vf9q6-zxh7o",
        "license_note": "Roboflow project declares CC BY 4.0; retain upstream RTTS citation.",
    },
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_zip_members(archive: zipfile.ZipFile, destination: Path) -> list[zipfile.ZipInfo]:
    destination_resolved = destination.resolve()
    members = archive.infolist()
    if not members:
        raise ValueError("Archive is empty.")
    for member in members:
        target = (destination / member.filename).resolve()
        try:
            target.relative_to(destination_resolved)
        except ValueError as error:
            raise ValueError(f"Unsafe archive member: {member.filename}") from error
    return members


def inventory(destination: Path) -> dict[str, object]:
    files = [path for path in destination.rglob("*") if path.is_file()]
    suffix_counts = Counter(path.suffix.lower() or "<no_suffix>" for path in files)
    image_count = sum(count for suffix, count in suffix_counts.items() if suffix in IMAGE_SUFFIXES)
    xml_files = sorted(path for path in files if path.suffix.lower() == ".xml")
    json_files = sorted(path for path in files if path.suffix.lower() == ".json")
    return {
        "file_count": len(files),
        "image_count": image_count,
        "xml_annotation_count": len(xml_files),
        "json_file_count": len(json_files),
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "candidate_annotation_paths": [
            str(path.relative_to(destination).as_posix())
            for path in [*xml_files[:20], *json_files[:20]]
        ],
    }


def import_archive(dataset: str, archive_path: Path, destination: Path) -> dict[str, object]:
    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    if archive_path.suffix.lower() != ".zip":
        raise ValueError("Only ZIP archives are accepted by the safe importer.")
    destination.mkdir(parents=True, exist_ok=True)
    existing = [path for path in destination.iterdir() if path.name != "IMPORT_RECORD.json"]
    if existing:
        raise FileExistsError(
            f"Destination is not empty: {destination}. Use a new directory to avoid overwriting data."
        )
    with zipfile.ZipFile(archive_path) as archive:
        members = validate_zip_members(archive, destination)
        archive.extractall(destination, members=members)
    result = {
        "dataset": dataset,
        "imported_at": datetime.now().astimezone().isoformat(),
        "archive_path": str(archive_path.resolve()),
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256(archive_path),
        "source": DATASET_METADATA[dataset]["source"],
        "license_note": DATASET_METADATA[dataset]["license_note"],
        "destination": str(destination.resolve()),
        "inventory": inventory(destination),
    }
    (destination / "IMPORT_RECORD.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_METADATA), required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    destination = args.destination or Path(DATASET_METADATA[args.dataset]["destination"])
    result = import_archive(args.dataset, args.archive, destination)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
