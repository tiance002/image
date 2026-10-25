"""Check manifest paths, labels and scene-level split leakage."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

VALID_WEATHER = {"rain", "fog", "rain_fog"}
VALID_SEVERITY = {"light", "medium", "heavy"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", help="Train/val/test CSV files.")
    parser.add_argument("--root", default=".", help="Root used to resolve image paths.")
    args = parser.parse_args()
    root = Path(args.root)
    scene_sets: dict[str, set[str]] = {}
    hash_sets: dict[str, set[str]] = {}
    errors: list[str] = []

    for manifest_name in args.manifests:
        manifest = Path(manifest_name)
        split = manifest.stem
        scene_sets[split] = set()
        hash_sets[split] = set()
        for row_number, row in enumerate(read_rows(manifest), start=2):
            if row.get("weather") not in VALID_WEATHER:
                errors.append(f"{manifest}:{row_number} invalid weather")
            if row.get("severity") not in VALID_SEVERITY:
                errors.append(f"{manifest}:{row_number} invalid severity")
            scene_sets[split].add(row.get("scene_id", ""))
            for key in ("weather_path", "clean_path"):
                value = row.get(key, "").strip()
                if not value:
                    continue
                image_path = root / value
                if not image_path.is_file():
                    errors.append(f"{manifest}:{row_number} missing {key}: {value}")
                else:
                    hash_sets[split].add(file_hash(image_path))

    splits = list(scene_sets)
    for index, left in enumerate(splits):
        for right in splits[index + 1 :]:
            shared_scenes = scene_sets[left] & scene_sets[right]
            shared_hashes = hash_sets[left] & hash_sets[right]
            if shared_scenes:
                errors.append(f"Scene leakage between {left} and {right}: {len(shared_scenes)}")
            if shared_hashes:
                errors.append(f"File leakage between {left} and {right}: {len(shared_hashes)}")

    if errors:
        raise SystemExit("Manifest check failed:\n- " + "\n- ".join(errors))
    print("Manifest check passed with no path, label, scene, or file-hash errors.")


if __name__ == "__main__":
    main()

