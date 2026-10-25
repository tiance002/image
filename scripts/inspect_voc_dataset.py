"""Inspect a VOC-style dataset before evaluation and emit a JSON quality report."""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--annotations-dir", required=True)
    parser.add_argument("--output", default="data/reports/voc_inspection.json")
    parser.add_argument("--hash-images", action="store_true")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    annotations_dir = Path(args.annotations_dir)
    image_by_stem: dict[str, Path] = {}
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        for image_path in images_dir.glob(pattern):
            image_by_stem[image_path.stem] = image_path

    classes: Counter[str] = Counter()
    sizes: Counter[str] = Counter()
    missing_images: list[str] = []
    invalid_images: list[str] = []
    invalid_annotations: list[str] = []
    hashes: dict[str, str] = {}
    annotation_paths = sorted(annotations_dir.glob("*.xml"))
    for annotation_path in annotation_paths:
        image_path = image_by_stem.get(annotation_path.stem)
        if image_path is None:
            missing_images.append(annotation_path.name)
            continue
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                sizes[f"{image.width}x{image.height}"] += 1
            if args.hash_images:
                hashes[image_path.name] = sha256(image_path)
        except Exception as error:  # noqa: BLE001 - report all corrupt-image failures
            invalid_images.append(f"{image_path.name}: {error}")
        try:
            root = ET.parse(annotation_path).getroot()
            for node in root.findall("object"):
                classes[(node.findtext("name") or "unknown").strip().lower()] += 1
        except ET.ParseError as error:
            invalid_annotations.append(f"{annotation_path.name}: {error}")

    report = {
        "images": len(image_by_stem),
        "annotations": len(annotation_paths),
        "matched_pairs": len(annotation_paths) - len(missing_images),
        "classes": dict(classes.most_common()),
        "image_sizes": dict(sizes.most_common()),
        "missing_images": missing_images,
        "invalid_images": invalid_images,
        "invalid_annotations": invalid_annotations,
        "hashes": hashes,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "hashes"}, indent=2))
    if missing_images or invalid_images or invalid_annotations:
        raise SystemExit("Dataset inspection found blocking quality errors.")


if __name__ == "__main__":
    main()

