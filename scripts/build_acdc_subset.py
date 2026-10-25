"""Build a deterministic ACDC weather subset from a combined COCO annotation file."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_condition(image: dict[str, Any]) -> str:
    normalized = str(image["file_name"]).replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    return parts[0].lower() if parts else ""


def build_subset(
    payload: dict[str, Any], conditions: set[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized_conditions = {
        condition.strip().lower() for condition in conditions if condition.strip()
    }
    images = [
        image
        for image in payload.get("images", [])
        if image_condition(image) in normalized_conditions
    ]
    image_ids = {int(image["id"]) for image in images}
    annotations = [
        annotation
        for annotation in payload.get("annotations", [])
        if int(annotation["image_id"]) in image_ids
    ]
    condition_counts = Counter(image_condition(image) for image in images)
    subset = {
        key: value
        for key, value in payload.items()
        if key not in {"images", "annotations", "saferestore_provenance"}
    }
    subset["images"] = images
    subset["annotations"] = annotations
    summary = {
        "conditions": sorted(normalized_conditions),
        "images": len(images),
        "annotations": len(annotations),
        "condition_counts": dict(sorted(condition_counts.items())),
    }
    return subset, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--conditions", nargs="+", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    input_hash = file_sha256(input_path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    subset, summary = build_subset(payload, set(args.conditions))
    if not subset["images"]:
        raise ValueError("No images matched --conditions.")
    subset["saferestore_provenance"] = {
        "source": str(input_path.resolve()),
        "source_sha256": input_hash,
        "conditions": summary["conditions"],
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps({**summary, "output": str(output_path.resolve())}, ensure_ascii=False, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
