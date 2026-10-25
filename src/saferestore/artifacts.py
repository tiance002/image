"""Reproducible result packaging for interactive image experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

SAFE_EXPERIMENT_ID = re.compile(r"^[A-Za-z0-9_-]+$")
BATCH_SUMMARY_FIELDS = [
    "source_file",
    "experiment_id",
    "rain_probability",
    "fog_probability",
    "severity",
    "safety_score",
    "decision",
    "detected_objects",
    "pipeline_total_ms",
]


def create_image_result_bundle(
    output_root: Path,
    experiment_id: str,
    images: dict[str, np.ndarray],
    metrics: dict[str, Any],
) -> Path:
    """Save image outputs and package them with a checksummed manifest."""
    if not SAFE_EXPERIMENT_ID.fullmatch(experiment_id):
        raise ValueError("experiment_id may only contain letters, numbers, '_' and '-'.")
    if not images:
        raise ValueError("At least one image artifact is required.")

    run_directory = output_root / experiment_id
    run_directory.mkdir(parents=True, exist_ok=False)
    artifacts: list[tuple[Path, str]] = []
    for role, array in images.items():
        if not SAFE_EXPERIMENT_ID.fullmatch(role):
            raise ValueError(f"Invalid image role: {role}")
        image = np.asarray(array)
        if image.ndim not in (2, 3):
            raise ValueError(f"Image '{role}' must have 2 or 3 dimensions.")
        image = np.clip(image, 0, 255).astype(np.uint8)
        path = run_directory / f"{role}.png"
        Image.fromarray(image).save(path)
        artifacts.append((path, role))

    metrics_path = run_directory / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    artifacts.append((metrics_path, "metrics"))

    manifest_files = []
    for path, role in artifacts:
        content = path.read_bytes()
        manifest_files.append(
            {
                "name": path.name,
                "role": role,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "files": manifest_files,
    }
    manifest_path = run_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    bundle_path = output_root / f"{experiment_id}_result_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, _ in artifacts:
            archive.write(path, arcname=path.name)
        archive.write(manifest_path, arcname=manifest_path.name)
    return bundle_path


def summarize_image_runs(output_root: Path, limit: int = 10) -> list[list[object]]:
    """Return recent valid image experiment metrics for the demo history table."""
    if not output_root.is_dir():
        return []
    metrics_files = sorted(
        output_root.glob("saferestore_image_*/metrics.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    rows: list[list[object]] = []
    for path in metrics_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            degradation = payload["degradation"]
            safety = payload["safety"]
            timings = payload["timing_ms"]
            counts = payload.get("detections", {}).get("counts", {})
            rows.append(
                [
                    path.parent.name,
                    round(float(degradation["rain_probability"]), 3),
                    round(float(degradation["fog_probability"]), 3),
                    round(float(degradation["severity"]), 3),
                    round(float(safety["score"]), 3),
                    "采用复原" if safety["accepted_restoration"] else "回退输入",
                    int(sum(int(value) for value in counts.values())),
                    round(float(timings["pipeline_total"]), 1),
                ]
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
            continue
        if len(rows) >= limit:
            break
    return rows


def read_image_result_summary(bundle_path: Path, source_file: str) -> dict[str, object]:
    """Read the compact batch row from a complete single-image result bundle."""
    with zipfile.ZipFile(bundle_path) as archive:
        metrics = json.loads(archive.read("metrics.json"))
        manifest = json.loads(archive.read("manifest.json"))
    degradation = metrics["degradation"]
    safety = metrics["safety"]
    counts = metrics.get("detections", {}).get("counts", {})
    return {
        "source_file": Path(source_file).name,
        "experiment_id": manifest["experiment_id"],
        "rain_probability": round(float(degradation["rain_probability"]), 4),
        "fog_probability": round(float(degradation["fog_probability"]), 4),
        "severity": round(float(degradation["severity"]), 4),
        "safety_score": round(float(safety["score"]), 4),
        "decision": "accepted" if safety["accepted_restoration"] else "input_fallback",
        "detected_objects": int(sum(int(value) for value in counts.values())),
        "pipeline_total_ms": round(float(metrics["timing_ms"]["pipeline_total"]), 3),
    }


def create_batch_result_bundle(
    output_root: Path,
    batch_id: str,
    item_bundles: list[Path],
    summary_rows: list[dict[str, object]],
) -> tuple[Path, Path]:
    """Package per-image result bundles and a CSV summary for one batch run."""
    if not SAFE_EXPERIMENT_ID.fullmatch(batch_id):
        raise ValueError("batch_id may only contain letters, numbers, '_' and '-'.")
    if not item_bundles or len(item_bundles) != len(summary_rows):
        raise ValueError("Batch bundles and summary rows must have the same non-zero length.")
    missing = [str(path) for path in item_bundles if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot create batch bundle; missing: {', '.join(missing)}")
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / f"{batch_id}_summary.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=BATCH_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)

    files = [(csv_path, "batch_summary")]
    files.extend((path, "single_image_result_bundle") for path in item_bundles)
    manifest_files = []
    for path, role in files:
        content = path.read_bytes()
        manifest_files.append(
            {
                "name": path.name,
                "role": role,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    manifest = {
        "schema_version": 1,
        "batch_id": batch_id,
        "item_count": len(item_bundles),
        "files": manifest_files,
    }
    bundle_path = output_root / f"{batch_id}_result_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(csv_path, arcname=csv_path.name)
        for path in item_bundles:
            archive.write(path, arcname=f"items/{path.name}")
        archive.writestr(
            "manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2),
        )
    return bundle_path, csv_path
