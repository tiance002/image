"""Generate balanced weather x severity data without factor confounding."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

if __package__:
    from .generate_demo_data import add_fog, add_rain, clean_scene
else:
    from generate_demo_data import add_fog, add_rain, clean_scene

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "factorial"
WEATHERS = ("rain", "fog", "rain_fog")
SEVERITIES = (("light", 0.3), ("medium", 0.6), ("heavy", 0.9))


def build_plan(split: str, samples_per_cell: int, seed_offset: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seed = seed_offset
    for repetition in range(samples_per_cell):
        for weather in WEATHERS:
            for severity_name, severity_value in SEVERITIES:
                scene_id = f"factorial_{split}_{weather}_{severity_name}_{repetition:02d}"
                rows.append(
                    {
                        "scene_id": scene_id,
                        "weather": weather,
                        "severity": severity_name,
                        "severity_value": severity_value,
                        "seed": seed,
                    }
                )
                seed += 1
    return rows


def generate_split(split: str, samples_per_cell: int, seed_offset: int) -> list[dict[str, str]]:
    clean_dir = DATA_ROOT / "clean" / split
    weather_dir = DATA_ROOT / "weather" / split
    manifest_dir = DATA_ROOT / "manifests"
    for directory in (clean_dir, weather_dir, manifest_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    for item in build_plan(split, samples_per_cell, seed_offset):
        seed = int(item["seed"])
        clean = clean_scene(seed)
        degraded = clean
        weather = str(item["weather"])
        severity_value = float(item["severity_value"])
        if weather in {"fog", "rain_fog"}:
            degraded = add_fog(degraded, severity_value)
        if weather in {"rain", "rain_fog"}:
            degraded = add_rain(degraded, severity_value, seed + 100_000)
        scene_id = str(item["scene_id"])
        clean_path = clean_dir / f"{scene_id}.png"
        weather_path = weather_dir / f"{scene_id}.png"
        clean.save(clean_path)
        degraded.save(weather_path)
        manifest_rows.append(
            {
                "weather_path": weather_path.relative_to(ROOT).as_posix(),
                "clean_path": clean_path.relative_to(ROOT).as_posix(),
                "scene_id": scene_id,
                "weather": weather,
                "severity": str(item["severity"]),
                "source": "balanced_factorial_synthetic_only",
            }
        )
    manifest_path = manifest_dir / f"{split}.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    return manifest_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-per-cell", type=int, default=6)
    parser.add_argument("--val-per-cell", type=int, default=2)
    parser.add_argument("--test-per-cell", type=int, default=2)
    args = parser.parse_args()
    counts = {
        "train": args.train_per_cell,
        "val": args.val_per_cell,
        "test": args.test_per_cell,
    }
    if any(value <= 0 for value in counts.values()):
        raise ValueError("Every split must contain at least one sample per factor cell.")
    metadata: dict[str, object] = {
        "scope": "pipeline_and_ablation_only_not_public_scientific_evidence",
        "design": "full_factorial_weather_x_severity",
        "weather_levels": list(WEATHERS),
        "severity_levels": [name for name, _ in SEVERITIES],
        "splits": {},
    }
    seed_offset = 10_000
    for split, samples_per_cell in counts.items():
        rows = generate_split(split, samples_per_cell, seed_offset)
        seed_offset += len(rows) + 10_000
        cell_counts = Counter((row["weather"], row["severity"]) for row in rows)
        metadata["splits"][split] = {
            "sample_count": len(rows),
            "samples_per_cell": samples_per_cell,
            "cell_counts": {f"{weather}/{severity}": count for (weather, severity), count in sorted(cell_counts.items())},
        }
    (DATA_ROOT / "design.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
