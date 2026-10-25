"""Run the frozen five-method ACDC restoration and detection benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

METHODS = ("weather_input", "dcp", "direct", "raw_conservative", "calibrated_fusion")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_image_coverage(images_dir: Path, annotations: Path) -> dict[str, Any]:
    payload = json.loads(annotations.read_text(encoding="utf-8"))
    missing = [
        str(item["file_name"])
        for item in payload.get("images", [])
        if not (images_dir / str(item["file_name"])).is_file()
    ]
    return {
        "declared": len(payload.get("images", [])),
        "found": len(payload.get("images", [])) - len(missing),
        "missing": len(missing),
        "missing_examples": missing[:20],
    }


def build_commands(
    *,
    python: str,
    images_dir: Path,
    annotations: Path,
    output_dir: Path,
    config: Path,
    checkpoint: Path,
    calibration: Path,
    detector_config: Path,
    methods: list[str],
    limit: int = 0,
    bootstrap_resamples: int = 2000,
) -> list[dict[str, Any]]:
    image_outputs = output_dir / "images"
    evaluation_outputs = output_dir / "evaluation"
    sources = {
        "weather_input": images_dir,
        "dcp": image_outputs / "dcp",
        "direct": image_outputs / "direct",
        "raw_conservative": image_outputs / "raw_conservative",
        "calibrated_fusion": image_outputs / "calibrated_fusion",
    }
    commands: list[dict[str, Any]] = []
    if "dcp" in methods:
        command = [
            python,
            "scripts/dcp_directory.py",
            "--input-dir",
            str(images_dir),
                "--output-dir",
                str(sources["dcp"]),
                "--annotations",
                str(annotations),
        ]
        if limit > 0:
            command.extend(["--limit", str(limit)])
        commands.append({"stage": "process", "method": "dcp", "command": command})
    restoration_modes = {
        "direct": "restored",
        "raw_conservative": "conservative",
        "calibrated_fusion": "calibrated",
    }
    for method, mode in restoration_modes.items():
        if method not in methods:
            continue
        command = [
            python,
            "scripts/restore_directory.py",
            "--input-dir",
            str(images_dir),
            "--output-dir",
            str(sources[method]),
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--annotations",
            str(annotations),
            "--mode",
            mode,
        ]
        if method == "calibrated_fusion":
            command.extend(["--fusion-calibration", str(calibration)])
        if limit > 0:
            command.extend(["--limit", str(limit)])
        commands.append({"stage": "process", "method": method, "command": command})
    for method in methods:
        command = [
            python,
            "scripts/evaluate_coco_detection.py",
            "--images-dir",
            str(sources[method]),
            "--annotations",
            str(annotations),
            "--config",
            str(detector_config),
            "--output",
            str(evaluation_outputs / f"{method}.json"),
        ]
        if limit > 0:
            command.extend(["--limit", str(limit)])
        commands.append({"stage": "evaluate", "method": method, "command": command})
    if len(methods) == len(METHODS) and set(methods) == set(METHODS):
        commands.append(
            {
                "stage": "report",
                "method": "five_method_summary",
                "command": [
                    python,
                    "scripts/summarize_acdc_benchmark.py",
                    "--evaluation-dir",
                    str(evaluation_outputs),
                    "--output-dir",
                    str(output_dir / "report"),
                    "--resamples",
                    str(bootstrap_resamples),
                ],
            }
        )
    return commands


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", default="data/raw/ACDC_images/rgb_anon")
    parser.add_argument("--annotations", default="data/derived/ACDC_detection/fog_rain_val.json")
    parser.add_argument("--output-dir", default="runs/acdc_benchmark")
    parser.add_argument("--config", default="configs/v3_factorial_seed42.yaml")
    parser.add_argument("--checkpoint", default="runs/v3_factorial_seed42/best.pt")
    parser.add_argument("--calibration", default="runs/v3_factorial_seed42/fusion_calibration.json")
    parser.add_argument("--detector-config", default="configs/detector.yaml")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    images_dir = Path(args.images_dir)
    annotations = Path(args.annotations)
    output_dir = Path(args.output_dir)
    config = Path(args.config)
    checkpoint = Path(args.checkpoint)
    calibration = Path(args.calibration)
    detector_config = Path(args.detector_config)
    required_files = [annotations, config, checkpoint, calibration, detector_config]
    missing_files = [str(path) for path in required_files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"Required benchmark files are missing: {missing_files}")

    coverage = inspect_image_coverage(images_dir, annotations)
    commands = build_commands(
        python=sys.executable,
        images_dir=images_dir,
        annotations=annotations,
        output_dir=output_dir,
        config=config,
        checkpoint=checkpoint,
        calibration=calibration,
        detector_config=detector_config,
        methods=args.methods,
        limit=args.limit,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    manifest_path = output_dir / "RUN_MANIFEST.json"
    manifest: dict[str, Any] = {
        "scope": "acdc_frozen_five_method_benchmark",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": "planned" if args.dry_run else "running",
        "project_root": str(project_root),
        "images_dir": str(images_dir.resolve()),
        "annotations": str(annotations.resolve()),
        "annotation_sha256": file_sha256(annotations),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "calibration": str(calibration.resolve()),
        "calibration_sha256": file_sha256(calibration),
        "detector_config": str(detector_config.resolve()),
        "detector_config_sha256": file_sha256(detector_config),
        "methods": args.methods,
        "limit": args.limit,
        "bootstrap_resamples": args.bootstrap_resamples,
        "coverage": coverage,
        "commands": commands,
        "executions": [],
    }
    write_manifest(manifest_path, manifest)
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    if coverage["missing"]:
        manifest["status"] = "blocked_missing_images"
        write_manifest(manifest_path, manifest)
        raise FileNotFoundError(
            f"ACDC image coverage is incomplete: found={coverage['found']}, "
            f"missing={coverage['missing']}"
        )

    environment = os.environ.copy()
    environment["YOLO_CONFIG_DIR"] = str((project_root / ".cache" / "ultralytics").resolve())
    environment["NO_PROXY"] = "127.0.0.1,localhost"
    environment["no_proxy"] = "127.0.0.1,localhost"
    for item in commands:
        started = time.perf_counter()
        execution = {**item, "started_at": datetime.now(timezone.utc).astimezone().isoformat()}
        try:
            subprocess.run(item["command"], cwd=project_root, env=environment, check=True)
        except subprocess.CalledProcessError as error:
            execution["returncode"] = error.returncode
            execution["duration_seconds"] = time.perf_counter() - started
            manifest["executions"].append(execution)
            manifest["status"] = "failed"
            write_manifest(manifest_path, manifest)
            raise
        execution["returncode"] = 0
        execution["duration_seconds"] = time.perf_counter() - started
        manifest["executions"].append(execution)
        write_manifest(manifest_path, manifest)
    manifest["status"] = "complete"
    manifest["completed_at"] = datetime.now(timezone.utc).astimezone().isoformat()
    write_manifest(manifest_path, manifest)
    print(f"ACDC benchmark complete: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
