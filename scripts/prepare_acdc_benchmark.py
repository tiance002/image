"""Check ACDC RGB readiness and optionally continue import and benchmarking.

This command is intentionally idempotent.  It can be run repeatedly while the
large RGB archive is downloading, after the archive arrives, and after images
have already been extracted.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from import_public_dataset import import_archive
from run_acdc_benchmark import inspect_image_coverage

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_NAME = "rgb_anon_trainvaltest.zip"
DEFAULT_ANNOTATIONS = ROOT / "data" / "derived" / "ACDC_detection" / "fog_rain_val.json"
DEFAULT_IMAGES_DIR = ROOT / "data" / "raw" / "ACDC_images" / "rgb_anon"
DEFAULT_DESTINATION = ROOT / "data" / "raw" / "ACDC_images"
DEFAULT_REPORT = ROOT / "runs" / "data_audit" / "acdc_readiness.json"
DEFAULT_MARKDOWN = ROOT / "runs" / "data_audit" / "acdc_readiness.md"
REQUIRED_BENCHMARK_FILES = (
    ROOT / "configs" / "v3_factorial_seed42.yaml",
    ROOT / "runs" / "v3_factorial_seed42" / "best.pt",
    ROOT / "runs" / "v3_factorial_seed42" / "fusion_calibration.json",
    ROOT / "configs" / "detector.yaml",
)


def default_search_dirs() -> list[Path]:
    candidates = [ROOT / "data" / "archives" / "ACDC", Path.home() / "Downloads"]
    return [path for path in candidates if path.exists()]


def find_archive(explicit: Path | None, search_dirs: list[Path]) -> Path | None:
    if explicit is not None:
        return explicit
    for directory in search_dirs:
        candidate = directory / ARCHIVE_NAME
        if candidate.is_file():
            return candidate
    return None


def find_partial_downloads(search_dirs: list[Path]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    patterns = (f"{ARCHIVE_NAME}*", "*.crdownload", "*.part", "*.download")
    recent_cutoff = time.time() - 6 * 60 * 60
    seen: set[Path] = set()
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        for pattern in patterns:
            for path in directory.glob(pattern):
                if not path.is_file() or path in seen or path.name == ARCHIVE_NAME:
                    continue
                if not path.name.startswith(ARCHIVE_NAME) and path.stat().st_mtime < recent_cutoff:
                    continue
                seen.add(path)
                results.append(
                    {
                        "path": str(path.resolve()),
                        "size_bytes": path.stat().st_size,
                        "modified_at": datetime.fromtimestamp(path.stat().st_mtime)
                        .astimezone()
                        .isoformat(),
                    }
                )
    return sorted(results, key=lambda item: item["modified_at"], reverse=True)


def inspect_archive(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
        "valid_zip": False,
        "member_count": 0,
        "image_member_count": 0,
        "error": None,
    }
    if not path.is_file():
        result["error"] = "archive_not_found"
        return result
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            result["valid_zip"] = True
            result["member_count"] = len(members)
            result["image_member_count"] = sum(
                Path(item.filename).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
                for item in members
            )
    except (OSError, zipfile.BadZipFile) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def determine_status(
    *,
    archive: dict[str, Any] | None,
    partial_downloads: list[dict[str, Any]],
    coverage: dict[str, Any] | None,
    missing_prerequisites: list[str],
) -> str:
    if missing_prerequisites:
        return "prerequisites_missing"
    if coverage is not None and coverage["declared"] > 0 and coverage["missing"] == 0:
        return "ready_for_benchmark"
    if archive is not None and archive["exists"] and archive["valid_zip"]:
        return "archive_ready_for_import"
    if archive is not None and archive["exists"] and not archive["valid_zip"]:
        return "archive_invalid_or_incomplete"
    if partial_downloads:
        return "download_in_progress_or_stale_partial"
    return "waiting_for_rgb_archive"


def next_action(status: str, archive: dict[str, Any] | None) -> str:
    if status == "ready_for_benchmark":
        return f'"{sys.executable}" scripts/run_acdc_benchmark.py'
    if status == "archive_ready_for_import" and archive is not None:
        return (
            f'"{sys.executable}" scripts/prepare_acdc_benchmark.py '
            f'--archive "{archive["path"]}" --import-if-ready'
        )
    if status == "prerequisites_missing":
        return "Restore the listed frozen configs/checkpoint/calibration before benchmarking."
    if status == "archive_invalid_or_incomplete":
        return "Wait for the ZIP download to finish; do not rename or extract a partial file."
    if status == "download_in_progress_or_stale_partial":
        return (
            "Keep the browser and this app open while the temporary file is growing; rerun "
            "this command after it is renamed to rgb_anon_trainvaltest.zip."
        )
    return (
        "Download rgb_anon_trainvaltest.zip from the official ACDC page, then rerun "
        "this command."
    )


def build_report(
    *,
    archive_path: Path | None,
    search_dirs: list[Path],
    images_dir: Path,
    annotations: Path,
) -> dict[str, Any]:
    archive = inspect_archive(archive_path)
    partial_downloads = find_partial_downloads(search_dirs)
    prerequisites = [annotations, *REQUIRED_BENCHMARK_FILES]
    missing_prerequisites = [str(path.resolve()) for path in prerequisites if not path.is_file()]
    coverage = (
        inspect_image_coverage(images_dir, annotations)
        if annotations.is_file() and images_dir.is_dir()
        else None
    )
    status = determine_status(
        archive=archive,
        partial_downloads=partial_downloads,
        coverage=coverage,
        missing_prerequisites=missing_prerequisites,
    )
    return {
        "scope": "acdc_rgb_readiness",
        "checked_at": datetime.now().astimezone().isoformat(),
        "status": status,
        "archive": archive,
        "search_dirs": [str(path.resolve()) for path in search_dirs],
        "partial_downloads": partial_downloads,
        "images_dir": str(images_dir.resolve()),
        "annotations": str(annotations.resolve()),
        "coverage": coverage,
        "missing_prerequisites": missing_prerequisites,
        "next_action": next_action(status, archive),
    }


def render_markdown(report: dict[str, Any]) -> str:
    coverage = report["coverage"]
    coverage_text = (
        "尚未形成可审计图像目录"
        if coverage is None
        else f"{coverage['found']}/{coverage['declared']}（缺少 {coverage['missing']}）"
    )
    archive = report["archive"]
    archive_text = "未发现"
    if archive is not None:
        archive_text = (
            f"`{archive['path']}`；{archive['size_bytes']} bytes；"
            f"ZIP={'有效' if archive['valid_zip'] else '无效或未完成'}"
        )
    lines = [
        "# ACDC RGB 数据就绪报告",
        "",
        f"- 状态：`{report['status']}`",
        f"- RGB 压缩包：{archive_text}",
        f"- fog/rain 验证图覆盖：{coverage_text}",
        f"- 冻结评测前置文件缺失：{len(report['missing_prerequisites'])}",
        f"- 检出的临时下载文件：{len(report['partial_downloads'])}",
        "",
        "## 下一步",
        "",
        f"`{report['next_action']}`",
        "",
        "此报告只说明数据和工程流水线是否就绪，不代表已经得到科研实验结论。",
        "",
    ]
    return "\n".join(lines)


def write_report(report: dict[str, Any], output: Path, markdown_output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_output.write_text(render_markdown(report), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--search-dir", type=Path, action="append")
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--import-if-ready", action="store_true")
    parser.add_argument("--run-benchmark", action="store_true")
    args = parser.parse_args()

    search_dirs = args.search_dir or default_search_dirs()
    archive_path = find_archive(args.archive, search_dirs)
    report = build_report(
        archive_path=archive_path,
        search_dirs=search_dirs,
        images_dir=args.images_dir,
        annotations=args.annotations,
    )

    if args.import_if_ready and report["status"] == "archive_ready_for_import":
        destination = args.destination
        existing = [path for path in destination.iterdir()] if destination.is_dir() else []
        if existing:
            raise FileExistsError(
                f"Refusing to import into non-empty destination without complete coverage: {destination}"
            )
        import_archive("acdc", archive_path, destination)
        report = build_report(
            archive_path=archive_path,
            search_dirs=search_dirs,
            images_dir=args.images_dir,
            annotations=args.annotations,
        )

    write_report(report, args.output, args.markdown_output)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.run_benchmark:
        if report["status"] != "ready_for_benchmark":
            print("Benchmark not started because ACDC data is not ready.", file=sys.stderr)
            return 2
        subprocess.run(
            [sys.executable, "scripts/run_acdc_benchmark.py"], cwd=ROOT, check=True
        )
    return 0 if report["status"] == "ready_for_benchmark" else 2


if __name__ == "__main__":
    raise SystemExit(main())
