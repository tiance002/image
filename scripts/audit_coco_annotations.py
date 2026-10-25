"""Audit COCO annotations before running restoration and detection experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _duplicates(values: list[int]) -> list[int]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def audit_annotation_file(path: Path, images_dir: Path | None = None) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    categories = payload.get("categories", [])
    if not all(isinstance(items, list) for items in (images, annotations, categories)):
        raise ValueError(f"COCO arrays are malformed in {path}")

    image_ids = [int(item["id"]) for item in images]
    annotation_ids = [int(item["id"]) for item in annotations if "id" in item]
    category_ids = [int(item["id"]) for item in categories]
    image_lookup = {int(item["id"]): item for item in images}
    category_lookup = {int(item["id"]): str(item["name"]) for item in categories}

    invalid_image_references = 0
    invalid_category_references = 0
    invalid_boxes = 0
    out_of_bounds_boxes = 0
    crowd_annotations = 0
    category_counts: Counter[str] = Counter()
    for annotation in annotations:
        image_id = int(annotation.get("image_id", -1))
        category_id = int(annotation.get("category_id", -1))
        image = image_lookup.get(image_id)
        if image is None:
            invalid_image_references += 1
        if category_id not in category_lookup:
            invalid_category_references += 1
        else:
            category_counts[category_lookup[category_id]] += 1
        if int(annotation.get("iscrowd", 0)) == 1:
            crowd_annotations += 1
        box = annotation.get("bbox")
        if not isinstance(box, list) or len(box) != 4:
            invalid_boxes += 1
            continue
        x, y, width, height = (float(value) for value in box)
        if width <= 0 or height <= 0:
            invalid_boxes += 1
            continue
        if image is not None:
            image_width = float(image.get("width", 0))
            image_height = float(image.get("height", 0))
            if x < 0 or y < 0 or x + width > image_width + 1e-6 or y + height > image_height + 1e-6:
                out_of_bounds_boxes += 1

    weather_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    missing_images: list[str] = []
    for image in images:
        relative = Path(str(image["file_name"]))
        parts = relative.parts
        if parts:
            weather_counts[parts[0]] += 1
        if len(parts) > 1:
            split_counts[parts[1]] += 1
        if images_dir is not None and not (images_dir / relative).is_file():
            missing_images.append(relative.as_posix())

    structural_errors = (
        len(_duplicates(image_ids))
        + len(_duplicates(annotation_ids))
        + len(_duplicates(category_ids))
        + invalid_image_references
        + invalid_category_references
        + invalid_boxes
        + out_of_bounds_boxes
    )
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "images": len(images),
        "annotations": len(annotations),
        "categories": {str(key): value for key, value in sorted(category_lookup.items())},
        "category_counts": dict(sorted(category_counts.items())),
        "weather_counts": dict(sorted(weather_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "duplicate_image_ids": _duplicates(image_ids),
        "duplicate_annotation_ids": _duplicates(annotation_ids),
        "duplicate_category_ids": _duplicates(category_ids),
        "invalid_image_references": invalid_image_references,
        "invalid_category_references": invalid_category_references,
        "invalid_boxes": invalid_boxes,
        "out_of_bounds_boxes": out_of_bounds_boxes,
        "crowd_annotations": crowd_annotations,
        "images_dir": str(images_dir.resolve()) if images_dir is not None else None,
        "images_found": len(images) - len(missing_images) if images_dir is not None else None,
        "missing_image_count": len(missing_images) if images_dir is not None else None,
        "missing_image_examples": missing_images[:20],
        "structural_errors": structural_errors,
    }


def audit_directory(root: Path, images_dir: Path | None = None) -> dict[str, Any]:
    annotation_files = sorted(root.rglob("*.json"))
    reports = [audit_annotation_file(path, images_dir) for path in annotation_files]
    hashes: dict[str, list[str]] = defaultdict(list)
    for report in reports:
        hashes[report["sha256"]].append(report["path"])
    duplicate_payloads = [paths for paths in hashes.values() if len(paths) > 1]
    return {
        "scope": "coco_annotation_audit",
        "root": str(root.resolve()),
        "images_dir": str(images_dir.resolve()) if images_dir is not None else None,
        "annotation_file_count": len(reports),
        "total_declared_images": sum(report["images"] for report in reports),
        "total_annotations": sum(report["annotations"] for report in reports),
        "total_structural_errors": sum(report["structural_errors"] for report in reports),
        "duplicate_payload_groups": duplicate_payloads,
        "files": reports,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# COCO 标注审计报告",
        "",
        f"- 标注目录：`{report['root']}`",
        f"- JSON 文件：{report['annotation_file_count']}",
        f"- 声明图像记录：{report['total_declared_images']}",
        f"- 标注框记录：{report['total_annotations']}",
        f"- 结构错误：{report['total_structural_errors']}",
        f"- 重复载荷组：{len(report['duplicate_payload_groups'])}",
        "",
        "| 文件 | 图像 | 标注 | 结构错误 | 已找到图像 |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in report["files"]:
        found = "未检查" if item["images_found"] is None else str(item["images_found"])
        lines.append(
            f"| `{Path(item['path']).name}` | {item['images']} | {item['annotations']} | "
            f"{item['structural_errors']} | {found} |"
        )
    if report["duplicate_payload_groups"]:
        lines.extend(["", "## 重复 JSON 载荷", ""])
        for group in report["duplicate_payload_groups"]:
            lines.append("- " + " = ".join(f"`{path}`" for path in group))
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-root", required=True)
    parser.add_argument("--images-dir")
    parser.add_argument("--output", default="runs/data_audit/coco_annotation_audit.json")
    parser.add_argument("--markdown-output", default="runs/data_audit/coco_annotation_audit.md")
    parser.add_argument("--require-images", action="store_true")
    args = parser.parse_args()

    images_dir = Path(args.images_dir) if args.images_dir else None
    report = audit_directory(Path(args.annotations_root), images_dir)
    output = Path(args.output)
    markdown_output = Path(args.markdown_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if report["total_structural_errors"]:
        return 1
    if args.require_images and any(item["missing_image_count"] for item in report["files"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
