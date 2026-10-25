"""Path helpers shared by recursive image-processing pipelines."""

from __future__ import annotations

import json
from pathlib import Path

SUPPORTED_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png"}


def find_images(root: Path, *, recursive: bool = True) -> list[Path]:
    """Return supported image files in deterministic relative-path order."""
    iterator = root.rglob("*") if recursive else root.iterdir()
    paths = [
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    ]
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def find_coco_images(root: Path, annotations: Path) -> list[Path]:
    """Return exactly the image paths declared by a COCO annotation file.

    Annotation order is preserved so ``--limit`` selects the same records in
    preprocessing and evaluation. Unsafe, duplicate, missing, or unsupported
    paths fail fast instead of silently widening the experiment scope.
    """
    payload = json.loads(annotations.read_text(encoding="utf-8"))
    images = payload.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError(f"COCO annotations contain no image records: {annotations}")

    root_resolved = root.resolve()
    paths: list[Path] = []
    seen: set[str] = set()
    missing: list[str] = []
    for item in images:
        relative_text = str(item.get("file_name", ""))
        relative = Path(relative_text)
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root_resolved)
        except ValueError as error:
            raise ValueError(f"Unsafe COCO image path: {relative_text}") from error
        normalized = relative.as_posix()
        if normalized in seen:
            raise ValueError(f"Duplicate COCO image path: {relative_text}")
        seen.add(normalized)
        if candidate.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES or not candidate.is_file():
            missing.append(relative_text)
        paths.append(candidate)
    if missing:
        raise FileNotFoundError(
            f"COCO image coverage is incomplete: missing={len(missing)}, examples={missing[:10]}"
        )
    return paths


def mirrored_output_path(
    input_path: Path,
    input_root: Path,
    output_root: Path,
    *,
    suffix: str | None = None,
) -> Path:
    """Map an input path below its root to the same relative output location."""
    relative = input_path.resolve().relative_to(input_root.resolve())
    output = output_root / relative
    return output.with_suffix(suffix) if suffix is not None else output
