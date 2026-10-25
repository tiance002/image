"""Validate the repository structure and experiment configuration without training."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = [
    "README.md",
    "pyproject.toml",
    "configs/baseline.yaml",
    "configs/task_aware.yaml",
    "docs/PROJECT_CHARTER.md",
    "docs/ROADMAP.md",
    "docs/ACCEPTANCE_CRITERIA.md",
    "docs/EXPERIMENT_PLAN.md",
    "docs/DATA_GUIDE.md",
    "src/saferestore/train.py",
    "src/saferestore/infer.py",
]
REQUIRED_CONFIG_SECTIONS = {"project", "data", "model", "training", "loss", "evaluation"}


def main() -> int:
    errors: list[str] = []
    for relative in REQUIRED_FILES:
        if not (ROOT / relative).is_file():
            errors.append(f"Missing required file: {relative}")

    for relative in ("configs/baseline.yaml", "configs/task_aware.yaml"):
        path = ROOT / relative
        if not path.is_file():
            continue
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        missing = REQUIRED_CONFIG_SECTIONS - set(config or {})
        if missing:
            errors.append(f"{relative} missing sections: {sorted(missing)}")

    if errors:
        print("Project validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"Project validation passed: {len(REQUIRED_FILES)} required files found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

