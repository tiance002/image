"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REQUIRED_SECTIONS = {"project", "data", "model", "training", "loss", "evaluation"}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment configuration and validate its top-level schema."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise TypeError("Configuration root must be a mapping.")
    missing = REQUIRED_SECTIONS - set(config)
    if missing:
        raise ValueError(f"Configuration is missing sections: {sorted(missing)}")
    return config
