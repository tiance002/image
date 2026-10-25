"""Paired adverse-weather image dataset backed by a CSV manifest."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

WEATHER_TO_INDEX = {"rain": 0, "fog": 1, "rain_fog": 2}
SEVERITY_TO_VALUE = {"light": 0.25, "medium": 0.5, "heavy": 0.85}


class PairedWeatherDataset(Dataset):
    """Read paired weather/clean images and their degradation metadata."""

    def __init__(
        self,
        manifest: str | Path,
        root: str | Path = ".",
        image_size: tuple[int, int] = (256, 256),
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        manifest_path = Path(manifest)
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise ValueError(f"Manifest contains no samples: {manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def _load_rgb(self, relative_path: str) -> torch.Tensor:
        path = self.root / relative_path
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = TF.resize(image, self.image_size, antialias=True)
            return TF.to_tensor(image)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        weather_name = row["weather"].strip()
        severity_name = row["severity"].strip()
        if weather_name not in WEATHER_TO_INDEX:
            raise ValueError(f"Unsupported weather type: {weather_name}")
        if severity_name not in SEVERITY_TO_VALUE:
            raise ValueError(f"Unsupported severity: {severity_name}")

        weather = self._load_rgb(row["weather_path"])
        clean = self._load_rgb(row["clean_path"])
        weather_label = torch.tensor(WEATHER_TO_INDEX[weather_name], dtype=torch.long)
        severity = torch.tensor(SEVERITY_TO_VALUE[severity_name], dtype=torch.float32)
        return {
            "weather": weather,
            "clean": clean,
            "weather_label": weather_label,
            "severity": severity,
            "scene_id": row["scene_id"],
        }

