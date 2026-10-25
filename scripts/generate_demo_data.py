"""Generate a tiny synthetic road-like dataset for pipeline validation only."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"


def clean_scene(seed: int, width: int = 192, height: int = 128) -> Image.Image:
    rng = random.Random(seed)
    image = Image.new("RGB", (width, height), (105, 170, 220))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, height // 2, width, height), fill=(80, 95, 75))
    draw.polygon(
        [(width // 2 - 20, height // 2), (width // 2 + 20, height // 2), (width, height), (0, height)],
        fill=(65, 65, 70),
    )
    for y in range(height // 2 + 8, height, 20):
        scale = (y - height // 2) / (height // 2)
        half = int(2 + scale * 5)
        draw.rectangle((width // 2 - half, y, width // 2 + half, min(y + 8, height)), fill=(230, 220, 120))
    for _ in range(4):
        x = rng.randint(20, width - 35)
        y = rng.randint(height // 2 + 8, height - 25)
        color = rng.choice([(190, 40, 35), (35, 90, 185), (225, 225, 225)])
        draw.rectangle((x, y, x + 18, y + 10), fill=color, outline=(20, 20, 20))
        draw.ellipse((x + 2, y + 8, x + 6, y + 12), fill=(10, 10, 10))
        draw.ellipse((x + 12, y + 8, x + 16, y + 12), fill=(10, 10, 10))
    return image


def add_fog(image: Image.Image, severity: float) -> Image.Image:
    array = np.asarray(image, dtype=np.float32)
    atmosphere = np.full_like(array, 225.0)
    transmission = 1.0 - 0.65 * severity
    fogged = array * transmission + atmosphere * (1.0 - transmission)
    return Image.fromarray(np.clip(fogged, 0, 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(radius=severity * 1.2)
    )


def add_rain(image: Image.Image, severity: float, seed: int) -> Image.Image:
    rng = random.Random(seed)
    rainy = image.copy()
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    count = int(70 + severity * 180)
    for _ in range(count):
        x = rng.randrange(image.width)
        y = rng.randrange(image.height)
        length = rng.randint(5, 13)
        draw.line((x, y, x - 3, y + length), fill=(220, 230, 240, int(70 + severity * 90)), width=1)
    return Image.alpha_composite(rainy.convert("RGBA"), overlay).convert("RGB")


def main() -> None:
    random.seed(42)
    severities = [("light", 0.3), ("medium", 0.6), ("heavy", 0.9)]
    split_sizes = {"train": 30, "val": 10, "test": 10}
    clean_dir = DATA_ROOT / "processed" / "clean"
    weather_dir = DATA_ROOT / "processed" / "weather"
    manifest_dir = DATA_ROOT / "manifests"
    for directory in (clean_dir, weather_dir, manifest_dir):
        directory.mkdir(parents=True, exist_ok=True)

    global_index = 0
    for split, size in split_sizes.items():
        rows: list[dict[str, str]] = []
        for local_index in range(size):
            index = global_index + local_index
            weather_type = ("rain", "fog", "rain_fog")[index % 3]
            severity_name, severity = severities[index % 3]
            clean = clean_scene(index)
            degraded = clean
            if weather_type in {"fog", "rain_fog"}:
                degraded = add_fog(degraded, severity)
            if weather_type in {"rain", "rain_fog"}:
                degraded = add_rain(degraded, severity, index)
            clean_path = clean_dir / f"scene_{index:04d}.png"
            weather_path = weather_dir / f"scene_{index:04d}_{weather_type}.png"
            clean.save(clean_path)
            degraded.save(weather_path)
            rows.append(
                {
                    "weather_path": clean_path.relative_to(ROOT).as_posix().replace("clean", "weather").replace(f"scene_{index:04d}.png", weather_path.name),
                    "clean_path": clean_path.relative_to(ROOT).as_posix(),
                    "scene_id": f"scene_{index:04d}",
                    "weather": weather_type,
                    "severity": severity_name,
                    "source": "synthetic_demo_only",
                }
            )
        global_index += size
        with (manifest_dir / f"{split}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print("Generated 50 synthetic demo pairs. Do not use them as scientific evidence.")


if __name__ == "__main__":
    main()

