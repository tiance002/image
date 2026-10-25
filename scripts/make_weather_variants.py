"""Create deterministic rain/fog variants of one image for demo comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from generate_demo_data import add_fog, add_rain
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="runs/assets")
    parser.add_argument("--severity", type=float, default=0.6)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(args.input) as handle:
        clear = handle.convert("RGB")
    fog = add_fog(clear, args.severity)
    rain = add_rain(clear, args.severity, seed=42)
    rain_fog = add_rain(fog, args.severity, seed=42)
    clear.save(output_dir / "clear.jpg", quality=95)
    fog.save(output_dir / "fog.jpg", quality=95)
    rain.save(output_dir / "rain.jpg", quality=95)
    rain_fog.save(output_dir / "rain_fog.jpg", quality=95)
    print(f"Saved deterministic demo variants to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

