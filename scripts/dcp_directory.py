"""Apply the classical Dark Channel Prior baseline to an image directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from saferestore.baselines import dark_channel_dehaze
from saferestore.paths import find_coco_images, find_images, mirrored_output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--annotations", help="Optional COCO JSON; process only its declared image paths."
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=15)
    parser.add_argument("--omega", type=float, default=0.95)
    parser.add_argument("--save-transmission", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    transmission_dir = output_dir.parent / f"{output_dir.name}_transmission"
    if args.save_transmission:
        transmission_dir.mkdir(parents=True, exist_ok=True)
    paths = (
        find_coco_images(input_dir, Path(args.annotations))
        if args.annotations
        else find_images(input_dir)
    )
    if args.limit > 0:
        paths = paths[: args.limit]
    if not paths:
        raise FileNotFoundError(f"No supported images found in {input_dir}")

    processed = 0
    skipped = 0
    for input_path in tqdm(paths, desc="DCP dehazing"):
        output_path = mirrored_output_path(input_path, input_dir, output_dir)
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        with Image.open(input_path) as handle:
            image = np.asarray(handle.convert("RGB"))
        restored, transmission = dark_channel_dehaze(
            image,
            patch_size=args.patch_size,
            omega=args.omega,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(restored).save(output_path)
        if args.save_transmission:
            transmission_image = np.clip(transmission * 255.0, 0, 255).astype(np.uint8)
            transmission_path = mirrored_output_path(
                input_path, input_dir, transmission_dir, suffix=".png"
            )
            transmission_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(transmission_image).save(transmission_path)
        processed += 1
    print(
        f"Completed DCP directory processing: processed={processed}, skipped={skipped}, "
        f"output={output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
