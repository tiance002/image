"""Restore every image in a directory while preserving stems for VOC evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF
from tqdm import tqdm

from saferestore.config import load_config
from saferestore.models import calibrated_conservative_fusion
from saferestore.paths import find_coco_images, find_images, mirrored_output_path
from saferestore.train import build_model


def select_restoration(
    outputs: dict[str, torch.Tensor],
    weather: torch.Tensor,
    mode: str,
    fusion_parameters: dict[str, float] | None = None,
) -> torch.Tensor:
    """Select a direct, raw-conservative, or validation-calibrated output."""
    if mode == "restored":
        return outputs["restored"]
    if mode == "conservative":
        return outputs["conservative"]
    if mode != "calibrated":
        raise ValueError(f"Unsupported restoration mode: {mode}")
    if fusion_parameters is None:
        raise ValueError("Calibrated mode requires fusion parameters.")
    fused, _ = calibrated_conservative_fusion(
        outputs["restored"],
        weather,
        outputs["uncertainty"],
        outputs["degradation"][:, 2],
        **fusion_parameters,
    )
    return fused


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--annotations", help="Optional COCO JSON; process only its declared image paths."
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--mode", choices=("restored", "conservative", "calibrated"), default="conservative"
    )
    parser.add_argument(
        "--fusion-calibration",
        default="runs/v3_factorial_seed42/fusion_calibration.json",
        help="Validation-frozen parameters required by --mode calibrated.",
    )
    parser.add_argument("--save-uncertainty", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device).eval()
    payload = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(payload.get("model", payload))
    fusion_parameters: dict[str, float] | None = None
    if args.mode == "calibrated":
        calibration_path = Path(args.fusion_calibration)
        if not calibration_path.is_file():
            raise FileNotFoundError(f"Fusion calibration is missing: {calibration_path}")
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        fusion_parameters = {
            name: float(value) for name, value in calibration["parameters"].items()
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    uncertainty_dir = output_dir.parent / f"{output_dir.name}_uncertainty"
    if args.save_uncertainty:
        uncertainty_dir.mkdir(parents=True, exist_ok=True)

    input_dir = Path(args.input_dir)
    paths = (
        find_coco_images(input_dir, Path(args.annotations))
        if args.annotations
        else find_images(input_dir)
    )
    if args.limit > 0:
        paths = paths[: args.limit]
    if not paths:
        raise FileNotFoundError(f"No supported images found in {args.input_dir}")
    processed = 0
    skipped = 0
    for input_path in tqdm(paths, desc="Restoring"):
        output_path = mirrored_output_path(input_path, input_dir, output_dir)
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(input_path) as image:
            tensor = TF.to_tensor(image.convert("RGB")).unsqueeze(0).to(device)
        with torch.inference_mode():
            outputs = model(tensor)
        selected = select_restoration(outputs, tensor, args.mode, fusion_parameters)
        TF.to_pil_image(selected[0].cpu()).save(output_path)
        if args.save_uncertainty:
            uncertainty_path = mirrored_output_path(
                input_path, input_dir, uncertainty_dir, suffix=".png"
            )
            uncertainty_path.parent.mkdir(parents=True, exist_ok=True)
            TF.to_pil_image(outputs["uncertainty"][0].cpu()).save(uncertainty_path)
        processed += 1
    print(f"Completed directory restoration: processed={processed}, skipped={skipped}")


if __name__ == "__main__":
    main()
