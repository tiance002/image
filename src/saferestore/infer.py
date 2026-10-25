"""Single-image inference for SafeRestore-RF."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from saferestore.config import load_config
from saferestore.train import build_model


def load_weights(model: torch.nn.Module, checkpoint: str | None, device: torch.device) -> None:
    if checkpoint is None:
        return
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    state_dict = payload.get("model", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state_dict)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device).eval()
    load_weights(model, args.checkpoint, device)

    input_path = Path(args.input)
    with Image.open(input_path) as image:
        tensor = TF.to_tensor(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model(tensor)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    TF.to_pil_image(outputs["restored"][0].cpu()).save(output_dir / "restored.png")
    TF.to_pil_image(outputs["conservative"][0].cpu()).save(output_dir / "conservative.png")
    TF.to_pil_image(outputs["uncertainty"][0].cpu()).save(output_dir / "uncertainty.png")
    print(f"Saved inference outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

