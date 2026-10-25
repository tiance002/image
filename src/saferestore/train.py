"""Training entry point for the restoration baseline."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from saferestore.config import load_config
from saferestore.data import PairedWeatherDataset
from saferestore.losses import SafeRestorationLoss
from saferestore.models import FidelityPreservingRestorer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(config: dict) -> FidelityPreservingRestorer:
    model_config = config["model"]
    return FidelityPreservingRestorer(
        base_channels=int(model_config["base_channels"]),
        degradation_dim=int(model_config["degradation_dim"]),
        uncertainty_floor=float(model_config["uncertainty_floor"]),
        use_degradation_gate=bool(model_config.get("use_degradation_gate", False)),
        minimum_restoration_strength=float(
            model_config.get("minimum_restoration_strength", 0.2)
        ),
    )


def smoke_train(config: dict) -> tuple[dict[str, float], str]:
    """Run two deterministic optimization steps on synthetic tensors."""
    seed = int(config["project"]["seed"])
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    criterion = SafeRestorationLoss(config["loss"]).to(device)
    optimizer = AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]))

    weather = torch.rand(2, 3, 64, 64, device=device)
    clean = torch.clamp(weather * 0.8 + 0.1, 0.0, 1.0)
    weather_label = torch.tensor([0, 2], device=device)
    severity = torch.tensor([0.25, 0.85], device=device)
    final_components: dict[str, torch.Tensor] = {}
    model.train()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(weather)
        loss, final_components = criterion(outputs, clean, weather_label, severity)
        if not torch.isfinite(loss):
            raise RuntimeError("Smoke test produced a non-finite loss.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    metrics = {name: float(value.cpu()) for name, value in final_components.items()}
    return metrics, str(device)


def run_epoch(
    model: FidelityPreservingRestorer,
    loader: DataLoader,
    criterion: SafeRestorationLoss,
    device: torch.device,
    optimizer: AdamW | None = None,
) -> dict[str, float]:
    """Run one train or validation epoch and return averaged loss components."""
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    sample_count = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in loader:
            weather = batch["weather"].to(device, non_blocking=True)
            clean = batch["clean"].to(device, non_blocking=True)
            weather_label = batch["weather_label"].to(device, non_blocking=True)
            severity = batch["severity"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(weather)
            loss, components = criterion(outputs, clean, weather_label, severity)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            batch_size = weather.shape[0]
            sample_count += batch_size
            for name, value in components.items():
                totals[name] = totals.get(name, 0.0) + float(value.cpu()) * batch_size
    return {name: value / sample_count for name, value in totals.items()}


def train_paired(config: dict) -> Path:
    """Train the restoration model from paired CSV manifests."""
    set_seed(int(config["project"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = tuple(int(value) for value in config["data"]["image_size"])
    train_dataset = PairedWeatherDataset(config["data"]["train_manifest"], image_size=image_size)
    val_dataset = PairedWeatherDataset(config["data"]["val_manifest"], image_size=image_size)
    loader_kwargs = {
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": int(config["data"]["num_workers"]),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    model = build_model(config).to(device)
    criterion = SafeRestorationLoss(config["loss"]).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    output_dir = Path(config["project"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.jsonl"
    best_path = output_dir / "best.pt"
    best_loss = float("inf")
    patience = 0

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record))
        if val_metrics["total"] < best_loss:
            best_loss = val_metrics["total"]
            patience = 0
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch}, best_path)
        else:
            patience += 1
            if patience >= int(config["training"]["early_stopping_patience"]):
                break
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML configuration.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a two-step synthetic training check without external data.",
    )
    parser.add_argument(
        "--train-paired",
        action="store_true",
        help="Train from the paired manifests declared in the configuration.",
    )
    parser.add_argument("--seed", type=int, help="Optional reproducibility seed override.")
    parser.add_argument("--output-dir", help="Optional run directory override.")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["project"]["seed"] = args.seed
    if args.output_dir is not None:
        config["project"]["output_dir"] = args.output_dir

    if args.train_paired:
        best_path = train_paired(config)
        print(f"Training complete. Best checkpoint: {best_path.resolve()}")
        return
    if not args.smoke_test:
        raise SystemExit(
            "Choose --smoke-test for a synthetic tensor check or --train-paired for manifest training."
        )

    metrics, device = smoke_train(config)
    output_dir = Path(config["project"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "smoke_metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "device": device, **metrics}, indent=2))


if __name__ == "__main__":
    main()
