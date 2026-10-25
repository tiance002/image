"""Run a fixed YOLO detector on image variants and save comparable outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", help="Image paths compared with identical settings.")
    parser.add_argument("--config", default="configs/detector.yaml")
    parser.add_argument("--output-dir", default="runs/detection")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_classes = {int(key): value for key, value in config["classes"].items()}
    model = YOLO(config["model"])
    results = model.predict(
        source=args.images,
        imgsz=int(config["image_size"]),
        conf=float(config["confidence"]),
        iou=float(config["iou"]),
        device=config["device"],
        classes=list(selected_classes),
        verbose=False,
    )

    summary: dict[str, dict] = {}
    for image_path, result in zip(args.images, results, strict=True):
        detections: list[dict] = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": selected_classes[class_id],
                        "confidence": round(float(box.conf.item()), 6),
                        "xyxy": [round(float(value), 2) for value in box.xyxy[0].tolist()],
                    }
                )
        image_path_object = Path(image_path)
        stem = image_path_object.stem
        result_key = stem
        if result_key in summary:
            result_key = f"{image_path_object.parent.name}_{stem}"
        suffix = 2
        unique_key = result_key
        while unique_key in summary:
            unique_key = f"{result_key}_{suffix}"
            suffix += 1
        cv2.imwrite(str(output_dir / f"{unique_key}_detected.jpg"), result.plot())
        summary[unique_key] = {
            "source": str(Path(image_path).resolve()),
            "count": len(detections),
            "detections": detections,
            "speed_ms": {key: round(float(value), 3) for key, value in result.speed.items()},
        }

    output_path = output_dir / "detections.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved fixed-protocol detection comparison to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
