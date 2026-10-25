"""Evaluate a YOLO detector against VOC XML annotations at IoU 0.5."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO

from saferestore.metrics import average_precision_101, box_iou

VOC_TO_COCO = {
    "person": 0,
    "bicycle": 1,
    "bike": 1,
    "car": 2,
    "motorbike": 3,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7,
}


def parse_voc(path: Path) -> list[dict]:
    """Parse supported objects from one Pascal VOC XML file."""
    root = ET.parse(path).getroot()
    objects: list[dict] = []
    for node in root.findall("object"):
        name = (node.findtext("name") or "").strip().lower()
        if name not in VOC_TO_COCO:
            continue
        box = node.find("bndbox")
        if box is None:
            continue
        objects.append(
            {
                "class_id": VOC_TO_COCO[name],
                "box": np.array(
                    [
                        float(box.findtext("xmin", "0")),
                        float(box.findtext("ymin", "0")),
                        float(box.findtext("xmax", "0")),
                        float(box.findtext("ymax", "0")),
                    ],
                    dtype=np.float32,
                ),
            }
        )
    return objects


def collect_pairs(images_dir: Path, annotations_dir: Path) -> list[tuple[Path, Path]]:
    image_by_stem: dict[str, Path] = {}
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        for path in images_dir.glob(pattern):
            image_by_stem[path.stem] = path
    pairs = [(image_by_stem[path.stem], path) for path in annotations_dir.glob("*.xml") if path.stem in image_by_stem]
    if not pairs:
        raise FileNotFoundError("No matching image/XML pairs were found.")
    return sorted(pairs)


def evaluate_class(
    predictions: list[dict],
    ground_truth: dict[str, list[np.ndarray]],
    iou_threshold: float,
) -> dict[str, float | int]:
    total_ground_truth = sum(len(boxes) for boxes in ground_truth.values())
    matched = {image_id: np.zeros(len(boxes), dtype=bool) for image_id, boxes in ground_truth.items()}
    true_positives: list[float] = []
    false_positives: list[float] = []

    for prediction in sorted(predictions, key=lambda item: item["confidence"], reverse=True):
        boxes = ground_truth.get(prediction["image_id"], [])
        best_iou = 0.0
        best_index = -1
        for index, target_box in enumerate(boxes):
            if not matched[prediction["image_id"]][index]:
                overlap = box_iou(prediction["box"], target_box)
                if overlap > best_iou:
                    best_iou = overlap
                    best_index = index
        is_match = best_index >= 0 and best_iou >= iou_threshold
        if is_match:
            matched[prediction["image_id"]][best_index] = True
        true_positives.append(float(is_match))
        false_positives.append(float(not is_match))

    tp = np.cumsum(np.asarray(true_positives, dtype=np.float32))
    fp = np.cumsum(np.asarray(false_positives, dtype=np.float32))
    recall = tp / max(total_ground_truth, 1)
    precision = tp / np.maximum(tp + fp, 1e-12)
    final_tp = int(tp[-1]) if tp.size else 0
    final_fp = int(fp[-1]) if fp.size else 0
    return {
        "ground_truth": total_ground_truth,
        "predictions": len(predictions),
        "true_positives": final_tp,
        "false_positives": final_fp,
        "false_negatives": total_ground_truth - final_tp,
        "precision": float(precision[-1]) if precision.size else 0.0,
        "recall": float(recall[-1]) if recall.size else 0.0,
        "ap50": average_precision_101(recall, precision),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--annotations-dir", required=True)
    parser.add_argument("--config", default="configs/detector.yaml")
    parser.add_argument("--output", default="runs/evaluation/rtts_raw.json")
    parser.add_argument("--limit", type=int, default=0, help="Optional deterministic sample limit.")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    pairs = collect_pairs(Path(args.images_dir), Path(args.annotations_dir))
    if args.limit > 0:
        pairs = pairs[: args.limit]

    ground_truth: dict[int, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for image_path, annotation_path in pairs:
        for target in parse_voc(annotation_path):
            ground_truth[target["class_id"]][image_path.stem].append(target["box"])

    target_classes = sorted({class_id for class_id in ground_truth})
    model = YOLO(config["model"])
    results = model.predict(
        [str(image_path) for image_path, _ in pairs],
        imgsz=int(config["image_size"]),
        conf=float(config["confidence"]),
        iou=float(config["iou"]),
        device=config["device"],
        classes=target_classes,
        stream=True,
        verbose=False,
    )
    predictions: dict[int, list[dict]] = defaultdict(list)
    for (image_path, _), result in zip(pairs, results, strict=True):
        if result.boxes is None:
            continue
        for box in result.boxes:
            class_id = int(box.cls.item())
            predictions[class_id].append(
                {
                    "image_id": image_path.stem,
                    "confidence": float(box.conf.item()),
                    "box": np.asarray(box.xyxy[0].cpu(), dtype=np.float32),
                }
            )

    class_names = {int(key): value for key, value in config["classes"].items()}
    per_class = {
        class_names[class_id]: evaluate_class(predictions[class_id], ground_truth[class_id], 0.5)
        for class_id in target_classes
    }
    valid_ap = [metrics["ap50"] for metrics in per_class.values() if metrics["ground_truth"] > 0]
    summary = {
        "images": len(pairs),
        "iou_threshold": 0.5,
        "confidence_threshold": float(config["confidence"]),
        "map50": float(np.mean(valid_ap)) if valid_ap else 0.0,
        "per_class": per_class,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

