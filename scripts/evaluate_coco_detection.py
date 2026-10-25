"""Inspect and evaluate a COCO-format road detection set with the fixed YOLO protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO

from saferestore.metrics import average_precision_101, box_iou

NAME_TO_YOLO = {
    "person": 0,
    "pedestrian": 0,
    "bicycle": 1,
    "bike": 1,
    "car": 2,
    "motorcycle": 3,
    "motorbike": 3,
    "bus": 5,
    "train": 6,
    "truck": 7,
}


def xywh_to_xyxy(box: list[float]) -> np.ndarray:
    x, y, width, height = (float(value) for value in box)
    return np.asarray([x, y, x + width, y + height], dtype=np.float32)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_class(
    predictions: list[dict], ground_truth: dict[int, list[np.ndarray]], iou_threshold: float
) -> dict[str, float | int]:
    total_ground_truth = sum(len(boxes) for boxes in ground_truth.values())
    matched = {
        image_id: np.zeros(len(boxes), dtype=bool) for image_id, boxes in ground_truth.items()
    }
    true_positives: list[float] = []
    false_positives: list[float] = []
    for prediction in sorted(predictions, key=lambda item: item["confidence"], reverse=True):
        image_id = int(prediction["image_id"])
        boxes = ground_truth.get(image_id, [])
        best_iou, best_index = 0.0, -1
        for index, target_box in enumerate(boxes):
            if not matched[image_id][index]:
                overlap = box_iou(prediction["box"], target_box)
                if overlap > best_iou:
                    best_iou, best_index = overlap, index
        is_match = best_index >= 0 and best_iou >= iou_threshold
        if is_match:
            matched[image_id][best_index] = True
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


def match_image_records(
    ground_truth: dict[str, list[list[float]]],
    predictions: dict[str, list[dict]],
    iou_threshold: float,
) -> dict[str, float | int]:
    """Return micro detection counts for one image across supported classes."""
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    class_ids = sorted(set(ground_truth) | set(predictions), key=int)
    for class_id in class_ids:
        targets = [np.asarray(box, dtype=np.float32) for box in ground_truth.get(class_id, [])]
        matched = np.zeros(len(targets), dtype=bool)
        class_predictions = sorted(
            predictions.get(class_id, []), key=lambda item: float(item["confidence"]), reverse=True
        )
        for prediction in class_predictions:
            best_iou, best_index = 0.0, -1
            prediction_box = np.asarray(prediction["box"], dtype=np.float32)
            for index, target_box in enumerate(targets):
                if matched[index]:
                    continue
                overlap = box_iou(prediction_box, target_box)
                if overlap > best_iou:
                    best_iou, best_index = overlap, index
            if best_index >= 0 and best_iou >= iou_threshold:
                matched[best_index] = True
                true_positives += 1
            else:
                false_positives += 1
        false_negatives += int((~matched).sum())
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def build_per_image_records(
    images: list[dict],
    predictions: dict[int, list[dict]],
    ground_truth: dict[int, dict[int, list[np.ndarray]]],
    iou_threshold: float,
) -> list[dict]:
    predictions_by_image: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for class_id, items in predictions.items():
        for item in items:
            predictions_by_image[int(item["image_id"])][str(class_id)].append(
                {
                    "confidence": float(item["confidence"]),
                    "box": np.asarray(item["box"], dtype=np.float32).tolist(),
                }
            )
    records: list[dict] = []
    for image in images:
        image_id = int(image["id"])
        targets = {
            str(class_id): [box.tolist() for box in image_targets.get(image_id, [])]
            for class_id, image_targets in ground_truth.items()
            if image_targets.get(image_id)
        }
        image_predictions = dict(predictions_by_image.get(image_id, {}))
        metrics = match_image_records(targets, image_predictions, iou_threshold)
        records.append(
            {
                "image_id": image_id,
                "file_name": str(image["file_name"]),
                "weather": Path(str(image["file_name"])).parts[0],
                "ground_truth": targets,
                "predictions": image_predictions,
                **metrics,
            }
        )
    return records


def load_dataset(
    images_dir: Path, annotation_path: Path, limit: int = 0
) -> tuple[list[dict], dict[int, str], dict[int, dict[int, list[np.ndarray]]], dict]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {
        int(item["id"]): str(item["name"]).strip().lower() for item in payload["categories"]
    }
    supported_categories = {
        category_id: NAME_TO_YOLO[name]
        for category_id, name in categories.items()
        if name in NAME_TO_YOLO
    }
    images = sorted(payload["images"], key=lambda item: int(item["id"]))
    valid_images: list[dict] = []
    missing_images: list[str] = []
    for item in images:
        path = images_dir / str(item["file_name"])
        if path.is_file():
            valid_images.append({**item, "path": path})
        else:
            missing_images.append(str(path))
    if limit > 0:
        valid_images = valid_images[:limit]
    selected_ids = {int(item["id"]) for item in valid_images}
    ground_truth: dict[int, dict[int, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    invalid_boxes = 0
    ignored_annotations = 0
    class_counts: Counter[str] = Counter()
    for annotation in payload["annotations"]:
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        if image_id not in selected_ids:
            continue
        if category_id not in supported_categories or int(annotation.get("iscrowd", 0)) == 1:
            ignored_annotations += 1
            continue
        box = xywh_to_xyxy(annotation["bbox"])
        if box[2] <= box[0] or box[3] <= box[1]:
            invalid_boxes += 1
            continue
        yolo_id = supported_categories[category_id]
        ground_truth[yolo_id][image_id].append(box)
        class_counts[categories[category_id]] += 1
    yolo_names = {
        class_id: name for name, class_id in NAME_TO_YOLO.items() if name in categories.values()
    }
    inspection = {
        "annotation_file": str(annotation_path.resolve()),
        "images_declared": len(images),
        "images_evaluated": len(valid_images),
        "missing_images": missing_images,
        "supported_category_count": len(supported_categories),
        "class_counts": dict(sorted(class_counts.items())),
        "ignored_annotations": ignored_annotations,
        "invalid_boxes": invalid_boxes,
    }
    return valid_images, yolo_names, ground_truth, inspection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--config", default="configs/detector.yaml")
    parser.add_argument("--output", default="runs/evaluation/coco_detection.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Validate files and labels without loading YOLO.",
    )
    args = parser.parse_args()

    images, class_names, ground_truth, inspection = load_dataset(
        Path(args.images_dir), Path(args.annotations), args.limit
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not images:
        raise FileNotFoundError("No annotated image files were found under --images-dir.")
    if args.inspect_only:
        output_path.write_text(
            json.dumps(inspection, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(inspection, ensure_ascii=False, indent=2))
        return

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    iou_threshold = 0.5
    target_classes = sorted(ground_truth)
    model_path = Path(str(config["model"]))
    model = YOLO(str(model_path if model_path.is_file() else config["model"]))
    results = model.predict(
        source=[str(item["path"]) for item in images],
        imgsz=int(config["image_size"]),
        conf=float(config["confidence"]),
        iou=float(config["iou"]),
        device=config["device"] if torch.cuda.is_available() else "cpu",
        classes=target_classes,
        stream=True,
        verbose=False,
    )
    predictions: dict[int, list[dict]] = defaultdict(list)
    for item, result in zip(images, results, strict=True):
        if result.boxes is None:
            continue
        for box in result.boxes:
            class_id = int(box.cls.item())
            predictions[class_id].append(
                {
                    "image_id": int(item["id"]),
                    "confidence": float(box.conf.item()),
                    "box": np.asarray(box.xyxy[0].cpu(), dtype=np.float32),
                }
            )
    fixed_names = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        6: "train",
        7: "truck",
    }
    per_class = {
        fixed_names.get(class_id, class_names.get(class_id, str(class_id))): evaluate_class(
            predictions[class_id], ground_truth[class_id], iou_threshold
        )
        for class_id in target_classes
    }
    valid_ap = [metrics["ap50"] for metrics in per_class.values() if metrics["ground_truth"]]
    per_image = build_per_image_records(images, predictions, ground_truth, iou_threshold)
    total_tp = sum(int(item["true_positives"]) for item in per_image)
    total_fp = sum(int(item["false_positives"]) for item in per_image)
    total_fn = sum(int(item["false_negatives"]) for item in per_image)
    summary = {
        "scope": "fixed_yolov8n_coco_detection_evaluation",
        "inspection": inspection,
        "annotation_sha256": file_sha256(Path(args.annotations)),
        "detector_config_sha256": file_sha256(Path(args.config)),
        "iou_threshold": iou_threshold,
        "confidence_threshold": float(config["confidence"]),
        "map50": float(np.mean(valid_ap)) if valid_ap else 0.0,
        "micro_precision": total_tp / max(total_tp + total_fp, 1),
        "micro_recall": total_tp / max(total_tp + total_fn, 1),
        "miss_rate": total_fn / max(total_tp + total_fn, 1),
        "false_positives_per_image": total_fp / max(len(per_image), 1),
        "per_class": per_class,
        "per_image": per_image,
    }
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
