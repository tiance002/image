"""真实天气帧的检测训练流：完整视野 letterbox，并同步变换真实框。"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


ACDC_TO_COCO = {24: 0, 26: 2, 27: 7, 28: 5, 31: 6, 32: 3, 33: 1}
SUPERVISED_COCO_CLASSES = (0, 1, 2, 3, 5, 6, 7)


def letterbox_rgb(image, boxes, ignore_boxes, size=640, flip=False):
    """RGB uint8 -> RGB float CHW；框均为原图 xyxy，返回归一化坐标。"""
    height, width = image.shape[:2]
    scale = min(size / height, size / width)
    new_width, new_height = round(width * scale), round(height * scale)
    left, top = (size - new_width) // 2, (size - new_height) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top + new_height, left:left + new_width] = cv2.resize(image, (new_width, new_height))

    def transform(values):
        values = np.asarray(values, dtype=np.float32).reshape(-1, 4).copy()
        # 使用实际整数缩放尺寸，避免 round 带来的半像素框偏差。
        values[:, [0, 2]] = values[:, [0, 2]] * (new_width / width) + left
        values[:, [1, 3]] = values[:, [1, 3]] * (new_height / height) + top
        if flip:
            values[:, [0, 2]] = size - values[:, [2, 0]]
        return values / size

    transformed = transform(boxes)
    ignored = transform(ignore_boxes)
    if flip:
        canvas = canvas[:, ::-1].copy()
    # native criterion 接受归一化 cx,cy,w,h。
    xywh = transformed.copy()
    xywh[:, :2] = (transformed[:, :2] + transformed[:, 2:]) / 2
    xywh[:, 2:] = transformed[:, 2:] - transformed[:, :2]
    tensor = torch.from_numpy(np.ascontiguousarray(canvas.transpose(2, 0, 1))).float() / 255
    return tensor, torch.from_numpy(xywh), torch.from_numpy(ignored)


class ACDCDetectionDataset(Dataset):
    """只接受 train 天气帧；crowd、rider 和 ignore 框进入忽略区域。"""

    def __init__(self, annotation_paths, images_root, image_size=640, horizontal_flip=False, seed=42):
        if image_size <= 0 or image_size % 32:
            raise ValueError("image_size 必须是正的32倍数")
        if isinstance(annotation_paths, (str, Path)):
            annotation_paths = [annotation_paths]
        self.image_size = int(image_size)
        self.horizontal_flip = horizontal_flip
        self.seed = int(seed)
        self.images_root = Path(images_root).resolve()
        self.records = []
        seen = set()
        counts = Counter()
        for annotation_path in annotation_paths:
            payload = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
            categories = {int(c["id"]): c["name"] for c in payload["categories"]}
            expected = {24: "person", 26: "car", 27: "truck", 28: "bus", 31: "train", 32: "motorcycle", 33: "bicycle"}
            if any(categories.get(key) != value for key, value in expected.items()):
                raise ValueError("ACDC 类别定义不完整或 ID 映射不符")
            grouped = defaultdict(list)
            for annotation in payload["annotations"]:
                grouped[annotation["image_id"]].append(annotation)
            for image in payload["images"]:
                name = str(image["file_name"]).replace("\\", "/")
                parts = PurePosixPath(name).parts
                if len(parts) < 4 or parts[0] not in {"fog", "rain", "night", "snow"} or parts[1] != "train" or ".." in parts:
                    raise ValueError(f"检测训练禁止 val/test/reference 路径：{name}")
                if name in seen:
                    raise ValueError(f"重复训练图像（请勿混合 combined 与分天气标注）：{name}")
                seen.add(name)
                path = (self.images_root / name).resolve()
                if not path.is_relative_to(self.images_root) or not path.is_file():
                    raise FileNotFoundError(path)
                boxes, classes, ignore_boxes = [], [], []
                for annotation in grouped[image["id"]]:
                    x, y, width, height = map(float, annotation["bbox"])
                    if width <= 0 or height <= 0 or not np.isfinite([x, y, width, height]).all():
                        raise ValueError(f"非法 GT 框：{name}")
                    box = [max(0, x), max(0, y), min(image["width"], x + width), min(image["height"], y + height)]
                    if box[2] <= box[0] or box[3] <= box[1]:
                        raise ValueError(f"GT 框完全位于图像外：{name}")
                    category = int(annotation["category_id"])
                    if annotation.get("iscrowd", 0) or annotation.get("ignore", 0) or category not in ACDC_TO_COCO:
                        ignore_boxes.append(box)
                        counts["ignored_boxes"] += 1
                    else:
                        boxes.append(box)
                        classes.append(ACDC_TO_COCO[category])
                        counts[f"class_{ACDC_TO_COCO[category]}"] += 1
                self.records.append(dict(file_name=name, path=path, image_id=image["id"],
                                         width=image["width"], height=image["height"],
                                         boxes=boxes, classes=classes, ignore_boxes=ignore_boxes))
                counts[parts[0]] += 1
        if not self.records:
            raise ValueError("真实检测训练集为空")
        self.counts = dict(counts)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[int(index)]
        bgr = cv2.imdecode(np.fromfile(record["path"], dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape[:2] != (record["height"], record["width"]):
            raise ValueError(f"天气帧无法读取或尺寸与 GT 不符：{record['path']}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        flip = self.horizontal_flip and bool(np.random.default_rng(self.seed + int(index)).integers(2))
        image, boxes, ignored = letterbox_rgb(rgb, record["boxes"], record["ignore_boxes"], self.image_size, flip)
        return {"img": image, "cls": torch.tensor(record["classes"], dtype=torch.float32).reshape(-1, 1),
                "bboxes": boxes, "ignore_boxes": ignored, "file_name": record["file_name"],
                "image_id": record["image_id"], "index": int(index)}


def collate_detection_batch(items):
    """构造 Ultralytics 原生 batch；忽略区域独立附加。"""
    return {"img": torch.stack([item["img"] for item in items]),
            "cls": torch.cat([item["cls"] for item in items]),
            "bboxes": torch.cat([item["bboxes"] for item in items]),
            "batch_idx": torch.cat([torch.full((len(item["cls"]),), i, dtype=torch.float32) for i, item in enumerate(items)]),
            "ignore_boxes": torch.cat([item["ignore_boxes"] for item in items]),
            "ignore_batch_idx": torch.cat([torch.full((len(item["ignore_boxes"]),), i, dtype=torch.long) for i, item in enumerate(items)]),
            "file_names": [item["file_name"] for item in items],
            "indices": [item["index"] for item in items]}
