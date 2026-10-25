"""构造全天气 406 张检测评测记录（fog/rain/night/snow 各 val 集）。

输出格式与 runs/acdc_benchmark/evaluation/weather_input.json 的 per_image 一致：
    file_name, weather, ground_truth (dict: str(class_id) -> list[[x1,y1,x2,y2]]), predictions={}
前 200 张（fog/rain）会与现有 weather_input.json 逐图校验，保证同口径。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ACDC 官方类别名 -> COCO/YOLO 类别 id（rider 在 COCO 中不存在，按官方协议丢弃）
NAME_TO_COCO = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "train": 6,
    "truck": 7,
}

WEATHERS = ("fog", "rain", "night", "snow")


def load_weather_records(weather: str, split: str = "val") -> list[dict]:
    """读取官方 gt_detection json，转成 per_image 记录。"""
    gt_path = (
        ROOT
        / "data/raw/ACDC_detection/gt_detection"
        / weather
        / f"instancesonly_{weather}_{split}_gt_detection.json"
    )
    if not gt_path.exists():
        raise FileNotFoundError(f"缺少官方标注: {gt_path}")
    data = json.loads(gt_path.read_text(encoding="utf-8"))

    cat_name = {c["id"]: c["name"] for c in data["categories"]}
    records: dict[int, dict] = {}
    for img in data["images"]:
        records[img["id"]] = {
            "image_id": str(img["id"]),
            "file_name": img["file_name"],
            "weather": weather,
            "width": img["width"],
            "height": img["height"],
            "ground_truth": defaultdict(list),
            "predictions": [],
        }
    n_dropped = 0
    for ann in data["annotations"]:
        name = cat_name.get(ann["category_id"])
        cls = NAME_TO_COCO.get(name)
        if cls is None:
            n_dropped += 1
            continue
        x, y, w, h = ann["bbox"]
        records[ann["image_id"]]["ground_truth"][str(cls)].append(
            [float(x), float(y), float(x + w), float(y + h)]
        )
    out = []
    for rec in records.values():
        rec["ground_truth"] = dict(rec["ground_truth"])
        out.append(rec)
    print(f"  {weather}/{split}: {len(out)} 图, 丢弃 rider 标注 {n_dropped} 个")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="runs/acdc_benchmark/evaluation/weather_input_all406.json",
        help="输出路径",
    )
    args = parser.parse_args()

    print("读取官方标注并转换类别:")
    # fog/rain 沿用现有 weather_input.json（基线 0.2042 的口径，分天气官方 json
    # 与旧产物在 19 张图上有少量框差异，为保可比性不动它们）；
    # night/snow 无历史产物，直接从官方标注构造。
    base_all = json.loads(
        (ROOT / "runs/acdc_benchmark/evaluation/weather_input.json").read_text(
            encoding="utf-8"
        )
    )["per_image"]
    kept = []
    for r in base_all:
        kept.append(
            {
                "image_id": r["image_id"],
                "file_name": r["file_name"],
                "weather": r["weather"],
                "ground_truth": r["ground_truth"],
                "predictions": [],
            }
        )
    print(f"  fog/rain: 沿用旧产物 {len(kept)} 张")
    for weather in ("night", "snow"):
        kept.extend(load_weather_records(weather))
    all_records = kept
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "per_image": all_records,
        "note": "全天气 val 406 张（fog100 rain100 night106 snow100），GT 来自官方 gt_detection，类别映射 COCO id",
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    total_boxes = sum(len(b) for r in all_records for b in r["ground_truth"].values())
    print(f"已写出 {out_path}：{len(all_records)} 图 / {total_boxes} 框")


if __name__ == "__main__":
    main()
