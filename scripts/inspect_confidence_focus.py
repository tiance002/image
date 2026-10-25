"""离线导出406张图像中全部bus/train真值框的四方法匹配明细。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def require(condition: bool, message: str) -> None:
    """检查不满足时拒绝静默生成不完整案例集。"""
    if not condition:
        raise ValueError(message)


def iou(box: list, target: list) -> float:
    """计算xyxy框IoU，不启动检测器或使用GPU。"""
    width = max(0.0, min(box[2], target[2]) - max(box[0], target[0]))
    height = max(0.0, min(box[3], target[3]) - max(box[1], target[1]))
    intersection = width * height
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    target_area = max(0.0, target[2] - target[0]) * max(0.0, target[3] - target[1])
    union = area + target_area - intersection
    return intersection / union if union > 0 else 0.0


def matched_details(boxes: list, predictions: list[dict]) -> dict[int, dict]:
    """按置信度降序将每个预测贪心匹配至尚未使用且IoU最高的GT。"""
    matches = {}
    for prediction in sorted(predictions, key=lambda p: p["confidence"], reverse=True):
        candidates = [(iou(prediction["box"], box), index)
                      for index, box in enumerate(boxes) if index not in matches]
        if candidates:
            overlap, index = max(candidates, key=lambda pair: pair[0])
            if overlap >= 0.5:
                matches[index] = {
                    "score": float(prediction["confidence"]), "iou": float(overlap),
                    "prediction_box": prediction["box"],
                }
    return matches


def load_table(path: Path) -> dict:
    """读取两种本地逐图格式并拒绝重复文件名。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["per_image"] if isinstance(payload, dict) else payload
    table = {}
    for row in rows:
        key = str(row.get("file_name") or row.get("image_id") or "")
        require(bool(key) and key not in table, f"缺少或重复图像标识：{key}，来源{path}")
        table[key] = row
    return table


def export_cases(folder: Path) -> dict:
    """所有方法对每个真值框保留匹配或空值，不筛选有利案例。"""
    plan = json.loads((folder / "comparison_plan.json").read_text(encoding="utf-8"))
    require(plan.get("status") == "completed", "两组尚未完成")
    require(all(plan.get(k) is True for k in
                ("config_match", "batch_match", "teacher_selection_match")), "公平性核查未完成")
    arms = plan.get("arms", [])
    require(len(arms) == 2 and all(isinstance(a, str) and re.fullmatch(r"[A-Za-z0-9_-]+", a)
                                 for a in arms), "实验组路径无效")
    sources = {
        "input": "runs/acdc_benchmark/evaluation/weather_input_direct_406.json",
        "v4_48k": "runs/real_v4_allweather/preds_047999.json",
        "control_52k": f"runs/{arms[0]}/preds_051999.json",
        "guard_52k": f"runs/{arms[1]}/preds_051999.json",
    }
    records_path = "runs/acdc_benchmark/evaluation/weather_input_all406.json"
    records = load_table(ROOT / records_path)
    require(len(records) == 406, "需要完整406张记录")
    require(sum(r["weather"] == "fog" for r in records.values()) == 100
            and sum(r["weather"] == "rain" for r in records.values()) == 100,
            "主口径需要fog100与rain100")
    tables = {name: load_table(ROOT / path) for name, path in sources.items()}
    for name, table in tables.items():
        require(set(table) == set(records), f"{name}图像集合不一致")
        require(all(table[k]["ground_truth"] == r["ground_truth"] for k, r in records.items()),
                f"{name}逐图真值不一致")
    counts = {scope: {name: {"ground_truth": 0, "input_detected": 0,
                             "control_lost_input_detected": 0, "guard_recovered_control_lost": 0,
                             "guard_new_loss_vs_control": 0, "guard_lost_input_detected": 0,
                             "detected_per_method": {method: 0 for method in sources}}
                      for name in ("bus", "train")} for scope in ("all406", "rain_fog200")}
    cases = []
    for key, record in records.items():
        for class_id, class_name in (("5", "bus"), ("6", "train")):
            boxes = record["ground_truth"].get(class_id, [])
            matches = {method: matched_details(boxes, table[key]["predictions"].get(class_id, []))
                       for method, table in tables.items()}
            for index, box in enumerate(boxes):
                detected = {method: index in values for method, values in matches.items()}
                control_lost = detected["input"] and not detected["control_52k"]
                recovered = control_lost and detected["guard_52k"]
                new_loss = detected["control_52k"] and not detected["guard_52k"]
                flags = {"input_detected": detected["input"],
                         "control_lost_input_detected": control_lost,
                         "guard_recovered_control_lost": recovered,
                         "guard_new_loss_vs_control": new_loss,
                         "guard_lost_input_detected": detected["input"] and not detected["guard_52k"]}
                cases.append({"file_name": key, "weather": record["weather"],
                              "class": class_name, "class_id": int(class_id), "gt_index": index,
                              "gt_box": box,
                              "matches": {method: values.get(index) for method, values in matches.items()},
                              "flags": flags})
                scopes = ["all406"] + (["rain_fog200"] if record["weather"] in {"rain", "fog"} else [])
                for scope in scopes:
                    total = counts[scope][class_name]
                    total["ground_truth"] += 1
                    for flag, enabled in flags.items():
                        total[flag] += int(enabled)
                    for method, enabled in detected.items():
                        total["detected_per_method"][method] += int(enabled)
    return {
        "scope": "ACDC val；全部bus/train GT框；非独立test、非挑选案例",
        "sources": {**sources, "records": records_path},
        "matching": "同类预测按置信度降序，贪心匹配尚未使用的最高IoU GT，IoU>=.5；GT索引从0开始",
        "definitions": {
            "control_lost_input_detected": "输入已检出、control未检出",
            "guard_recovered_control_lost": "输入已检出、control未检出、guard检出",
            "guard_new_loss_vs_control": "control检出、guard未检出，不要求输入检出",
            "guard_lost_input_detected": "输入已检出、guard未检出",
        },
        "limitation": "未匹配不能单独归因为置信度下降；也可能涉及定位、类别或NMS。仅对已有预测做匹配，不进行低阈值复测。",
        "counts": counts, "n_cases": len(cases), "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", default="confidence_v8_20261025")
    args = parser.parse_args()
    require(bool(re.fullmatch(r"[A-Za-z0-9_-]+", args.comparison)), "比较名称无效")
    folder = ROOT / "runs" / args.comparison
    output = export_cases(folder)
    target = folder / "bus_train_cases.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
