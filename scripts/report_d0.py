"""离线生成 D0 逐类对照，严格按文件名与真值配对。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from scripts.evaluate_coco_detection import evaluate_class  # noqa: E402
from saferestore.metrics import box_iou  # noqa: E402

CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 6: "train", 7: "truck"}


def load_table(path: Path) -> dict[str, dict]:
    """兼容两种导出格式，拒绝重复文件名导致静默覆盖。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["per_image"] if isinstance(payload, dict) else payload
    result = {}
    for row in rows:
        key = str(row.get("file_name") or row.get("image_id") or "")
        if not key or key in result:
            raise ValueError(f"缺少文件名或重复文件名：{key!r}，来源 {path}")
        result[key] = row
    return result


def detected_targets(boxes: list, predictions: list[dict]) -> set[int]:
    """按置信度一对一匹配，防止一个预测同时算作多个正确目标。"""
    matched = set()
    for pred in sorted(predictions, key=lambda p: p["confidence"], reverse=True):
        overlaps = [
            (box_iou(np.asarray(pred["box"]), np.asarray(box)), i)
            for i, box in enumerate(boxes) if i not in matched
        ]
        if overlaps:
            overlap, index = max(overlaps, key=lambda pair: pair[0])
            if overlap >= 0.5:
                matched.add(index)
    return matched


def summarize(table: dict[str, dict], keys: list[str]) -> dict:
    """固定七类；无真值的类别标为 null，不伪造 AP。"""
    per_class = {}
    for class_id, name in CLASSES.items():
        gt, predictions = {}, []
        for sample_id, key in enumerate(keys):
            row = table[key]
            boxes = row["ground_truth"].get(str(class_id), [])
            if boxes:
                gt[sample_id] = [np.asarray(box, dtype=np.float32) for box in boxes]
            predictions.extend(
                {**pred, "image_id": sample_id, "box": np.asarray(pred["box"], dtype=np.float32)}
                for pred in row["predictions"].get(str(class_id), [])
            )
        metrics = evaluate_class(predictions, gt, 0.5)
        if not metrics["ground_truth"]:
            metrics["ap50"] = None
        per_class[name] = metrics
    present = [m["ap50"] for m in per_class.values() if m["ap50"] is not None]
    six = [m["ap50"] for name, m in per_class.items()
           if name != "train" and m["ap50"] is not None]
    tp = sum(m["true_positives"] for m in per_class.values())
    fp = sum(m["false_positives"] for m in per_class.values())
    total = sum(m["ground_truth"] for m in per_class.values())
    return {
        "n_images": len(keys), "n_classes_with_gt": len(present),
        "map50": float(np.mean(present)) if present else None,
        "map50_without_train": float(np.mean(six)) if six else None,
        "recall": tp / max(total, 1), "precision": tp / max(tp + fp, 1),
        "false_positives_per_image": fp / max(len(keys), 1), "per_class": per_class,
    }


def compare(base: dict, method: dict, keys: list[str]) -> dict:
    """比较完整指定集合，缺图或真值不一致时立即失败。"""
    if not keys:
        raise ValueError("评测集合为空")
    kept = total = legacy_kept = legacy_total = 0
    changes = {name: {"kept": 0, "lost": 0, "gained": 0} for name in CLASSES.values()}
    for key in keys:
        if key not in base or key not in method:
            raise ValueError(f"缺少配对图像：{key}")
        if base[key]["ground_truth"] != method[key]["ground_truth"]:
            raise ValueError(f"真值不一致：{key}")
        for class_id, boxes in base[key]["ground_truth"].items():
            ref = base[key]["predictions"].get(class_id, [])
            cur = method[key]["predictions"].get(class_id, [])
            ref_hits, cur_hits = detected_targets(boxes, ref), detected_targets(boxes, cur)
            total += len(ref_hits)
            kept += len(ref_hits & cur_hits)
            counts = changes[CLASSES[int(class_id)]]
            counts["kept"] += len(ref_hits & cur_hits)
            counts["lost"] += len(ref_hits - cur_hits)
            counts["gained"] += len(cur_hits - ref_hits)
            for box in boxes:
                if any(box_iou(np.asarray(p["box"]), np.asarray(box)) >= 0.5 for p in ref):
                    legacy_total += 1
                    legacy_kept += int(any(
                        box_iou(np.asarray(p["box"]), np.asarray(box)) >= 0.5 for p in cur
                    ))
    baseline, candidate = summarize(base, keys), summarize(method, keys)
    deltas = {
        name: candidate["per_class"][name]["ap50"] - metrics["ap50"]
        for name, metrics in baseline["per_class"].items() if metrics["ap50"] is not None
    }
    return {
        "baseline": baseline, "method": candidate, "delta_ap50": deltas,
        "retention": candidate["map50"] / baseline["map50"] if baseline["map50"] else None,
        "opr": kept / total if total else None, "opr_denominator": total,
        "target_changes": changes,
        "opr_legacy": legacy_kept / legacy_total if legacy_total else None,
        "class_redline_pass": all(delta >= -0.05 for delta in deltas.values()),
        "recall_redline_pass": candidate["recall"] >= baseline["recall"] - 0.01,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--records", default="runs/acdc_benchmark/evaluation/weather_input_all406.json")
    parser.add_argument("--out", required=True, help="输出前缀，生成 .json 和 .md")
    args = parser.parse_args()
    base, method, records = [load_table(ROOT / p) for p in (args.baseline, args.method, args.records)]
    if set(base) != set(records) or set(method) != set(records):
        raise ValueError("记录、基线、方法的图像集合必须完全一致，禁止静默取交集")
    scopes = {"all406": list(records)}
    scopes["rain_fog200"] = [k for k, r in records.items() if r["weather"] in {"rain", "fog"}]
    if len(scopes["all406"]) != 406 or len(scopes["rain_fog200"]) != 200:
        raise ValueError("本报告要求完整 406 张记录及其中 200 张雨雾记录")
    scopes.update({w: [k for k, r in records.items() if r["weather"] == w]
                   for w in ("fog", "rain", "night", "snow")})
    result = {scope: compare(base, method, keys) for scope, keys in scopes.items()}
    output = {"split": "ACDC val（模型选择集，非独立测试）", "sources": vars(args), "scopes": result}
    lines = ["# D0 检测对照（ACDC val）", "", output["split"], "",
             "OPR 使用一对一匹配；opr_legacy 保留旧版按任意框重叠计算的结果。",
             "PSNR/LPIPS 与多种子未在本报告验证，不能据此宣告完整 D1 验收。", ""]
    for scope, row in result.items():
        b, m = row["baseline"], row["method"]
        lines.extend([f"## {scope}（{m['n_images']} 张）", "",
                      f"七类口径 mAP50：{b['map50']:.6f} → {m['map50']:.6f}；"
                      f"本集合有真值的类别数：{m['n_classes_with_gt']}。",
                      f"去 train：{b['map50_without_train']:.6f} → {m['map50_without_train']:.6f}。",
                      f"Recall：{b['recall']:.6f} → {m['recall']:.6f}；"
                      f"Precision：{b['precision']:.6f} → {m['precision']:.6f}。",
                      f"保留率：{row['retention']:.4%}；OPR：{row['opr']}；"
                      f"类别红线通过：{row['class_redline_pass']}；Recall 红线通过：{row['recall_redline_pass']}。",
                      "", "| 类别 | GT 框数 | 原图 AP50 | 方法 AP50 | 差值 | 丢失目标 | 新增检出 |",
                      "|---|---:|---:|---:|---:|---:|---:|"])
        for name, metrics in b["per_class"].items():
            ap, other = metrics["ap50"], m["per_class"][name]["ap50"]
            counts = row["target_changes"][name]
            if ap is None:
                lines.append(f"| {name} | 0 | — | — | — | 0 | 0 |")
            else:
                lines.append(f"| {name} | {metrics['ground_truth']} | {ap:.6f} | {other:.6f} | {other-ap:+.6f} | {counts['lost']} | {counts['gained']} |")
        # D0 要求只在基线 AP 确实为零的集合使用这句结论。
        if b["per_class"]["motorcycle"]["ap50"] == 0.0:
            lines.extend(["", "motorcycle 在基线下的 AP 为 0.0000，属检测器能力上限而非复原引入的退化。"])
        else:
            lines.extend(["", "此集合 motorcycle 基线 AP 非零；不能套用雨雾口径的零 AP 说明。"])
        lines.append("")
    prefix = ROOT / args.out
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    prefix.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(prefix.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
