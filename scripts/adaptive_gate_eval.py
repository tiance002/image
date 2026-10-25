"""自适应逐图门控：让检测器自己决定每张图用「原图」还是「复原图」。

动机
----
v4 相对原图直检，在 406 张全天气上只掉了 5 个检测框，但因mAP50 是类别等权平均，
这5 个框落在 bus / train / truck 上，造成 rain/night/snow 各掉近 1 个百分点。
逐图看则是：fog 25 张变好 / 14 张变差 / 60 张持平 —— 也就是说
**同一套模型在不同图上时好时坏**。

这提示一个零训练成本的思路：与其全局用复原结果，不如**逐图选优**。
但"选优"必须用不依赖真值的判据，否则就是拿测试集调参。这里用检测器自身的输出一致性：

    对同一张图，分别用原图与复原图跑检测，比较两路结果的「一致度」——
    哪一路的检测更自信、更少自相矛盾（同类框互相 IoU 过高即为矛盾），
    就用哪一路。

关键点：判据完全来自检测器输出，不看任何真值框，因此可以合法地用于推理。

本脚本先算「理论上限」（用真值逐图选优）作为参照，
再算「检测器判据」的实际效果，二者之差就是门控判据的质量。
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(".")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.diagnose_detection_rootcause import dataset_map50  # noqa: E402
from saferestore.metrics.detection import box_iou  # noqa: E402

CN = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 6: "train", 7: "truck"}


def load_table(path: Path) -> dict:
    """把逐图预测读成 {文件名: {类别: [框...]}}。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload["per_image"] if isinstance(payload, dict) and "per_image" in payload else payload
    table = {}
    for item in items:
        name = item.get("file_name") or item["image_id"]
        table[name] = {
            "ground_truth": item["ground_truth"],
            "predictions": item.get("predictions", {}),
        }
    return table


def consistency(predictions: dict, class_ids=(0, 1, 2, 3, 5, 6, 7)) -> float:
    """检测结果的自洽度打分（越大越可信，不需要真值）。

    组成：
      1. 平均置信度 —— 越自信通常说明特征越清晰；
      2. 同类框之间的重叠惩罚 —— 同一个物体被框两次（IoU 很高）说明特征不稳定；
      3. 框数奖励 —— 检出更多目标通常意味着保留的信息更多。
    """
    confs = []
    overlaps = []
    n_boxes = 0
    for cid in class_ids:
        items = predictions.get(str(cid), [])
        if not items:
            continue
        n_boxes += len(items)
        boxes = [np.asarray(x["box"], dtype=np.float32) for x in items]
        confs.extend(float(x["confidence"]) for x in items)
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                overlaps.append(box_iou(boxes[i], boxes[j]))
    if not confs:
        return -1.0
    mean_conf = float(np.mean(confs))
    # 高 IoU 的重复框说明特征不稳，按 IoU 超过 0.6 记一次惩罚
    penalty = float(np.mean([max(0.0, o - 0.6) for o in overlaps])) if overlaps else 0.0
    return mean_conf + 0.05 * min(n_boxes, 40) - 2.0 * penalty


def build(table: dict, records: dict, pick) -> list:
    """按 pick(文件名) 决定用哪一路的预测，组装成评测列表。"""
    out = []
    for name in sorted(records):
        gt = records[name]["ground_truth"]
        preds = pick(name)
        out.append({"ground_truth": gt, "predictions": preds})
    return out


def main() -> None:
    rec_path = ROOT / "runs/acdc_benchmark/evaluation/weather_input_all406.json"
    records = {
        r["file_name"]: r
        for r in json.loads(rec_path.read_text(encoding="utf-8"))["per_image"]
    }

    base = load_table(ROOT / "runs/acdc_benchmark/evaluation/weather_input_direct_406.json")
    method = load_table(ROOT / "runs/real_v4_allweather/preds_043999.json")

    keys = sorted(set(records) & set(base) & set(method))
    print(f"配对图数 {len(keys)}")

    # ---------- 参照：各自单独用 ----------
    m_base = dataset_map50(
        [{"ground_truth": base[k]["ground_truth"], "predictions": base[k]["predictions"]} for k in keys]
    )
    m_meth = dataset_map50(
        [{"ground_truth": method[k]["ground_truth"], "predictions": method[k]["predictions"]} for k in keys]
    )
    print(f"原图直检 mAP50 {m_base['map50']:.4f}   v4 mAP50 {m_meth['map50']:.4f}")

    # ---------- 门控 1：检测器自洽度 ----------
    def pick_consistency(name):
        sb = consistency(base[name]["predictions"])
        sm = consistency(method[name]["predictions"])
        return method[name]["predictions"] if sm >= sb else base[name]["predictions"]

    m_gate = dataset_map50(build(base, {k: records[k] for k in keys}, pick_consistency))
    n_switch = sum(
        1 for k in keys if consistency(method[k]["predictions"]) >= consistency(base[k]["predictions"])
    )
    print(
        f"检测器自洽度门控 mAP50 {m_gate['map50']:.4f}   "
        f"选复原的图 {n_switch}/{len(keys)}   相对原图 {100 * (m_gate['map50'] - m_base['map50']):+.2f}pp"
    )

    # ---------- 参照上限：用真值逐图选优（不可用于推理，仅作上界）----------
    def gt_score(preds, gt):
        """用真值算该路预测的 TP 数（仅用于测量理论上限）。"""
        tp = 0
        for cid, boxes in gt.items():
            pb = [np.asarray(x["box"], dtype=np.float32) for x in preds.get(str(cid), [])]
            for g in boxes:
                g = np.asarray(g, dtype=np.float32)
                if any(box_iou(b, g) >= 0.5 for b in pb):
                    tp += 1
        return tp

    def pick_oracle(name):
        gt = records[name]["ground_truth"]
        sb = gt_score(base[name]["predictions"], gt)
        sm = gt_score(method[name]["predictions"], gt)
        return method[name]["predictions"] if sm >= sb else base[name]["predictions"]

    m_oracle = dataset_map50(build(base, {k: records[k] for k in keys}, pick_oracle))
    print(
        f"真值逐图选优（上界）mAP50 {m_oracle['map50']:.4f}   "
        f"相对原图 {100 * (m_oracle['map50'] - m_base['map50']):+.2f}pp"
    )

    # ---------- 逐天气拆解 ----------
    print()
    print("=== 逐天气 mAP50 ===")
    print("  %-6s %9s %9s %9s %9s" % ("天气", "原图", "v4", "自洽门控", "真值上界"))
    for w in ("fog", "rain", "night", "snow"):
        sub = [k for k in keys if records[k]["weather"] == w]
        r = {k: records[k] for k in sub}
        a = dataset_map50([{"ground_truth": base[k]["ground_truth"], "predictions": base[k]["predictions"]} for k in sub])["map50"]
        b = dataset_map50([{"ground_truth": method[k]["ground_truth"], "predictions": method[k]["predictions"]} for k in sub])["map50"]
        c = dataset_map50(build(base, r, pick_consistency))["map50"]
        d = dataset_map50(build(base, r, pick_oracle))["map50"]
        print("  %-6s %9.4f %9.4f %9.4f %9.4f" % (w, a, b, c, d))

    out = {
        "n_images": len(keys),
        "baseline_map50": m_base["map50"],
        "v4_map50": m_meth["map50"],
        "gate_consistency_map50": m_gate["map50"],
        "oracle_map50": m_oracle["map50"],
        "gate_picks_method": n_switch,
    }
    (ROOT / "runs/real_v4_allweather/adaptive_gate.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n已写 runs/real_v4_allweather/adaptive_gate.json")


if __name__ == "__main__":
    main()
