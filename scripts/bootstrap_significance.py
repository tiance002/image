"""配对 bootstrap：判断两个方法的 mAP 差值是否超出抽样噪声。

动机
----
ACDC val 的 fog 100 + rain 100 就是评测集的全部，无法再扩样本。
200 张图下 mAP50 的抽样标准差在 0.005 量级，而我们关心的差距（例如 0.1930 与
0.2042 差 0.011）与它同阶 —— 只报两个点估计，无法区分"真的变好了"和"抽到了
一批好图"。

做法是**配对 bootstrap**：每次有放回地重采样 200 张图，两个方法使用**同一组**
重采样索引，各自重算 mAP 后再取差值，重复若干次得到差值的分布。配对能消掉
"这张图本身就难/易"带来的共同方差，只留下处理方式造成的差异。

输出：两个方法的点估计、差值的 95% 置信区间，以及"差值 > 0"的经验概率。
若置信区间跨过 0，则应表述为"无显著差异"，而不是"提升了 x 个点"。

用法：
    PYTHONPATH=src;. python scripts/bootstrap_significance.py \
        --baseline runs/acdc_benchmark/evaluation/weather_input.json \
        --method runs/real_v3/preds_024000.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from scripts.diagnose_detection_rootcause import dataset_map50, evaluate_class  # noqa: E402

CLASS_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 6: "train", 7: "truck"}


def load_per_image(path: Path) -> dict[str, dict]:
    """读取逐图结果，统一成 {image_id: {ground_truth, predictions}}。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "per_image" in payload:
        payload = payload["per_image"]
    table: dict[str, dict] = {}
    for item in payload:
        # 基线产物的 image_id 是序号（"0"、"1"…），真正的路径在 file_name；
        # 本项目的导出脚本则把路径写在 image_id。优先取 file_name 才能对齐两边。
        key = item.get("file_name") or item.get("image_id")
        if not key:
            raise ValueError(f"{path} 中缺少可用的图像标识")
        table[str(key)] = {
            "ground_truth": item["ground_truth"],
            "predictions": item["predictions"],
        }
    return table


def per_class_ap(per_image: list[dict]) -> dict[int, float]:
    """逐类 AP50，用于看差值是不是被单一类别主导。"""
    predictions: dict[int, list[dict]] = {}
    ground_truth: dict[int, dict[int, list[np.ndarray]]] = {}
    for sample_id, record in enumerate(per_image):
        for class_id_text, boxes in record["ground_truth"].items():
            class_id = int(class_id_text)
            ground_truth.setdefault(class_id, {})[sample_id] = [
                np.asarray(box, dtype=np.float32) for box in boxes
            ]
        for class_id_text, items in record["predictions"].items():
            class_id = int(class_id_text)
            predictions.setdefault(class_id, []).extend(
                {
                    "image_id": sample_id,
                    "confidence": float(item["confidence"]),
                    "box": np.asarray(item["box"], dtype=np.float32),
                }
                for item in items
            )
    return {
        class_id: float(evaluate_class(predictions.get(class_id, []), images, 0.5)["ap50"])
        for class_id, images in ground_truth.items()
        if images
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="配对 bootstrap 显著性检验")
    parser.add_argument("--baseline", default="runs/acdc_benchmark/evaluation/weather_input.json")
    parser.add_argument("--method", required=True)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="结果 JSON 输出路径（可选）")
    args = parser.parse_args()

    base_table = load_per_image(ROOT / args.baseline)
    method_table = load_per_image(ROOT / args.method)
    keys = sorted(set(base_table) & set(method_table))
    if len(keys) < 20:
        raise SystemExit(f"可配对图像过少（{len(keys)} 张），请检查 image_id 是否一致")

    base_items = [base_table[key] for key in keys]
    method_items = [method_table[key] for key in keys]

    point_base = dataset_map50(base_items)
    point_method = dataset_map50(method_items)

    rng = np.random.default_rng(args.seed)
    n = len(keys)
    diffs = np.empty(args.iters)
    base_maps = np.empty(args.iters)
    method_maps = np.empty(args.iters)
    for index in range(args.iters):
        sample = rng.integers(0, n, n)
        base_map = dataset_map50([base_items[i] for i in sample])["map50"]
        method_map = dataset_map50([method_items[i] for i in sample])["map50"]
        base_maps[index] = base_map
        method_maps[index] = method_map
        diffs[index] = method_map - base_map

    low, high = np.percentile(diffs, [2.5, 97.5])
    delta = point_method["map50"] - point_base["map50"]

    print(f"配对数：{n} 张，重采样 {args.iters} 次\n")
    print("| 方法 | mAP50 | 每图虚警 |")
    print("|---|---:|---:|")
    print(f"| 基线 | {point_base['map50']:.4f} | {point_base['false_positives_per_image']:.3f} |")
    print(f"| 本方法 | {point_method['map50']:.4f} | {point_method['false_positives_per_image']:.3f} |")
    print()
    print(f"mAP50 差值：{delta:+.4f}")
    print(f"bootstrap 95% CI：[{low:+.4f}, {high:+.4f}]")
    print(f"P(差值 > 0) = {(diffs > 0).mean():.3f}")
    if low > 0 or high < 0:
        verdict = "差异显著（置信区间不含 0）"
    else:
        verdict = "差异不显著（置信区间包含 0），不得表述为提升/下降"
    print(f"结论：{verdict}")

    base_classes = per_class_ap(base_items)
    method_classes = per_class_ap(method_items)
    print("\n逐类 AP50（点估计）：")
    print("| 类别 | 基线 | 本方法 | 差值 |")
    print("|---|---:|---:|---:|")
    for class_id in sorted(base_classes):
        name = CLASS_NAMES.get(class_id, str(class_id))
        print(
            f"| {name} | {base_classes[class_id]:.4f} | "
            f"{method_classes.get(class_id, 0.0):.4f} | "
            f"{method_classes.get(class_id, 0.0) - base_classes[class_id]:+.4f} |"
        )

    if args.out:
        out_path = ROOT / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "n_pairs": n,
                    "iters": args.iters,
                    "baseline_map50": point_base["map50"],
                    "method_map50": point_method["map50"],
                    "delta": delta,
                    "ci95": [float(low), float(high)],
                    "prob_positive": float((diffs > 0).mean()),
                    "significant": bool(low > 0 or high < 0),
                    "per_class": {
                        CLASS_NAMES.get(c, str(c)): {
                            "baseline": base_classes[c],
                            "method": method_classes.get(c, 0.0),
                        }
                        for c in sorted(base_classes)
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n结果已写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
