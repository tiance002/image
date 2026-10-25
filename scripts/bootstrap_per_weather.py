"""按天气做配对 bootstrap，判断分天气差值是否只是抽样噪声。

为什么需要它
------------
整体 406 张的 bootstrap 只能回答"总体上复原有没有伤检测"，
但 D1/D2 的红线是**按天气**的（"任一单天气不得出现超过 3 个百分点的 mAP 下降"）。
而且 ACDC val 只有 406 张，其中 rain 100 张、每类真值框数差异极大
（rain 上 bus 只有 31 框、motorcycle 只有 11 框），
单类 AP 在这种样本量下的抽样噪声可以达到数个百分点。

所以必须逐天气做重采样，否则很容易把"抽到了几张难图"误读成"方法退化"。

做法与 bootstrap_significance.py 相同（配对重采样），
只是把重采样范围限制在单个天气内部。

用法：
    PYTHONPATH=src;. python scripts/bootstrap_per_weather.py \
        --baseline runs/acdc_benchmark/evaluation/weather_input_direct_406.json \
        --method runs/real_v4_allweather/preds_043999.json
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

from scripts.bootstrap_significance import load_per_image  # noqa: E402
from scripts.diagnose_detection_rootcause import dataset_map50  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="逐天气配对 bootstrap")
    parser.add_argument(
        "--baseline", default="runs/acdc_benchmark/evaluation/weather_input_direct_406.json"
    )
    parser.add_argument("--method", required=True)
    parser.add_argument(
        "--records", default="runs/acdc_benchmark/evaluation/weather_input_all406.json"
    )
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    base_table = load_per_image(ROOT / args.baseline)
    method_table = load_per_image(ROOT / args.method)
    records = json.loads((ROOT / args.records).read_text(encoding="utf-8"))["per_image"]
    weather_of = {str(rec.get("file_name") or rec.get("image_id")): rec.get("weather", "?")
                  for rec in records}

    keys = sorted(set(base_table) & set(method_table))
    print("配对图像数：%d 张，重采样 %d 次\n" % (len(keys), args.iters))

    rng = np.random.default_rng(args.seed)
    result = {"n_pairs": len(keys), "iters": args.iters, "per_weather": {}}

    # 总体
    base_items = [base_table[k] for k in keys]
    method_items = [method_table[k] for k in keys]
    diffs = np.empty(args.iters)
    n = len(keys)
    for index in range(args.iters):
        sample = rng.integers(0, n, n)
        diffs[index] = (
            dataset_map50([method_items[i] for i in sample])["map50"]
            - dataset_map50([base_items[i] for i in sample])["map50"]
        )
    low, high = np.percentile(diffs, [2.5, 97.5])
    delta = dataset_map50(method_items)["map50"] - dataset_map50(base_items)["map50"]
    result["overall"] = {
        "baseline_map50": dataset_map50(base_items)["map50"],
        "method_map50": dataset_map50(method_items)["map50"],
        "delta": delta,
        "ci95": [float(low), float(high)],
        "prob_positive": float((diffs > 0).mean()),
    }
    print("== 总体 ==")
    print("  基线 %.4f  本方法 %.4f  差值 %+.4f  CI [%+.4f, %+.4f]  P(>0)=%.3f"
          % (result["overall"]["baseline_map50"], result["overall"]["method_map50"],
             delta, low, high, result["overall"]["prob_positive"]))

    # 逐天气
    for weather in ("fog", "rain", "night", "snow"):
        sub = [k for k in keys if weather_of.get(k) == weather]
        if len(sub) < 10:
            continue
        b_items = [base_table[k] for k in sub]
        m_items = [method_table[k] for k in sub]
        m = len(sub)
        w_diffs = np.empty(args.iters)
        for index in range(args.iters):
            sample = rng.integers(0, m, m)
            w_diffs[index] = (
                dataset_map50([m_items[i] for i in sample])["map50"]
                - dataset_map50([b_items[i] for i in sample])["map50"]
            )
        w_low, w_high = np.percentile(w_diffs, [2.5, 97.5])
        w_delta = dataset_map50(m_items)["map50"] - dataset_map50(b_items)["map50"]
        result["per_weather"][weather] = {
            "n_images": m,
            "baseline_map50": dataset_map50(b_items)["map50"],
            "method_map50": dataset_map50(m_items)["map50"],
            "delta": w_delta,
            "ci95": [float(w_low), float(w_high)],
            "prob_positive": float((w_diffs > 0).mean()),
        }
        entry = result["per_weather"][weather]
        verdict = "显著" if (w_low > 0 or w_high < 0) else "不显著(CI跨0)"
        print("== %s (n=%d) ==" % (weather, m))
        print("  基线 %.4f  本方法 %.4f  差值 %+.4f (%.2fpp)  CI [%+.4f, %+.4f]  P(>0)=%.3f  -> %s"
              % (entry["baseline_map50"], entry["method_map50"], w_delta, w_delta * 100,
                 w_low, w_high, entry["prob_positive"], verdict))

    if args.out:
        out_path = ROOT / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n结果已写入 %s" % out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
