"""校验自监督训练对的质量。

必须同时满足两条，缺一不可：
1. 输入比目标"更差"（更暗/噪声更大/更模糊）——否则监督信号为空；
2. 目标的统计分布与真实 night/snow 一致——否则模型会学错先验。
"""

import csv
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(".")


def imread(p: Path):
    """读图（走字节流，避开中文路径导致的静默失败）。"""
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def stats(bgr01: np.ndarray) -> tuple[float, float, float, float]:
    """返回 (中位亮度, 对比度, 饱和度, 噪声)。"""
    lab = cv2.cvtColor(np.clip(bgr01, 0, 1).astype(np.float32), cv2.COLOR_BGR2LAB)
    lum = lab[:, :, 0] / 100.0
    mx = bgr01.max(axis=2)
    mn = bgr01.min(axis=2)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    gray = cv2.cvtColor(np.clip(bgr01, 0, 1).astype(np.float32), cv2.COLOR_BGR2GRAY)
    resp = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    noise = float(np.median(np.abs(resp)) * 1.4826)
    return float(np.median(lum)), float(lum.std()), float(sat.mean()), noise


def main() -> None:
    rows = list(csv.DictReader(open(ROOT / "data/real_scene/manifests/train.csv", encoding="utf-8")))
    print("共 %d 对\n" % len(rows))

    for weather in ("night", "snow"):
        sel = [r for r in rows if weather in r["weather"]]
        print("=== %s (%d 对) ===" % (weather, len(sel)))
        inp, tgt = [], []
        for r in sel[:60]:
            a = imread(ROOT / r["degraded_path"])
            b = imread(ROOT / r["clean_path"])
            if a is None or b is None:
                continue
            inp.append(stats(a.astype(np.float32) / 255.0))
            tgt.append(stats(b.astype(np.float32) / 255.0))
        ai = np.array(inp)
        at = np.array(tgt)
        names = ["中位亮度", "对比度", "饱和度", "噪声"]
        for i, nm in enumerate(names):
            print("  %-8s 输入 %.4f -> 目标 %.4f  (差 %+.4f)"
                  % (nm, ai[:, i].mean(), at[:, i].mean(), at[:, i].mean() - ai[:, i].mean()))
        print()

    # 与真实分布对照：目标的统计应贴近 ACDC 真实 night/snow
    calib = __import__("json").loads(
        (ROOT / "data/real_weather/calibration/night_snow.json").read_text(encoding="utf-8")
    )
    print("=== 目标 vs 真实分布标定值 ===")
    for weather in ("night", "snow"):
        sel = [r for r in rows if weather in r["weather"]][:60]
        tgt = np.array([stats(imread(ROOT / r["clean_path"]).astype(np.float32) / 255.0)
                        for r in sel if imread(ROOT / r["clean_path"]) is not None])
        c = calib[weather]
        print("%s:" % weather)
        print("  中位亮度  目标 %.4f   真实 p50 %.4f" % (np.median(tgt[:, 0]), c["lum_p50"]["p50"]))
        print("  暗部占比参照: 真实 dark_ratio p50 %.4f" % c["dark_ratio"]["p50"])
        print("  噪声      目标 %.4f   真实 p50 %.4f" % (tgt[:, 3].mean(), c["noise"]["p50"]))


if __name__ == "__main__":
    main()
