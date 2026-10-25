"""为 night / snow 两种天气标定物理退化参数。

标定思路（与 fog 的暗通道反演同理，但机理不同）：
- night：退化不是散射雾，而是「低照度 + 光源眩光 + 暗部噪声」。
  因此统计量取：亮度分布分位数（暗化程度）、亮部占比（眩光强度）、
  暗部像素的高频能量（噪声强度）。
- snow：退化是「强散射雾 + 雪花层 + 地面积雪反光」。
  因此统计量取：暗通道反演出的 (A, s)（散射强度）+ 亮部占比（积雪）。

标定的目标是得到"真实退化帧"的统计画像，
后续用同一套统计量去合成退化，使训练分布贴近真实分布。
"""

import json
from pathlib import Path

import numpy as np
import cv2

ROOT = Path("data/raw/ACDC_images/rgb_anon")
OUT = Path("data/real_weather/calibration")
OUT.mkdir(parents=True, exist_ok=True)


def imread(p: Path):
    """读图（走字节流，避开中文路径导致的静默失败）。"""
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def lum_quantiles(bgr01: np.ndarray) -> dict[str, float]:
    """亮度分位数 + 暗部占比 + 饱和度。"""
    lab = cv2.cvtColor(np.clip(bgr01, 0, 1).astype(np.float32), cv2.COLOR_BGR2LAB)
    lum = lab[:, :, 0] / 100.0
    mx = bgr01.max(axis=2)
    mn = bgr01.min(axis=2)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return {
        "lum_p05": float(np.percentile(lum, 5)),
        "lum_p25": float(np.percentile(lum, 25)),
        "lum_p50": float(np.percentile(lum, 50)),
        "lum_p75": float(np.percentile(lum, 75)),
        "lum_p95": float(np.percentile(lum, 95)),
        "lum_max": float(lum.max()),
        "dark_ratio": float((lum < 0.15).mean()),
        "bright_ratio": float((lum > 0.75).mean()),
        "sat_mean": float(sat.mean()),
        "mean": float(lum.mean()),
    }


def noise_level(bgr01: np.ndarray) -> float:
    """用拉普拉斯响应估计噪声水平（MAD 法，抗结构干扰）。"""
    gray = cv2.cvtColor(np.clip(bgr01, 0, 1).astype(np.float32), cv2.COLOR_BGR2GRAY)
    resp = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    return float(np.median(np.abs(resp)) * 1.4826)


def inverse_fog(bgr01: np.ndarray):
    """暗通道先验反演：(大气光 A, 等效散射强度 s)。"""
    dark = bgr01.min(axis=2)
    flat = dark.ravel()
    k = max(1, int(0.001 * flat.size))
    idx = np.argpartition(flat, -k)[-k:]
    air = bgr01.reshape(-1, 3)[idx].mean(axis=0)
    air = np.clip(air, 0.45, 1.0)
    norm = bgr01 / np.maximum(air[None, None, :], 1e-3)
    trans = np.clip(1.0 - 0.95 * norm.min(axis=2), 0.03, 1.0)
    return air.astype(np.float32), float(np.median(-np.log(trans)))


def main() -> None:
    result = {}
    for weather in ("night", "snow"):
        files = sorted((ROOT / weather / "train").rglob("*_rgb_anon.png"))
        rows = []
        for path in files:
            img = imread(path)
            if img is None:
                continue
            small = cv2.resize(img, (480, 270), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            entry = lum_quantiles(small)
            entry["noise"] = round(noise_level(small), 5)
            if weather == "snow":
                air, strength = inverse_fog(small)
                entry["air"] = [round(float(v), 4) for v in air]
                entry["strength"] = round(strength, 4)
            rows.append(entry)

        agg = {}
        for k in rows[0]:
            if k == "air":
                agg[k] = [round(float(np.mean([r[k][i] for r in rows])), 4) for i in range(3)]
                continue
            vals = np.array([r[k] for r in rows], dtype=np.float32)
            agg[k] = {
                "p10": round(float(np.percentile(vals, 10)), 4),
                "p50": round(float(np.percentile(vals, 50)), 4),
                "p90": round(float(np.percentile(vals, 90)), 4),
                "mean": round(float(vals.mean()), 4),
            }
        if weather == "snow":
            strengths = np.array([r["strength"] for r in rows], dtype=np.float32)
            agg["strength"] = {
                "p10": round(float(np.percentile(strengths, 10)), 4),
                "p50": round(float(np.percentile(strengths, 50)), 4),
                "p90": round(float(np.percentile(strengths, 90)), 4),
            }
        result[weather] = agg
        print("\n=== %s 标定 (n=%d) ===" % (weather, len(rows)))
        for k, v in agg.items():
            if isinstance(v, dict):
                print("  %-14s p10=%-8.4f p50=%-8.4f p90=%-8.4f"
                      % (k, v["p10"], v["p50"], v["p90"]))
            else:
                print("  %-14s %s" % (k, v))

    out = OUT / "night_snow.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n标定表已写入 %s" % out)


if __name__ == "__main__":
    main()
