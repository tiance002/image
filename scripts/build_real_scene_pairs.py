"""构建「真实场景自监督」训练对，把 night/snow 纳入训练。

问题背景：
v3 只在 fog/rain 上训练，night 上把画面亮度从 0.1 提到 0.2、饱和度从 0.56 砍到 0.26，
等于把夜晚"洗成白天"，406 张全天气口径因此显著为负（-0.0145）。
根因是训练数据里所有目标都是明亮饱和的白天画面，模型学到"暗=欠曝，要提亮"。

为什么不能直接用 ACDC 的 night_ref 当清晰目标：
实测（runs/pair_check/night_degraded_vs_ref.png）night/snow 的 ref 帧是**白天画面**，
拿它当目标，模型会学着把夜晚还原成白天，比现在更糟。

本脚本的思路：自监督（self-supervised）——不需要"清晰目标"，只需要"退化程度可分"。
  输入 = 真实退化帧 再叠加一层**像素对齐的**额外退化
  目标 = 真实退化帧 本身
这样：
  1. 目标的明暗/色彩分布与真实夜间完全一致，模型不会学会"提亮"；
  2. 输入比目标多了一层已知退化，模型仍能学到"移除可辨伪影"的能力；
  3. 像素严格对齐（不用 wet_road 那类位移），监督信号干净。

只用于 night/snow（fog/rain 已有真正的合成退化→清晰图配对，不必混用自监督）。
额外退化必须像素对齐，因此刻意避开会移动像素的湿路反射。
"""

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
ACDC = ROOT / "data/raw/ACDC_images/rgb_anon"
OUT = ROOT / "data/real_scene_rgb_v2"
CALIB = ROOT / "data/real_weather/calibration/night_snow.json"


def imread(path: Path):
    """读图（走字节流，避开中文路径导致的静默失败）。"""
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite(path: Path, img: np.ndarray, quality: int = 95, size=None) -> bool:
    """将 RGB uint8 编码为 JPEG；size 保留旧版先编码再缩图的压缩链路。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return False
    if size is not None and (img.shape[1], img.shape[0]) != tuple(size):
        decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        small = cv2.resize(decoded, tuple(size), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return False
    buf.tofile(str(path))
    return True


# ---------------------------------------------------------------- 额外退化


def extra_night_degrade(img01: np.ndarray, rng: random.Random) -> np.ndarray:
    """在真实夜拍上再叠一层夜间退化（像素对齐）。

    只做"可逆且对齐"的变换：
      - 轻微进一步压暗（gamma）
      - 追加光源光晕（加性，不移动原有像素）
      - 追加暗部噪声
    """
    h, w = img01.shape[:2]
    out = np.clip(img01, 0, 1)

    gamma = rng.uniform(1.05, 1.30)
    out = np.clip(out**gamma, 0, 1)

    bloom = np.zeros((h, w), dtype=np.float32)
    for _ in range(rng.randint(2, 6)):
        cx = rng.uniform(0.05 * w, 0.95 * w)
        cy = rng.uniform(0.05 * h, 0.70 * h)
        radius = rng.uniform(0.03 * w, 0.12 * w)
        peak = rng.uniform(0.25, 0.7)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        d2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / max(radius**2, 1.0)
        bloom += (peak * np.exp(-d2)).astype(np.float32)
    bloom = np.clip(bloom, 0, 1)
    out = np.clip(out + bloom[:, :, None] * rng.uniform(0.2, 0.5), 0, 1)

    noise = np.random.default_rng(rng.randrange(1 << 30)).normal(
        0.0, 1.0, size=out.shape
    ).astype(np.float32)
    lum = out.max(axis=2)
    weight = (1.0 - np.clip(lum, 0, 1)) * 0.7 + 0.3
    out = np.clip(out + noise * rng.uniform(0.010, 0.028) * weight[:, :, None], 0, 1)
    return out.astype(np.float32)


def extra_snow_degrade(img01: np.ndarray, rng: random.Random) -> np.ndarray:
    """在真实雪景上再叠一层雪天退化（像素对齐）。

    强散射 + 雪花层 + 轻噪声，全部加性或逐像素变换，不移动像素。
    """
    h, w = img01.shape[:2]
    out = np.clip(img01, 0, 1)

    # 强散射：大气光接近白，模拟雪幕
    air = np.float32([0.90, 0.92, 0.94])
    strength = rng.uniform(0.35, 0.75)
    v = np.linspace(1.0, 0.0, h, dtype=np.float32)[:, None]
    depth = np.repeat(0.20 + 0.80 * v**1.4, w, axis=1)
    beta = strength / max(float(np.median(depth)), 1e-3)
    trans = np.exp(-beta * depth)[:, :, None]
    out = np.clip(out * trans + air[None, None, :] * (1.0 - trans), 0, 1)

    # 雪花层
    layer = np.zeros((h, w), dtype=np.float32)
    for _ in range(rng.randint(600, 1800)):
        x = rng.uniform(0, w)
        y = rng.uniform(0, h)
        size = rng.choice([1, 1, 2, 2, 3])
        alpha = rng.uniform(0.3, 0.9)
        cv2.circle(layer, (int(x), int(y)), size, float(alpha), -1)
    layer = cv2.GaussianBlur(layer, (3, 3), 0.7)
    out = np.clip(out * (1.0 - layer[:, :, None] * 0.30) + layer[:, :, None] * 0.95, 0, 1)

    # 轻微去饱和（雪幕压制色彩）
    gray = out.mean(axis=2, keepdims=True)
    desat = rng.uniform(0.15, 0.40)
    out = np.clip(out * desat + gray * (1.0 - desat), 0, 1)

    noise = np.random.default_rng(rng.randrange(1 << 30)).normal(
        0.0, 1.0, size=out.shape
    ).astype(np.float32)
    out = np.clip(out + noise * rng.uniform(0.008, 0.022), 0, 1)
    return out.astype(np.float32)


DEGRADERS = {"night": extra_night_degrade, "snow": extra_snow_degrade}


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    OUT.relative_to(ROOT)  # 清单使用项目相对路径。
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"拒绝覆盖已有数据目录: {OUT}")
    rng = random.Random(20261025)
    rows: list[dict[str, str]] = []

    for weather in ("night", "snow"):
        degrade = DEGRADERS[weather]
        # 用 train 划分的真实退化帧；val 划分绝不进训练
        files = sorted((ACDC / weather / "train").rglob("*_rgb_anon.png"))
        print("[%s] 真实退化帧 %d 张" % (weather, len(files)))
        for path in files:
            bgr = imread(path)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            # 目标 = 真实帧本身（代表"这一帧已经足够好"）
            # 输入 = 真实帧 + 额外退化（代表"更差的同场景帧"）
            degraded = degrade(rgb, rng)

            scene = path.stem.replace("_rgb_anon", "")
            seq = path.parent.name
            tag = f"{scene}__selfsup{weather}"
            clean_path = OUT / "target" / weather / f"{tag}.jpg"
            deg_path = OUT / "input" / weather / f"{tag}.jpg"
            if not imwrite(clean_path, (rgb * 255).astype(np.uint8), size=(960, 540)):
                raise OSError(clean_path)
            if not imwrite(deg_path, (np.clip(degraded, 0, 1) * 255).astype(np.uint8), size=(960, 540)):
                raise OSError(deg_path)
            rows.append({
                "degraded_path": str(deg_path.relative_to(ROOT)).replace("\\", "/"),
                "clean_path": str(clean_path.relative_to(ROOT)).replace("\\", "/"),
                "scene_id": tag,
                "weather": f"{weather}_selfsup",
                "sequence": seq,
            })
        print("   已生成 %d 对" % sum(1 for r in rows if r["weather"].startswith(weather)))

    out_csv = OUT / "manifests/train.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["degraded_path", "clean_path", "scene_id", "weather", "sequence"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print("清单已写入 %s（%d 对）" % (out_csv, len(rows)))
    (OUT / "color_contract.json").write_text(json.dumps({
        "version": "rgb_v2", "synthesis_color_order": "RGB", "opencv_encoding": "RGB2BGR",
        "output_size": [960, 540], "seed": 20261025,
        "jpeg_pipeline": "full-resolution JPEG95 decode→INTER_AREA960x540→JPEG95 (same compression order as history)",
        "pairs": len(rows), "manifest_sha256": hashlib.sha256(out_csv.read_bytes()).hexdigest(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
