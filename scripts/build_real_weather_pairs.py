"""用 ACDC 真实清晰参考帧构建"物理标定"配对数据集。

背景：此前训练集是 PIL 画的 192x128 卡通道路图（54 张），与 ACDC 真实
行车场景完全不是一个分布，导致模型在真实雨雾上学到的是"把卡通图变蓝天"，
用到真实数据上就猛压暗、猛加饱和，把目标沉进暗部（诊断见
docs/ROOTCAUSE_AND_PLAN_2026-10-24.html）。

本脚本改为：在 ACDC train_ref 的 800 张 1920x1080 真实清晰帧上做物理退化。

标定方式（只用 ACDC train 划分，不碰 val，避免泄漏）：
  * 雾：逐图用暗通道先验反演大气光 A 与等效散射强度 s = median(-ln t)，
        合成时按 I = J*t + A(1-t), t = exp(-beta*d) 复现同样的 s。
  * 雨：ACDC 的雨天主要是"阴天灰蒙 + 湿滑反光 + 镜头水珠"，不是暴雨条纹。
        逐图统计真实雨天图像的分通道均值/方差，合成时做 Reinhard 统计迁移，
        保证天空是灰的、饱和度是低的。

用法：
    # 默认复用历史 BGR 标定表，输出到新版本目录，不覆盖旧训练数据。
    python scripts/build_real_weather_pairs.py --split train --per-image 3
    python scripts/build_real_weather_pairs.py --split dev --per-image 1
    # 重新标定必须显式选择一个不存在的新文件。
    python scripts/build_real_weather_pairs.py --calibrate --calibration data/real_weather_rgb_v2/calibration_new.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ACDC = ROOT / "data/raw/ACDC_images/rgb_anon"
OUT = ROOT / "data/real_weather_rgb_v2"
# 历史标定表按 OpenCV BGR 顺序保存；在合成入口显式转为 RGB，不改写历史文件。
CALIB = ROOT / "data/real_weather/calibration.json"

OUT_W, OUT_H = 960, 540


# ---------------------------------------------------------------- 中文路径安全 IO
# 项目路径含中文，cv2.imread/imwrite 在 Windows 下会静默失败，必须走字节流。


def imread(path: Path) -> np.ndarray | None:
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def imwrite(path: Path, img: np.ndarray, quality: int = 95) -> bool:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return False
    buf.tofile(str(path))
    return True


# ---------------------------------------------------------------- 统计与反演


def basic_stats(bgr01: np.ndarray) -> dict[str, float]:
    """亮度(0~1)/饱和度/暗部比例。OpenCV 对 float32 输入的 L 量纲是 0~100。"""
    lab = cv2.cvtColor(np.clip(bgr01, 0, 1).astype(np.float32), cv2.COLOR_BGR2LAB)
    lum = lab[:, :, 0] / 100.0
    mx = bgr01.max(axis=2)
    mn = bgr01.min(axis=2)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return {
        "mean": float(lum.mean()),
        "std": float(lum.std()),
        "sat": float(sat.mean()),
        "dark": float((lum < 0.15).mean()),
    }


def inverse_fog(bgr01: np.ndarray) -> tuple[np.ndarray, float]:
    """暗通道先验反演：(大气光 A, 等效散射强度 s = median(-ln t))。"""
    dark = bgr01.min(axis=2)
    flat = dark.ravel()
    k = max(1, int(0.001 * flat.size))
    idx = np.argpartition(flat, -k)[-k:]
    air = bgr01.reshape(-1, 3)[idx].mean(axis=0)
    air = np.clip(air, 0.45, 1.0)
    norm = bgr01 / np.maximum(air[None, None, :], 1e-3)
    trans = np.clip(1.0 - 0.95 * norm.min(axis=2), 0.03, 1.0)
    strength = float(np.median(-np.log(trans)))
    return air.astype(np.float32), strength


def channel_stats(bgr01: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """分通道均值与标准差，用于 Reinhard 统计迁移。"""
    flat = bgr01.reshape(-1, 3)
    return flat.mean(axis=0).astype(np.float32), (flat.std(axis=0) + 1e-6).astype(np.float32)


def build_calibration() -> dict:
    if CALIB.exists():
        raise FileExistsError(f"拒绝覆盖已有标定表，请用 --calibration 指定新路径: {CALIB}")
    payload: dict[str, list] = {"fog": [], "rain": []}
    for weather in ("fog", "rain"):
        files = sorted((ACDC / weather / "train").rglob("*.png"))
        for path in files:
            img = imread(path)
            if img is None:
                continue
            small = cv2.resize(img, (480, 270), interpolation=cv2.INTER_AREA)
            small = small.astype(np.float32) / 255.0
            entry: dict[str, object] = {"stats": basic_stats(small)}
            if weather == "fog":
                air, strength = inverse_fog(small)
                entry["air"] = [round(float(v), 4) for v in air]
                entry["strength"] = round(strength, 4)
            else:
                mean_c, std_c = channel_stats(small)
                entry["mean_c"] = [round(float(v), 4) for v in mean_c]
                entry["std_c"] = [round(float(v), 4) for v in std_c]
            payload[weather].append(entry)

        stats = [e["stats"] for e in payload[weather]]
        summary = {k: round(float(np.mean([s[k] for s in stats])), 4) for k in stats[0]}
        print(f"{weather}: n={len(payload[weather])} 视觉统计 {summary}")
        strengths = [e["strength"] for e in payload[weather] if "strength" in e]
        if strengths:
            arr = np.array(strengths, dtype=np.float32)
            print(
                "  雾参数 A≈%.3f  散射强度 s: p10=%.2f p50=%.2f p90=%.2f"
                % (
                    float(np.mean([e["air"][0] for e in payload[weather]])),
                    np.percentile(arr, 10),
                    np.percentile(arr, 50),
                    np.percentile(arr, 90),
                )
            )

    CALIB.parent.mkdir(parents=True, exist_ok=True)
    CALIB.write_text(json.dumps(payload), encoding="utf-8")
    print(f"标定表已写入 {CALIB}")
    return payload


# ---------------------------------------------------------------- 退化合成


def depth_proxy(h: int, w: int, rng: random.Random) -> np.ndarray:
    """行车场景深度先验：越靠上越远，叠加低频起伏，均值归一化为 1。"""
    v = np.linspace(1.0, 0.0, h, dtype=np.float32)[:, None]
    base = 0.20 + 0.80 * v**1.4
    base = np.repeat(base, w, axis=1)
    clutter = np.array(
        [[rng.random() for _ in range(7)] for _ in range(4)], dtype=np.float32
    )
    clutter = cv2.resize(clutter, (w, h), interpolation=cv2.INTER_CUBIC)
    clutter = (clutter - clutter.min()) / max(float(np.ptp(clutter)), 1e-6)
    d = base * (0.68 + 0.64 * clutter)
    return d / max(float(d.mean()), 1e-6)


def synth_fog(clean: np.ndarray, entry: dict, rng: random.Random) -> np.ndarray:
    """按反演出的 (A, s) 复现同等强度的大气散射。"""
    h, w = clean.shape[:2]
    depth = depth_proxy(h, w, rng)
    air = calibration_rgb(entry, "air") * rng.uniform(0.97, 1.03)
    air = np.clip(air, 0.5, 1.0)
    strength = float(entry["strength"]) * rng.uniform(0.75, 1.25)
    # median(-ln t) = beta * median(d)，令其等于目标 s
    beta = strength / max(float(np.median(depth)), 1e-3)
    trans = np.exp(-beta * depth)[:, :, None]
    return np.clip(clean * trans + air[None, None, :] * (1.0 - trans), 0, 1)


def render_rain_layer(h: int, w: int, rng: random.Random) -> np.ndarray:
    """雨丝层：远雨细短、近雨粗长，整体略微倾斜。"""
    layer = np.zeros((h, w), dtype=np.float32)
    for count, length, thick, alpha in (
        (int(rng.uniform(140, 260)), (12, 30), 1, 0.30),
        (int(rng.uniform(30, 80)), (32, 66), 2, 0.40),
    ):
        for _ in range(count):
            x = rng.uniform(-0.05 * w, 1.05 * w)
            y = rng.uniform(-0.1 * h, 1.1 * h)
            ln = rng.uniform(*length)
            ang = np.deg2rad(rng.uniform(-25, -11))
            cv2.line(
                layer,
                (int(x), int(y)),
                (int(x + ln * np.cos(ang)), int(y + ln * np.sin(ang))),
                float(alpha * rng.uniform(0.6, 1.0)),
                thick,
                cv2.LINE_AA,
            )
    return cv2.GaussianBlur(layer, (3, 3), 0.6)


def wet_road(clean: np.ndarray, rng: random.Random) -> np.ndarray:
    """湿滑路面：下部竖向拖影反射 + 轻微镜面提亮。"""
    h = clean.shape[0]
    out = clean.copy()
    horizon = int(h * rng.uniform(0.52, 0.68))
    region = out[horizon:]
    if region.size == 0:
        return out
    shifted = cv2.flip(region, 0)
    k = int(max(3, h * 0.045)) | 1
    shifted = cv2.GaussianBlur(shifted, (k, k), 0)
    alpha = rng.uniform(0.10, 0.28)
    out[horizon:] = region * (1 - alpha) + shifted * alpha
    glow = np.linspace(0.0, rng.uniform(0.02, 0.08), h, dtype=np.float32)[:, None, None]
    return np.clip(out + glow, 0, 1)


def lens_droplets(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """镜头水珠：少量半透明亮斑 + 局部模糊。"""
    h, w = img.shape[:2]
    out = img.copy()
    for _ in range(rng.randint(0, 5)):
        cx = rng.randint(int(0.05 * w), int(0.95 * w))
        cy = rng.randint(int(0.05 * h), int(0.95 * h))
        r = rng.randint(int(0.010 * w), int(0.040 * w))
        y0, y1 = max(0, cy - r), cy + r
        x0, x1 = max(0, cx - r), cx + r
        patch = out[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        blur = cv2.GaussianBlur(patch, (int(r * 0.9) | 1, int(r * 0.9) | 1), r * 0.35)
        bright = np.clip(blur * 1.06 + 0.05, 0, 1)
        mask = np.zeros(patch.shape[:2], dtype=np.float32)
        cv2.circle(mask, (patch.shape[1] // 2, patch.shape[0] // 2), r, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), r * 0.4)[:, :, None]
        out[y0:y1, x0:x1] = patch * (1 - mask) + bright * mask
    return out


def reinhard_transfer(
    img: np.ndarray, mean_c: np.ndarray, std_c: np.ndarray, weight: float
) -> np.ndarray:
    """把分通道均值/方差迁移到采样到的真实雨天图像上。

    直接迁移会发灰丢结构，所以按 weight 混合，保留一部分原始动态。
    """
    src_mean, src_std = channel_stats(img)
    transferred = (img - src_mean[None, None, :]) / src_std[None, None, :]
    transferred = transferred * std_c[None, None, :] + mean_c[None, None, :]
    out = img * (1 - weight) + transferred * weight
    return np.clip(out, 0, 1)


def synth_rain(clean: np.ndarray, entry: dict, rng: random.Random) -> np.ndarray:
    base = wet_road(clean, rng)
    h, w = base.shape[:2]
    streaks = render_rain_layer(h, w, rng)[:, :, None]
    with_streaks = np.clip(base * (1 - streaks * 0.5) + streaks * 0.9, 0, 1)
    # 阴天散射造成的低对比（veiling glare）
    k = int(max(3, min(h, w) * 0.02)) | 1
    veiled = cv2.GaussianBlur(with_streaks, (k, k), 0)
    out = with_streaks * 0.75 + veiled * 0.25

    mean_c = calibration_rgb(entry, "mean_c") * rng.uniform(0.97, 1.03)
    std_c = calibration_rgb(entry, "std_c") * rng.uniform(0.92, 1.08)
    out = reinhard_transfer(out, mean_c, std_c, weight=rng.uniform(0.75, 0.95))
    return np.clip(lens_droplets(out, rng), 0, 1)


# ---------------------------------------------------------------- 构建


def calibration_rgb(entry: dict, key: str) -> np.ndarray:
    """历史 calibration.json 的三通道参数是 BGR；合成图像使用 RGB。"""
    value = np.asarray(entry[key], dtype=np.float32)
    if value.shape != (3,):
        raise ValueError(f"标定 {key} 必须包含三个BGR通道")
    return value[::-1].copy()


def photometric_jitter(clean: np.ndarray, rng: random.Random) -> np.ndarray:
    """对清晰图本身做轻度光度抖动。

    否则"清晰图恒为晴天、退化图恒为灰蒙"会构成一个固定全局变换，
    模型只需学会这一个变换就能在训练集上收敛，泛化会很差。
    """
    gamma = rng.uniform(0.88, 1.14)
    gains = np.array([rng.uniform(0.96, 1.04) for _ in range(3)], dtype=np.float32)
    out = np.clip(clean, 0, 1) ** gamma
    return np.clip(out * gains[None, None, :], 0, 1)


def load_calibration() -> dict:
    if not CALIB.exists():
        return build_calibration()
    return json.loads(CALIB.read_text(encoding="utf-8"))


def build(split: str, per_image: int, seed: int) -> None:
    if (OUT / "manifests" / f"{split}.csv").exists() or any((OUT / name / split).exists() for name in ("clean", "degraded")):
        raise FileExistsError(f"拒绝覆盖已有数据划分: {OUT}/{split}")
    calib = load_calibration()
    rng = random.Random(seed)
    ref_split = "train_ref" if split == "train" else "val_ref"
    sources = [("fog", ref_split), ("rain", ref_split)]

    rows: list[dict[str, str]] = []
    clean_dir = OUT / "clean" / split
    deg_dir = OUT / "degraded" / split
    clean_dir.mkdir(parents=True, exist_ok=True)
    deg_dir.mkdir(parents=True, exist_ok=True)

    for weather, ref in sources:
        files = sorted((ACDC / weather / ref).rglob("*.png"))
        entries = calib[weather]
        print(f"[{split}] {weather}: {len(files)} 张清晰帧, {len(entries)} 条标定记录")
        for path in files:
            raw = imread(path)
            if raw is None:
                continue
            resized = cv2.resize(raw, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
            clean = resized.astype(np.float32)[:, :, ::-1] / 255.0  # BGR->RGB
            seq = path.parent.name
            stem = path.stem.replace("_rgb_ref_anon", "")
            clean_name = f"{seq}__{stem}.jpg"
            imwrite(
                clean_dir / clean_name,
                (np.clip(clean, 0, 1)[:, :, ::-1] * 255).astype(np.uint8),
                quality=97,
            )
            for k in range(per_image):
                entry = rng.choice(entries)
                target = clean
                degraded = (
                    synth_fog(target, entry, rng)
                    if weather == "fog"
                    else synth_rain(target, entry, rng)
                )
                deg_name = f"{seq}__{stem}__{weather}{k}.jpg"
                imwrite(
                    deg_dir / deg_name,
                    (np.clip(degraded, 0, 1)[:, :, ::-1] * 255).astype(np.uint8),
                    quality=95,
                )
                rows.append(
                    {
                        "degraded_path": str((deg_dir / deg_name).relative_to(ROOT)),
                        "clean_path": str((clean_dir / clean_name).relative_to(ROOT)),
                        "scene_id": f"{seq}__{stem}__{weather}{k}",
                        "weather": weather,
                        "sequence": seq,
                    }
                )

    manifest_dir = OUT / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_dir / f"{split}.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[{split}] 共 {len(rows)} 对 -> {manifest}")
    (OUT / f"color_contract_{split}.json").write_text(json.dumps({
        "version": "rgb_v2", "synthesis_color_order": "RGB", "calibration_color_order": "BGR",
        "conversion": "reverse air/mean_c/std_c at synthesis entry", "opencv_encoding": "RGB2BGR",
        "output_size": [OUT_W, OUT_H], "seed": seed, "per_image": per_image,
        "pairs": len(rows), "calibration_sha256": hashlib.sha256(CALIB.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    global OUT, CALIB
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--split", choices=["train", "dev"], default=None)
    parser.add_argument("--per-image", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--calibration", type=Path, default=CALIB)
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    OUT.relative_to(ROOT)
    CALIB = args.calibration.resolve()
    if args.calibrate:
        build_calibration()
    if args.split:
        build(args.split, args.per_image, args.seed)


if __name__ == "__main__":
    main()
