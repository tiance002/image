"""night / snow 的物理退化合成。

机理与 fog/rain 完全不同，不能套用同一套公式：

night（低照度 + 光源眩光 + 暗部噪声）
  1. 整体 gamma 压暗：out = clean ** gamma，gamma>1 让暗部更暗；
  2. 光源眩光：在若干随机位置叠加高斯亮斑，再做径向扩散（bloom），
     对应车灯/路灯在镜头与空气中的散射光晕；
  3. 传感器噪声：暗部按 (1-lum) 加权注入高斯噪声，暗部噪声更强；
  4. 冷色偏移：夜间路灯下画面偏暖，但天空/远景偏冷，这里做轻微的
     通道增益抖动而不是固定偏移，避免学到固定色偏。

snow（强散射雾 + 雪花层 + 地面积雪）
  1. 大气散射：沿用 fog 的 I = J*t + A(1-t)，A 与 s 取 snow 标定值；
  2. 雪花层：随机撒布亮斑，部分带拖尾（运动模糊的雪花）；
  3. 地面积雪：画面下部按"水平面掩膜"提亮并去饱和，模拟雪覆盖路面；
  4. 整体去饱和：雪天色彩被灰白散射压制。

标定来源：data/real_weather/calibration/night_snow.json
"""

import json
import random
from pathlib import Path

import numpy as np
import cv2


def load_night_snow_calib(path: Path) -> dict:
    """读取 night/snow 标定表。"""
    return json.loads(path.read_text(encoding="utf-8"))


def night_degrade(clean: np.ndarray, calib: dict, rng: random.Random) -> np.ndarray:
    """合成夜间退化图。

    calib 取该图的 p50 统计，rng 再做 ±25% 抖动以增加多样性。
    """
    h, w = clean.shape[:2]

    def pick(key: str, spread: float = 0.25) -> float:
        """取标定统计的中位数并施加随机抖动。"""
        base = calib[key]["p50"]
        return float(base) * rng.uniform(1.0 - spread, 1.0 + spread)

    # 1) 低照度：gamma 压暗。目标中位亮度约 0.165
    target_median = pick("lum_p50")
    src_median = float(np.median(clean))
    src_median = max(src_median, 1e-3)
    # gamma 使中位数按 (src_median**gamma) 落到目标附近
    gamma = float(np.log(max(target_median, 1e-3)) / np.log(src_median))
    gamma = float(np.clip(gamma * rng.uniform(0.92, 1.08), 1.6, 6.0))
    out = np.clip(clean, 0, 1) ** gamma

    # 2) 光源眩光：少量高亮光源 + 径向扩散
    bloom = np.zeros((h, w), dtype=np.float32)
    n_lights = rng.randint(3, 9)
    for _ in range(n_lights):
        cx = rng.uniform(0.05 * w, 0.95 * w)
        cy = rng.uniform(0.05 * h, 0.75 * h)
        radius = rng.uniform(0.03 * w, 0.14 * w)
        peak = rng.uniform(0.45, 1.0)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        d2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / max(radius**2, 1.0)
        bloom += (peak * np.exp(-d2)).astype(np.float32)
    bloom = np.clip(bloom, 0, 1)
    # 大范围辉光：模拟空气中的散射光晕
    halo = cv2.GaussianBlur(bloom, (0, 0), sigmaX=max(w * 0.04, 3))
    bloom = np.clip(bloom + halo * rng.uniform(0.3, 0.7), 0, 1)
    out = np.clip(out + bloom[:, :, None] * rng.uniform(0.55, 0.95), 0, 1)

    # 3) 传感器噪声：暗部噪声更强
    lum = out.max(axis=2)
    noise_sigma = pick("noise") * rng.uniform(0.7, 1.3)
    noise = rng.gauss(0.0, 1.0) * 0.0  # 占位，避免误用 random 的高斯
    noise = np.random.default_rng(rng.randrange(1 << 30)).normal(
        0.0, 1.0, size=out.shape
    ).astype(np.float32)
    weight = (1.0 - np.clip(lum, 0, 1)) * 0.8 + 0.2
    out = np.clip(out + noise * noise_sigma * weight[:, :, None], 0, 1)

    # 4) 通道增益抖动（轻微色温变化，不做固定偏移）
    gains = np.array([rng.uniform(0.95, 1.06) for _ in range(3)], dtype=np.float32)
    out = np.clip(out * gains[None, None, :], 0, 1)
    return out.astype(np.float32)


def snow_degrade(clean: np.ndarray, calib: dict, rng: random.Random) -> np.ndarray:
    """合成雪天退化图。"""
    h, w = clean.shape[:2]

    def pick(key: str, spread: float = 0.25) -> float:
        """取标定统计的中位数并施加随机抖动。"""
        base = calib[key]["p50"]
        return float(base) * rng.uniform(1.0 - spread, 1.0 + spread)

    # 1) 大气散射（强）：A 与 s 取 snow 标定
    air = np.asarray(calib["air"], dtype=np.float32) * rng.uniform(0.97, 1.03)
    air = np.clip(air, 0.5, 1.0)
    strength = pick("strength") * rng.uniform(0.85, 1.15)
    v = np.linspace(1.0, 0.0, h, dtype=np.float32)[:, None]
    depth = np.repeat(0.20 + 0.80 * v**1.4, w, axis=1)
    beta = strength / max(float(np.median(depth)), 1e-3)
    trans = np.exp(-beta * depth)[:, :, None]
    out = np.clip(clean * trans + air[None, None, :] * (1.0 - trans), 0, 1)

    # 2) 雪花层：亮斑，部分带拖尾
    layer = np.zeros((h, w), dtype=np.float32)
    n_flakes = rng.randint(900, 2600)
    for _ in range(n_flakes):
        x = rng.uniform(0, w)
        y = rng.uniform(0, h)
        size = rng.choice([1, 1, 2, 2, 3])
        alpha = rng.uniform(0.35, 0.95)
        if rng.random() < 0.35:
            # 带拖尾：模拟运动中的雪花
            length = rng.uniform(4, 16)
            cv2.line(layer, (int(x), int(y)),
                     (int(x + rng.uniform(-3, 3)), int(y + length)),
                     float(alpha), size, cv2.LINE_AA)
        else:
            cv2.circle(layer, (int(x), int(y)), size, float(alpha), -1)
    layer = cv2.GaussianBlur(layer, (3, 3), 0.7)
    out = np.clip(out * (1.0 - layer[:, :, None] * 0.35) + layer[:, :, None] * 0.95, 0, 1)

    # 3) 地面积雪：下部水平面掩膜提亮 + 去饱和
    ground = np.zeros((h, w), dtype=np.float32)
    horizon = int(h * rng.uniform(0.42, 0.62))
    soft = np.linspace(0.0, 1.0, h - horizon, dtype=np.float32)[:, None]
    ground[horizon:] = np.repeat(soft, w, axis=1)
    ground = cv2.GaussianBlur(ground, (0, 0), sigmaX=w * 0.01)
    ground = np.clip(ground * rng.uniform(0.55, 0.95), 0, 1)
    snow_white = np.array([0.93, 0.94, 0.95], dtype=np.float32)
    out = out * (1.0 - ground[:, :, None]) + snow_white[None, None, :] * ground[:, :, None]

    # 4) 整体去饱和（雪天色彩被灰白散射压制）
    gray = out.mean(axis=2, keepdims=True)
    desat = rng.uniform(0.55, 0.85)
    out = np.clip(out * desat + gray * (1.0 - desat), 0, 1)

    # 5) 轻微噪声（雪地反光高，噪声更明显）
    noise_sigma = pick("noise") * rng.uniform(0.4, 0.8)
    noise = np.random.default_rng(rng.randrange(1 << 30)).normal(
        0.0, 1.0, size=out.shape
    ).astype(np.float32)
    out = np.clip(out + noise * noise_sigma, 0, 1)
    return out.astype(np.float32)
