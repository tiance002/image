"""检查 ACDC 的 ref 清晰帧与退化帧是否为像素对齐的真实配对。

ACDC 里 fog/rain/snow/night 都有 *_ref_anon 目录，文件名编号与退化帧一致。
如果两者是像素对齐的，就能直接用"真实退化 -> 真实清晰"配对训练，
不必再依赖合成退化；night/snow 也能顺势纳入训练。
"""

import numpy as np
import cv2
from pathlib import Path

ROOT = Path("data/raw/ACDC_images/rgb_anon")


def imread(p: Path):
    """读图（走字节流，避开中文路径导致的静默失败）。"""
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def grad_corr(a: np.ndarray, b: np.ndarray) -> float:
    """梯度幅值图的相关性：1 表示结构完全一致，0 表示毫无关系。"""
    a = cv2.resize(a.astype(np.float32) / 255.0, (320, 180), interpolation=cv2.INTER_AREA)
    b = cv2.resize(b.astype(np.float32) / 255.0, (320, 180), interpolation=cv2.INTER_AREA)
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    kx = np.array([[-1, 0, 1]], np.float32)
    ky = kx.T
    gxa, gya = cv2.filter2D(ga, -1, kx), cv2.filter2D(ga, -1, ky)
    gxb, gyb = cv2.filter2D(gb, -1, kx), cv2.filter2D(gb, -1, ky)
    ma = np.sqrt(gxa**2 + gya**2)
    mb = np.sqrt(gxb**2 + gyb**2)
    va, vb = ma - ma.mean(), mb - mb.mean()
    denom = np.sqrt((va**2).sum() * (vb**2).sum())
    return float((va * vb).sum() / max(float(denom), 1e-9))


def main() -> None:
    for w in ("fog", "rain", "night", "snow"):
        deg_files = sorted((ROOT / w / "train").rglob("*_rgb_anon.png"))
        ref_files = {
            p.name.replace("_rgb_ref_anon.png", ""): p
            for p in sorted((ROOT / w / "train_ref").rglob("*_rgb_ref_anon.png"))
        }
        deg_keys = {}
        for p in deg_files:
            deg_keys[p.name.replace("_rgb_anon.png", "")] = p
        common = sorted(set(deg_keys) & set(ref_files))
        print("\n=== %s === 退化帧 %d  ref帧 %d  编号可匹配 %d"
              % (w, len(deg_files), len(ref_files), len(common)))
        if not common:
            continue
        cors, mses, lums = [], [], []
        for k in common[:12]:
            d = imread(deg_keys[k])
            r = imread(ref_files[k])
            if d is None or r is None or d.shape != r.shape:
                continue
            cors.append(grad_corr(d, r))
            mses.append(float(np.mean((d.astype(np.float32) - r.astype(np.float32)) ** 2)))
            lums.append((float(d.mean()), float(r.mean())))
        if cors:
            print("  梯度相关: " + " ".join("%.3f" % c for c in cors))
            print("  平均梯度相关 %.3f   MSE %.0f" % (np.mean(cors), np.mean(mses)))
            print("  亮度(退化/清晰): " + " ".join("%.0f/%.0f" % pair for pair in lums[:6]))


if __name__ == "__main__":
    main()
