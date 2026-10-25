"""核查 ACDC train/val 的场景重叠是否真的污染了评测。

背景：ACDC 官方按"序列"划分不严格——fog 的 val 3 个序列在 train 里也出现过，
snow 6/8 重叠，rain 5/8 重叠。如果我们的训练数据见过 val 场景，
评测数字就偏乐观。

关键判断：我们的训练输入是 ref 帧(白天画面)经过合成退化得到的，
ref 帧与 val 的退化帧不是同一时刻、不是像素对齐（已实测梯度相关仅 0.06~0.29），
所以"场景见过"不等于"像素见过"。本脚本量化：
1. 重叠序列的具体范围；
2. 重叠场景里，训练用的 ref 帧与 val 的退化帧在结构上有多接近（梯度相关）；
3. 与非重叠场景做对照——如果两组梯度相关没差别，说明重叠没有带来额外信息泄漏。
"""

import numpy as np
import cv2
from pathlib import Path

ROOT = Path("data/raw/ACDC_images/rgb_anon")


def imread(p: Path):
    """读图（走字节流，避开中文路径导致的静默失败）。"""
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def grad_corr(a: np.ndarray, b: np.ndarray) -> float:
    """梯度幅值图相关性：1=结构完全一致，0=毫无关系。"""
    a = cv2.resize(a.astype(np.float32) / 255.0, (160, 90), interpolation=cv2.INTER_AREA)
    b = cv2.resize(b.astype(np.float32) / 255.0, (160, 90), interpolation=cv2.INTER_AREA)
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    kx = np.array([[-1, 0, 1]], np.float32)
    ky = kx.T
    ma = np.sqrt(cv2.filter2D(ga, -1, kx) ** 2 + cv2.filter2D(ga, -1, ky) ** 2)
    mb = np.sqrt(cv2.filter2D(gb, -1, kx) ** 2 + cv2.filter2D(gb, -1, ky) ** 2)
    va, vb = ma - ma.mean(), mb - mb.mean()
    denom = np.sqrt((va**2).sum() * (vb**2).sum())
    return float((va * vb).sum() / max(float(denom), 1e-9))


def seq_of(p: Path) -> str:
    """取父目录名作为序列标识。"""
    return p.parent.name


def main() -> None:
    for weather in ("fog", "rain", "night", "snow"):
        val_files = sorted((ROOT / weather / "val").rglob("*_rgb_anon.png"))
        ref_files = sorted((ROOT / weather / "train_ref").rglob("*_rgb_ref_anon.png"))
        train_seq = {seq_of(p) for p in
                     sorted((ROOT / weather / "train").rglob("*_rgb_anon.png"))}

        overlap = [p for p in val_files if seq_of(p) in train_seq]
        disjoint = [p for p in val_files if seq_of(p) not in train_seq]
        print("\n=== %s === val %d 张：重叠序列 %d 张，非重叠 %d 张"
              % (weather, len(val_files), len(overlap), len(disjoint)))

        # 训练用的是 ref 帧，看它与 val 退化帧的结构接近程度
        for label, group in (("重叠", overlap), ("非重叠", disjoint)):
            if not group:
                continue
            cors = []
            for vp in group[:12]:
                seq = seq_of(vp)
                cand = [r for r in ref_files if seq_of(r) == seq]
                if not cand:
                    continue
                d = imread(vp)
                r = imread(cand[0])
                if d is None or r is None or d.shape != r.shape:
                    continue
                cors.append(grad_corr(d, r))
            if cors:
                print("  %-6s: 与训练ref帧的梯度相关 %.3f  (n=%d)"
                      % (label, np.mean(cors), len(cors)))


if __name__ == "__main__":
    main()
