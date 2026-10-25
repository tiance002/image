"""导出 night/snow 的 ref 帧与退化帧对照图，肉眼确认 ref 帧到底是什么画面。

这一步决定 night/snow 能否纳入训练：
- 若 ref 是同场景的白天帧，拿它训练会把夜晚"洗白"，是错的；
- 若 ref 是同场景的无眩光/轻退化夜拍，才能作为清晰目标。
"""

import numpy as np
import cv2
from pathlib import Path

ROOT = Path("data/raw/ACDC_images/rgb_anon")
OUT = Path("runs/pair_check")
OUT.mkdir(parents=True, exist_ok=True)


def imread(p: Path):
    """读图（走字节流，避开中文路径导致的静默失败）。"""
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite(path: Path, img: np.ndarray) -> None:
    """写图（同样走字节流）。"""
    ok, buf = cv2.imencode(".png", img)
    if ok:
        buf.tofile(str(path))


def main() -> None:
    for w in ("night", "snow", "fog"):
        deg = {p.name.replace("_rgb_anon.png", ""): p
               for p in sorted((ROOT / w / "train").rglob("*_rgb_anon.png"))}
        ref = {p.name.replace("_rgb_ref_anon.png", ""): p
               for p in sorted((ROOT / w / "train_ref").rglob("*_rgb_ref_anon.png"))}
        common = sorted(set(deg) & set(ref))
        picks = common[:3]
        rows = []
        for k in picks:
            d = imread(deg[k])
            r = imread(ref[k])
            if d is None or r is None:
                continue
            d = cv2.resize(d, (480, 270))
            r = cv2.resize(r, (480, 270))
            rows.append(np.hstack([d, r]))
        if not rows:
            continue
        canvas = np.vstack(rows)
        # 左列标注：退化帧 / 右列标注：ref 帧
        for i in range(len(rows)):
            cv2.putText(canvas, "degraded", (8, 24 + i * 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(canvas, "ref", (488, 24 + i * 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        out = OUT / f"{w}_degraded_vs_ref.png"
        imwrite(out, canvas)
        print("%-6s -> %s   样本帧 %s" % (w, out, picks))


if __name__ == "__main__":
    main()
