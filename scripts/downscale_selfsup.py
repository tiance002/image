"""把自监督训练对降采样到与原训练图一致的 960x540。

原因：
1. 原 fog/rain 训练图是 960x540，自监督图是 1920x1080。
   统一后模型的尺度先验与已有训练一致，也避免同一场景在不同尺度下出现两次。
2. 解码开销降到 1/4，DataLoader 不再是瓶颈。
3. 目标与输入同步缩放，像素对齐关系不受影响。
"""

import csv
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(".")
TARGET_W, TARGET_H = 960, 540


def imread(p: Path):
    """读图（走字节流，避开中文路径导致的静默失败）。"""
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite(p: Path, img: np.ndarray, quality: int = 95) -> bool:
    """写图（同样走字节流）。"""
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return False
    buf.tofile(str(p))
    return True


def main() -> None:
    rows = list(csv.DictReader(open(ROOT / "data/real_scene/manifests/train.csv", encoding="utf-8")))
    done = 0
    for r in rows:
        for key in ("degraded_path", "clean_path"):
            p = ROOT / r[key]
            img = imread(p)
            if img is None:
                print("跳过无法读取:", p)
                continue
            if img.shape[1] == TARGET_W and img.shape[0] == TARGET_H:
                continue
            small = cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
            imwrite(p, small)
        done += 1
        if done % 200 == 0:
            print("已处理 %d/%d" % (done, len(rows)))
    print("完成，共 %d 对已统一到 %dx%d" % (done, TARGET_W, TARGET_H))


if __name__ == "__main__":
    main()
