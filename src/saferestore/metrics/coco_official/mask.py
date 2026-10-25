"""COCO bbox IoU，兼容 crowd 以检测框面积作为分母的定义。"""

import numpy as np


def iou(detections, ground_truth, iscrowd):
    d = np.asarray(detections, dtype=np.float64).reshape(-1, 4)
    g = np.asarray(ground_truth, dtype=np.float64).reshape(-1, 4)
    if not len(d) or not len(g):
        return np.zeros((len(d), len(g)), dtype=np.float64)
    lo = np.maximum(d[:, None, :2], g[None, :, :2])
    hi = np.minimum(d[:, None, :2] + d[:, None, 2:], g[None, :, :2] + g[None, :, 2:])
    intersection = np.maximum(hi - lo, 0).prod(axis=2)
    da = np.maximum(d[:, 2:], 0).prod(axis=1)[:, None]
    ga = np.maximum(g[:, 2:], 0).prod(axis=1)[None, :]
    denominator = np.where(np.asarray(iscrowd, dtype=bool)[None, :], da, da + ga - intersection)
    return intersection / np.maximum(denominator, np.finfo(np.float64).eps)
