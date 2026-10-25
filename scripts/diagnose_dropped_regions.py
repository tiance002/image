"""诊断少数掉检框在「原图」与「复原图」之间的局部结构差异。

要回答的问题
------------
v4 相对原图直检，在 406 张全天气上只掉了 5 个检测框，但 mAP50 是类别等权平均，
这 5 个框集中在 bus / train / truck 上，就造成了 rain/night/snow 各掉近 1 个百分点。

这 5 个框到底是因为「目标被抹平了」还是「目标还在但特征被扰动」而掉检？
本脚本在每个掉检框的真值区域里，比较原图与复原图的：
  - 梯度能量（结构锐度）
  - 局部对比度
  - 亮度与饱和度
  - 区域内外亮度差（目标与背景的分离度，检测器最看重的信号之一）
从而判断复原到底把目标「洗掉」了，还是只是轻微扰动。
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(".")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from saferestore.metrics.detection import box_iou  # noqa: E402

RUN = ROOT / "runs/real_v4_allweather"


def imread(path: Path):
    """读图（走字节流，避开中文路径导致的静默失败）。"""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.float32) / 255.0


def grad_energy(rgb: np.ndarray) -> float:
    """Sobel 梯度幅值均值，衡量结构锐度。"""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.sqrt(gx * gx + gy * gy).mean())


def stats(rgb: np.ndarray) -> dict:
    """一组基础统计量。传入 (H,W,3) 或展平后的 (N,3) 都支持。"""
    flat = rgb.reshape(-1, rgb.shape[-1]) if rgb.ndim == 3 else rgb
    hsv = cv2.cvtColor(flat.reshape(1, -1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    gray = flat.mean(axis=1)
    gx = np.abs(np.diff(gray, append=gray[-1]))
    return {
        "lum": float(flat.mean()),
        "sat": float(hsv[:, 1].mean()),
        "grad": float(gx.mean()),
        "std": float(flat.std()),
    }


def region(rgb: np.ndarray, box, pad: int = 0):
    """裁出真值框区域（可外扩），返回该区域与其外圈背景。"""
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    inner = rgb[y1:y2, x1:x2]
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)
    outer = rgb[mask == 0]
    return inner, outer


def main() -> None:
    rec = {
        r["file_name"]: r
        for r in json.loads(
            (ROOT / "runs/acdc_benchmark/evaluation/weather_input_all406.json").read_text(encoding="utf-8")
        )["per_image"]
    }
    cases = json.loads((RUN / "dropped_cases.json").read_text(encoding="utf-8"))

    import torch

    from saferestore.models.nafnet import build_nafnet

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_nafnet("standard")
    state = torch.load(RUN / "best.pt", map_location="cpu")
    model.load_state_dict(state.get("ema", state), strict=True)
    model = model.to(device).eval()

    acdc = ROOT / "data/raw/ACDC_images/rgb_anon"
    rows = []
    for case in cases:
        name = case["file_name"]
        path = acdc / name
        bgr = imread(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        x = torch.from_numpy(rgb.transpose(2, 0, 1))[None].to(device)
        ph = (16 - h % 16) % 16
        pw = (16 - w % 16) % 16
        if ph or pw:
            x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
        with torch.inference_mode():
            restored = model(x).clamp(0, 1).cpu().numpy()[0].transpose(1, 2, 0)[:h, :w]

        gt = rec[name]["ground_truth"][str(case["class_id"])][case["gt_index"]]
        inner_a, outer_a = region(rgb, gt)
        inner_b, outer_b = region(restored, gt)

        s_in, s_out = stats(inner_a), stats(outer_a)
        r_in, r_out = stats(inner_b), stats(outer_b)
        rows.append(
            {
                "file": name,
                "weather": case["weather"],
                "cls": case["class"],
                "grad_ratio": r_in["grad"] / max(s_in["grad"], 1e-6),
                "sep_before": s_in["lum"] - s_out["lum"],
                "sep_after": r_in["lum"] - r_out["lum"],
                "sat_before": s_in["sat"],
                "sat_after": r_in["sat"],
                "lum_before": s_in["lum"],
                "lum_after": r_in["lum"],
                "std_before": s_in["std"],
                "std_after": r_in["std"],
            }
        )

    print("=== 5 个掉检框：原图 vs 复原图的局部结构 ===")
    print(
        "%-6s %-6s %10s %10s %10s %10s %8s"
        % ("天气", "类别", "梯度比", "分离度前", "分离度后", "饱和前", "饱和后")
    )
    for r in rows:
        print(
            "%-6s %-6s %10.3f %10.4f %10.4f %10.4f %8.4f"
            % (
                r["weather"],
                r["cls"],
                r["grad_ratio"],
                r["sep_before"],
                r["sep_after"],
                r["sat_before"],
                r["sat_after"],
            )
        )
    print()
    print("梯度比 = 复原图梯度能量 / 原图梯度能量（<1 表示结构被抹平）")
    print("分离度 = 框内亮度 − 框外亮度（目标与背景的对比，检测器最看重的信号）")

    (RUN / "dropped_region_stats.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n已写 runs/real_v4_allweather/dropped_region_stats.json")


if __name__ == "__main__":
    main()
