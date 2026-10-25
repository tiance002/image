"""诊断 v3/v4 在不同天气上的"改动性质"。

要回答的问题：v3 在 night/snow 上为什么掉点？
假设 A：模型把夜晚/雪景"洗白"（过度提亮），等于篡改场景；
假设 B：模型改动量过大，破坏了原有结构。
本脚本分别量化：亮度漂移(delta_lum)、对比度变化、饱和度变化、改动幅度。

v4 加入了真实场景自监督对，核心目的就是消除"把夜晚洗成白天"。
因此这个脚本要能同时跑多个 checkpoint 做横向对比：v3 是洗白的反例，
v4 是修复后的样本。用 --ckpts 传多个权重即可。

用法：
    PYTHONPATH=src;. python scripts/diagnose_weather_change.py \
        --ckpts runs/real_v3_conservative/best.pt runs/real_v4_allweather/best.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import cv2
import torch

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from saferestore.models.nafnet import build_nafnet  # noqa: E402

ROOT = Path(".")


def imread(p: Path):
    """读图（走字节流，避开中文路径导致的静默失败）。"""
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def stats(img01: np.ndarray) -> dict[str, float]:
    """亮度/对比度/饱和度统计。"""
    lab = cv2.cvtColor(np.clip(img01, 0, 1).astype(np.float32), cv2.COLOR_BGR2LAB)
    lum = lab[:, :, 0] / 100.0
    mx = img01.max(axis=2)
    mn = img01.min(axis=2)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return {"lum": float(lum.mean()), "std": float(lum.std()), "sat": float(sat.mean())}


def load_model(ckpt: Path, device: str):
    """加载一个复原权重。EMA 权重优先，没有则用原始权重。"""
    model = build_nafnet("standard")
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state.get("ema", state), strict=True)
    return model.to(device).eval()


def run_one(model, device: str, weather: str, files: list[Path]) -> np.ndarray:
    """对某个天气的样本集合跑复原，返回逐图统计矩阵。"""
    acc = []
    for path in files:
        bgr = imread(path)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
        # NAFNet 的下采样要求边长是 16 的倍数，先补齐再裁回
        _, _, hh, ww = x.shape
        ph = (16 - hh % 16) % 16
        pw = (16 - ww % 16) % 16
        if ph or pw:
            x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
        with torch.no_grad():
            y = model(x.to(device)).clamp(0, 1).cpu().numpy()[0].transpose(1, 2, 0)
        y = y[:hh, :ww]
        src = stats(rgb.astype(np.float32) / 255.0)
        out = stats(y)
        delta = float(np.abs(y - rgb.astype(np.float32) / 255.0).mean())
        acc.append((src["lum"], out["lum"], src["std"], out["std"],
                    src["sat"], out["sat"], delta))
    return np.array(acc)


def main() -> None:
    parser = argparse.ArgumentParser(description="多权重横向对比亮度漂移")
    parser.add_argument(
        "--ckpts",
        nargs="+",
        default=["runs/real_v3_conservative/best.pt"],
        help="要对比的 checkpoint 列表",
    )
    parser.add_argument("--n", type=int, default=40, help="每个天气取多少张")
    parser.add_argument("--out", default="", help="结果 JSON 输出路径")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    acdc = ROOT / "data/raw/ACDC_images/rgb_anon"
    # 样本集合固定：所有模型跑同一批图，差异只来自模型本身
    sample = {
        weather: sorted((acdc / weather / "val").rglob("*_rgb_anon.png"))[: args.n]
        for weather in ("fog", "rain", "night", "snow")
    }

    all_rows = {}
    for ckpt in args.ckpts:
        name = Path(ckpt).parent.name
        model = load_model(ROOT / ckpt, device)
        print("\n===== %s =====" % name)
        rows = {}
        for weather in ("fog", "rain", "night", "snow"):
            arr = run_one(model, device, weather, sample[weather])
            rows[weather] = arr.mean(axis=0).tolist()
            print("%-6s n=%3d  亮度 %6.1f -> %6.1f (%+6.1f)  对比度 %5.3f -> %5.3f  "
                  "饱和 %5.3f -> %5.3f  改动量 %.4f"
                  % (weather, len(arr), arr[:, 0].mean(), arr[:, 1].mean(),
                     (arr[:, 1] - arr[:, 0]).mean(), arr[:, 2].mean(), arr[:, 3].mean(),
                     arr[:, 4].mean(), arr[:, 5].mean(), arr[:, 6].mean()))
        all_rows[name] = rows
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print()
    print("说明：'亮度 +xx' 表示模型把画面整体提亮（过曝/洗白风险）。")
    print("      v3 的 night 亮度被大幅抬高即为'把夜晚洗成白天'。")

    if args.out:
        out_path = ROOT / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("结果已写入 %s" % out_path)


if __name__ == "__main__":
    main()
