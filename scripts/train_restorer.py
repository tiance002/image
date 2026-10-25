"""在真实感配对上训练复原网络，并用 ACDC 检测 mAP 做模型选择。

与旧流程的关键差异：
1. 训练数据是 ACDC train_ref 真实清晰帧 + 物理标定退化（见
   scripts/build_real_weather_pairs.py），不再是 192x128 的 PIL 卡通图；
2. 骨干换成 NAFNet（17M 参数），不再是 base_channels=16 的小网络；
3. 模型选择看 ACDC 真实雨雾上的检测 mAP50，而不是验证集 PSNR/SSIM；
4. 损失里显式加入"颜色恒常"项，直接压制此前诊断出的
   "猛压暗 + 猛加饱和把目标沉进暗部"这一失效模式。

用法：
    PYTHONPATH=src python scripts/train_restorer.py --name real_v1 --iters 30000
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

# 离线约束：任何库若尝试拉取权重/模型，一律落到项目目录内，不写系统盘、不重复下载。
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("YOLO_OFFLINE", "1")

from saferestore.losses.perceptual import DetectorFeatureLoss, prepare_detector_view  # noqa: E402
from saferestore.models.nafnet import build_nafnet  # noqa: E402


# ---------------------------------------------------------------- 数据


def imread_rgb(path: Path) -> np.ndarray:
    """中文路径安全读取，返回 RGB float32 0~1。"""
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


class PairDataset(Dataset):
    """退化图/清晰图配对，随机裁剪 + 水平翻转。"""

    def __init__(
        self,
        manifest: Path,
        crop: int,
        length: int,
        seed: int = 42,
        min_scale: float = 1.0,
        extra_manifests: list[Path] | None = None,
        extra_ratio: float = 0.5,
    ) -> None:
        with manifest.open(encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise ValueError(f"{manifest} 为空")
        # 多清单混合采样：把附加清单的样本按 ratio 比例混入主清单。
        # 用途：night/snow 用"真实场景自监督对"（目标=真实帧本身），
        # 与 fog/rain 的"合成退化→清晰图"配对共存，两种监督信号互不冲突。
        self.groups = [(self.rows, 1.0)]
        if extra_manifests:
            for path in extra_manifests:
                with path.open(encoding="utf-8") as handle:
                    extra = list(csv.DictReader(handle))
                if extra:
                    self.groups.append((extra, float(extra_ratio)))
        self.crop = crop
        self.length = length
        self.seed = seed
        self.min_scale = float(min_scale)

    def __len__(self) -> int:
        return self.length

    def _pick(self, index: int):
        """按比例从各清单里挑一行，返回 (行, 该行的随机种子偏移)。

        主清单权重 1.0，附加清单各占 extra_ratio。
        """
        rng = random.Random((self.seed + index) * 7919)
        total = sum(w for _, w in self.groups)
        draw = rng.random() * total
        acc = 0.0
        for rows, weight in self.groups:
            acc += weight
            if draw <= acc:
                row = rows[rng.randrange(len(rows))]
                return row, rng
        rows, _ = self.groups[-1]
        return rows[rng.randrange(len(rows))], rng

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # 样本选取：按各清单的权重比例随机抽取，并配合 rng 做随机裁剪与光度抖动。
        # 早期版本用 index % len(rows) 取样，会把同一场景的 3 个退化变体塞进同一批次，
        # 批内多样性极差；改为按比例随机抽取后批内样本互相独立。
        row, rng = self._pick(index)
        degraded = imread_rgb(ROOT / row["degraded_path"])
        clean = imread_rgb(ROOT / row["clean_path"])
        h, w = clean.shape[:2]
        # 尺度抖动：裁一块更大的区域再缩到 crop 尺寸，让模型见到更小的目标。
        # 动机：训练窗口是 320 裁剪，而推理是 1920 整图缩到 640 再送检测器，
        # 同一个目标在训练样本里比推理时大约大 3 倍 —— 模型实际上没见过真正的小目标。
        # 而 mAP 的分母主力（truck/bicycle/bus）恰恰是小目标。
        scale = rng.uniform(self.min_scale, 1.0) if self.min_scale < 1.0 else 1.0
        source = int(round(self.crop / max(scale, 1e-6)))
        ch = min(source, h)
        cw = min(source, w)
        top = rng.randrange(0, h - ch + 1)
        left = rng.randrange(0, w - cw + 1)
        degraded = degraded[top : top + ch, left : left + cw]
        clean = clean[top : top + ch, left : left + cw]
        if ch != self.crop or cw != self.crop:
            degraded = cv2.resize(
                degraded, (self.crop, self.crop), interpolation=cv2.INTER_AREA
            )
            clean = cv2.resize(
                clean, (self.crop, self.crop), interpolation=cv2.INTER_AREA
            )
        if rng.random() < 0.5:
            degraded = degraded[:, ::-1]
            clean = clean[:, ::-1]
        # 光度抖动必须对退化图与清晰图施加同一组参数：
        # 只抖动输入等于往标签里灌噪声（同一张清晰图会绑定多个不同色调的输入）。
        gamma = rng.uniform(0.90, 1.12)
        gain = np.array([rng.uniform(0.97, 1.03) for _ in range(3)], dtype=np.float32)
        degraded = np.clip(degraded**gamma * gain, 0, 1)
        clean = np.clip(clean**gamma * gain, 0, 1)
        # 退化图在 JPEG 编码下可能有极轻微越界
        degraded = np.ascontiguousarray(np.clip(degraded, 0, 1))
        clean = np.ascontiguousarray(np.clip(clean, 0, 1))
        to_tensor = lambda a: torch.from_numpy(a.transpose(2, 0, 1))  # noqa: E731
        return to_tensor(degraded), to_tensor(clean)


# ---------------------------------------------------------------- 损失


def charbonnier(
    x: torch.Tensor, y: torch.Tensor, eps: float = 1e-3, weight: torch.Tensor | None = None
) -> torch.Tensor:
    diff = torch.sqrt((x - y) ** 2 + eps**2)
    if weight is not None:
        diff = diff * weight
    return diff.mean()


def edge_weight_map(clean: torch.Tensor, beta: float) -> torch.Tensor:
    """按清晰图的梯度幅值构造空间权重图，让边缘区域承担更多重建压力。

    动机：v2 的 Recall 已经和基线持平，但 mAP 仍差一截 —— 说明目标"检得到、
    框不准"。框的精度取决于四条边的定位精度，而普通的像素/SSIM 损失把平坦
    区域（天空、路面、挡风玻璃）和边缘区域同等对待，模型为压低整体误差会优先
    把边缘一起磨平。给边缘加权相当于把优化容量挪到真正决定框位置的地方。

    两个实现细节：
    1. 用**逐样本 95 分位数**归一化，而不是固定阈值。不同场景对比度差异极大
       （隧道口 vs 空旷高速），固定阈值会让一批样本权重全饱和、另一批全为 1。
    2. 最后归一化到**均值 1**。否则调大 beta 等于同时放大了像素损失总权重，
       实验里就分不清是"边缘加权起了作用"还是"像素项变重了"。
    """
    gray = clean.float().mean(dim=1, keepdim=True)
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=gray.dtype,
        device=gray.device,
    )
    ky = kx.transpose(0, 1)
    # 手动 replicate 填充：补零会在画面外圈凭空造出梯度，把 95 分位数抬高，
    # 反而压低画面内部真实边缘的权重（检测目标常出现在贴边位置）。
    padded = F.pad(gray, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(padded, kx.view(1, 1, 3, 3))
    gy = F.conv2d(padded, ky.view(1, 1, 3, 3))
    # 平方项里的 eps 必须远小于分位数的兜底值（1e-3），否则在纯色图上
    # sqrt(eps) 会成为主导项、被归一化放大成"假边缘"（实测会到 3.4 倍）。
    mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
    scale = mag.flatten(1).quantile(0.95, dim=1).clamp_min(1e-3).view(-1, 1, 1, 1)
    weight = 1.0 + beta * (mag / scale).clamp(0.0, 1.0)
    return weight / weight.mean()


def _gauss_kernel(size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    return g / g.sum()


def ssim_index(x: torch.Tensor, y: torch.Tensor, size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """逐通道 SSIM，用可分离高斯窗，避免重复构造大卷积核。"""
    # 常数项 C1=1e-4 在 fp16 下已逼近可表示下限，这里强制用 float32 算
    x = x.float()
    y = y.float()
    device, dtype = x.device, x.dtype
    g = _gauss_kernel(size, sigma, device, dtype)
    channels = x.shape[1]
    kernel_h = g.view(1, 1, size, 1).repeat(channels, 1, 1, 1)
    kernel_w = g.view(1, 1, 1, size).repeat(channels, 1, 1, 1)
    pad = size // 2

    def blur(t: torch.Tensor) -> torch.Tensor:
        t = F.conv2d(t, kernel_h, padding=(pad, 0), groups=channels)
        return F.conv2d(t, kernel_w, padding=(0, pad), groups=channels)

    mu_x, mu_y = blur(x), blur(y)
    sigma_x = blur(x * x) - mu_x**2
    sigma_y = blur(y * y) - mu_y**2
    sigma_xy = blur(x * y) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    )
    return ssim_map.mean()


def gradient_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Sobel 梯度一致性，保住目标边缘（检测器依赖边缘）。"""
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device)
    ky = kx.transpose(0, 1)
    kx = kx.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    ky = ky.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    gx_x = F.conv2d(x, kx, padding=1, groups=3)
    gx_y = F.conv2d(y, ky, padding=1, groups=3)
    gy_x = F.conv2d(x, ky, padding=1, groups=3)
    gy_y = F.conv2d(y, kx, padding=1, groups=3)
    return (gx_x - gx_y).abs().mean() + (gy_x - gy_y).abs().mean()


def color_constancy(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """约束分通道均值与整体亮度，防止过度压暗/过饱和。

    这正是真实数据上失效的直接形式：旧模型把平均亮度从 48.4 压到 34.1、
    饱和度放大到 3.9 倍，目标沉入暗部后无法被检出。
    """
    mean_x = x.mean(dim=(2, 3))
    mean_y = y.mean(dim=(2, 3))
    return (mean_x - mean_y).abs().mean()


# ---------------------------------------------------------------- 推理与评测


@torch.inference_mode()
def restore_tiled(model, image: torch.Tensor, tile: int = 1024, overlap: int = 64) -> torch.Tensor:
    """分块推理，让 1920x1080 也能在 8GB 显存上跑完。"""
    _, _, h, w = image.shape
    if h <= tile and w <= tile:
        return model(image)
    out = torch.zeros_like(image)
    weight = torch.zeros_like(image)
    step = tile - overlap
    ys = list(range(0, max(h - tile, 0) + 1, step)) or [0]
    xs = list(range(0, max(w - tile, 0) + 1, step)) or [0]
    if ys[-1] + tile < h:
        ys.append(h - tile)
    if xs[-1] + tile < w:
        xs.append(w - tile)
    for y in ys:
        for x in xs:
            patch = image[:, :, y : y + tile, x : x + tile]
            pred = model(patch)
            out[:, :, y : y + tile, x : x + tile] += pred
            weight[:, :, y : y + tile, x : x + tile] += 1.0
    return out / weight.clamp_min(1e-6)


def load_gt_records(records_path: str | None = None) -> list[dict]:
    """读入评测记录。

    默认复用官方评测产物里的真值框，保证与已发表数字同一协议。
    传入 records_path 可切换到 406 张全天气评测集。
    """
    path = ROOT / (records_path or "runs/acdc_benchmark/evaluation/weather_input.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload["per_image"]
    return payload


def attach_baseline_predictions(records: list[dict]) -> int:
    """给每条评测记录挂上"直检原图"的基线预测，供 OPR 计算。

    为什么必须按 file_name 配对、不能按顺序 zip
    ------------------------------------------
    原本用 `zip(records, baseline)`：records 在 406 张口径下是
    fog100 / rain100 / night106 / snow100，而基线文件只有 200 张（fog+rain）。
    zip 只能配上前 200 条，night/snow 拿不到基线 -> OPR 的分母为 0 ->
    这两个天气的 OPR 恒为 0，看起来像"模型把目标全丢了"，其实是统计口径的 bug。

    现在改为按 file_name 建索引配对，并自动挑选覆盖全部天气的基线文件：
    优先用 406 张版本的直检产物，缺失时回退到 200 张口径。
    """
    candidates = [
        "runs/acdc_benchmark/evaluation/weather_input_direct_406.json",
        "runs/real_v3_conservative/preds_input_406.json",
        "runs/acdc_benchmark/evaluation/weather_input.json",
    ]
    index: dict[str, dict] = {}
    used = ""
    for rel in candidates:
        path = ROOT / rel
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = payload["per_image"] if isinstance(payload, dict) and "per_image" in payload else payload
        table = {str(item.get("file_name") or item.get("image_id")): item for item in items}
        # 选覆盖评测集最多的那个基线文件
        if len(table) > len(index):
            index = table
            used = rel

    hit = 0
    for rec in records:
        key = str(rec.get("file_name") or rec.get("image_id"))
        base = index.get(key)
        if base is not None:
            rec["baseline_predictions"] = base.get("predictions", {})
            hit += 1
    print(f"基线预测来源 {used}，配对成功 {hit}/{len(records)} 张")
    if hit < len(records):
        print("  警告：部分图像缺少基线预测，这些图的 OPR 不会计入分母")
    return hit


def build_detector():
    import yaml
    from ultralytics import YOLO

    config = yaml.safe_load((ROOT / "configs/detector.yaml").read_text(encoding="utf-8"))
    weights = config["model"]
    model_path = weights if Path(weights).is_absolute() else str(ROOT / weights)
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"检测器本地权重不存在（禁止自动下载）：{model_path}")
    return YOLO(model_path), config


def evaluate_detection(
    model,
    records: list[dict],
    detector,
    config: dict,
    limit: int,
    device: str,
    save_dir: Path | None = None,
    pred_path: Path | None = None,
) -> dict[str, float]:
    """在 ACDC 真实雨雾图上复原 -> 检测 -> mAP50 / Recall / OPR。

    pred_path 给出时，会把逐图的真值框与预测框写成 JSON。200 张的评测集已经是
    ACDC val（fog 100 + rain 100）的全部，无法靠扩样本降低噪声，因此必须保留逐图
    结果以便做配对 bootstrap，给出 mAP 差值的置信区间。
    """
    from scripts.diagnose_detection_rootcause import dataset_map50, run_detector
    from saferestore.metrics import box_iou

    subset = records[:limit] if limit > 0 else records
    target_classes = sorted(
        {int(c) for rec in subset for c in rec["ground_truth"]}
    )
    per_image: list[dict] = []
    kept = total = 0
    dark_before: list[float] = []
    dark_after: list[float] = []
    # 分天气累计：雾与雨的失效机制不同（雨还有反光/水珠），合在一起看会互相掩盖
    per_weather: dict[str, dict] = {}

    for index, rec in enumerate(subset):
        image = imread_rgb(ROOT / "data/raw/ACDC_images/rgb_anon" / rec["file_name"])
        tensor = torch.from_numpy(image.transpose(2, 0, 1))[None].to(device)
        # 评测用 bf16（不再是 fp32）：实测 fp32 推理在这张卡上比 bf16 慢约 6 倍
        # （bf16 走张量核心，fp32 走 CUDA 核心），一次 200 张评测要 25 分钟。
        # 而且 v1 的历史数字都是 bf16 下测的，v2 必须用同精度才能横向比较。
        # 注意：训练不能用混合精度（fp16 会出 NaN 梯度），但纯推理没有这个问题。
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device == "cuda"
        ):
            restored = restore_tiled(model, tensor)
        restored_np = restored[0].float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        restored_u8 = (restored_np * 255.0).round().astype(np.uint8)
        if save_dir is not None:
            target = save_dir / rec["file_name"]
            target.parent.mkdir(parents=True, exist_ok=True)
            ok, buf = cv2.imencode(".jpg", restored_u8[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
            if ok:
                buf.tofile(str(target))
        preds = run_detector(detector, restored_u8, config, target_classes)
        per_image.append({"ground_truth": rec["ground_truth"], "predictions": preds})

        weather = str(rec.get("weather") or "unknown")
        bucket = per_weather.setdefault(weather, {"images": [], "kept": 0, "total": 0})
        bucket["images"].append(
            {"ground_truth": rec["ground_truth"], "predictions": preds}
        )

        # 目标保持率：输入已正确检出的目标，复原后有多少仍能被检出
        baseline = rec.get("baseline_predictions")
        if baseline is not None:
            for class_id, boxes in rec["ground_truth"].items():
                # ground_truth 的键来自 JSON，是字符串；predictions 的键是 int。
                # 不转换会让 preds.get("26") 永远取空，OPR 恒为 0。
                key = int(class_id)
                ref = [p["box"] for p in baseline.get(class_id, [])]
                cur = [p["box"] for p in preds.get(key, [])]
                for gt in boxes:
                    if any(box_iou(np.asarray(p), np.asarray(gt)) >= 0.5 for p in ref):
                        total += 1
                        bucket["total"] += 1
                        stayed = int(
                            any(box_iou(np.asarray(p), np.asarray(gt)) >= 0.5 for p in cur)
                        )
                        kept += stayed
                        bucket["kept"] += stayed

        lab_in = cv2.cvtColor(
            cv2.resize(image, (320, 180)), cv2.COLOR_RGB2LAB
        )[:, :, 0]
        lab_out = cv2.cvtColor(
            cv2.resize(restored_np, (320, 180)), cv2.COLOR_RGB2LAB
        )[:, :, 0]
        dark_before.append(float((lab_in / 100.0 < 0.15).mean()))
        dark_after.append(float((lab_out / 100.0 < 0.15).mean()))

        # 评测是阻塞的，没有进度输出时完全看不出是在跑还是卡死（踩过一次）
        if (index + 1) % 25 == 0 or index + 1 == len(subset):
            print(f"  评测进度 {index + 1}/{len(subset)}", flush=True)

    metrics = dataset_map50(per_image)
    metrics["opr"] = kept / max(total, 1)
    metrics["dark_before"] = float(np.mean(dark_before))
    metrics["dark_after"] = float(np.mean(dark_after))
    metrics["n_images"] = len(subset)

    if pred_path is not None:
        # 逐图落盘：图像用 file_name 标识，便于与 baseline 的 per_image 逐条配对
        serializable = [
            {
                "image_id": rec["file_name"],
                "ground_truth": {
                    str(class_id): [np.asarray(box).tolist() for box in boxes]
                    for class_id, boxes in item["ground_truth"].items()
                },
                "predictions": {
                    str(class_id): [
                        {"box": np.asarray(pred["box"]).tolist(), "confidence": float(pred["confidence"])}
                        for pred in items
                    ]
                    for class_id, items in item["predictions"].items()
                },
            }
            for rec, item in zip(subset, per_image)
        ]
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_path.write_text(
            json.dumps(serializable, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  逐图预测已写入 {pred_path}", flush=True)

    metrics["per_weather"] = {
        weather: {
            **dataset_map50(bucket["images"]),
            "opr": bucket["kept"] / max(bucket["total"], 1),
            "n_images": len(bucket["images"]),
            "n_targets": bucket["total"],
        }
        for weather, bucket in sorted(per_weather.items())
        if bucket["images"]
    }
    return metrics


# ---------------------------------------------------------------- 主循环


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="real_nafnet_v1")
    parser.add_argument("--variant", default="standard")
    parser.add_argument("--train-manifest", default="data/real_weather/manifests/train.csv")
    parser.add_argument(
        "--extra-manifest",
        action="append",
        default=[],
        help="附加训练清单，可重复指定；与主清单按 --extra-ratio 比例混合采样",
    )
    parser.add_argument(
        "--extra-ratio",
        type=float,
        default=0.5,
        help="每个附加清单相对主清单的采样权重",
    )
    parser.add_argument("--crop", type=int, default=320)
    # 尺度抖动下限：1.0 表示关闭。取 0.7 表示每次随机在 [0.7,1.0] 之间选缩放比，
    # 裁更大区域再缩到 crop，从而让训练样本里的目标尺度分布覆盖推理时的实际情况。
    parser.add_argument("--scale-jitter", type=float, default=1.0)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--iters", type=int, default=30000)
    parser.add_argument(
        "--stop-after", type=int, default=0,
        help="本轮最多训练多少步（0 表示不限）；不改变 --iters 的学习率计划",
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--amp", action="store_true", help="启用 bf16 混合精度（实测不稳，默认关闭）")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--w-pixel", type=float, default=1.0)
    parser.add_argument("--w-ssim", type=float, default=0.15)
    parser.add_argument("--w-grad", type=float, default=0.05)
    parser.add_argument("--w-color", type=float, default=0.1)
    # 修改量惩罚：模型输出是残差形式（输入 + delta），这一项约束 delta 的幅度。
    # 0 表示关闭。取值过大会让模型退化成恒等映射（什么都不改）。
    parser.add_argument(
        "--w-delta",
        type=float,
        default=0.0,
        help="修改量 L1 惩罚权重（0 表示关闭）",
    )
    # 边缘加权像素损失：按清晰图梯度幅值放大边缘处的重建压力，0 表示关闭。
    # 针对"Recall 已追平但 mAP 不够"的症状（框定位不准）。
    parser.add_argument(
        "--edge-px",
        type=float,
        default=0.0,
        help="边缘区域像素损失的放大倍数（0 表示关闭）",
    )
    # 检测器特征一致性损失：直接用下游检测器（冻结）的特征空间做监督。
    # 默认关闭，v1 作为纯像素监督的基线。
    parser.add_argument(
        "--w-det-feat",
        type=float,
        default=0.0,
        help="检测器特征一致性损失权重（0 表示关闭）",
    )
    parser.add_argument("--det-weights", default="yolov8n.pt", help="冻结检测器权重（本地文件）")
    parser.add_argument("--w-det-conf", type=float, default=0.0,
                        help="输入教师的一侧置信度保护权重，独立于特征损失权重")
    parser.add_argument("--det-conf-monitor", action="store_true",
                        help="即使保护权重为0也运行同一教师前向，供公平对照与监督覆盖监测")
    parser.add_argument("--det-conf-threshold", type=float, default=0.25)
    parser.add_argument("--det-conf-topk", type=int, default=16)
    parser.add_argument(
        "--det-feat-start",
        type=int,
        default=1500,
        help="从第几步开始施加特征损失：先让像素损失把图拉回正常色域，再上任务监督",
    )
    parser.add_argument(
        "--det-feat-crop",
        type=int,
        default=256,
        help="特征损失只取中心裁剪的边长（0 表示用整块），用于控制显存",
    )
    parser.add_argument(
        "--det-feat-view", choices=("crop", "resize"), default="crop",
        help="crop 保持中心裁剪；resize 将完整训练块缩小到特征窗口尺寸",
    )
    parser.add_argument(
        "--det-feat-magnitude-weight", type=float, default=0.0,
        help="最低分辨率特征幅度损失的相对权重；0 表示关闭，不增加检测器前向次数",
    )
    # P3 是最高分辨率的特征层，专门负责小目标。mAP 是类别等权平均，
    # 而 truck/bicycle/bus 这些少样本小类恰恰是 mAP 的分母主力，
    # 因此把优化重心压到 P3 上比继续提升整体画质更能撬动 mAP。
    parser.add_argument(
        "--det-feat-scales",
        default="1,1,1",
        help="P3/P4/P5 三个尺度的特征损失权重，逗号分隔；默认等权",
    )
    parser.add_argument("--ema", type=float, default=0.999)
    parser.add_argument("--eval-det-every", type=int, default=4000)
    parser.add_argument(
        "--eval-records",
        default=None,
        help="评测用的记录集路径；留空用 200 张雨雾官方口径，传 all406 可评全天气",
    )
    parser.add_argument("--det-limit", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    if args.stop_after < 0:
        parser.error("--stop-after 不能为负数")
    if args.resume and not Path(args.resume).is_file():
        parser.error(f"续训权重不存在：{args.resume}")
    if not math.isfinite(args.w_det_conf) or args.w_det_conf < 0:
        parser.error("--w-det-conf 必须为有限非负数")
    confidence_enabled = args.det_conf_monitor or args.w_det_conf > 0
    if confidence_enabled and args.w_det_feat <= 0:
        parser.error("当前置信度保护复用特征损失检测器，需要 --w-det-feat > 0")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir = ROOT / "runs" / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    dataset = PairDataset(
        ROOT / args.train_manifest,
        args.crop,
        length=10**7,
        seed=args.seed,
        min_scale=args.scale_jitter,
        extra_manifests=[ROOT / p for p in args.extra_manifest],
        extra_ratio=args.extra_ratio,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    model = build_nafnet(variant=args.variant).to(device)
    ema_model = build_nafnet(variant=args.variant).to(device)
    ema_model.load_state_dict(model.state_dict())
    for param in ema_model.parameters():
        param.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99)
    )
    # bf16 与 fp32 同指数范围，不会像 fp16 那样在 LayerNorm 的方差上溢出；
    # 因此不需要 GradScaler 的缩放/回退机制。
    # 实测 fp16 在第 0 步即出现 NaN 梯度、bf16 梯度范数失控（10->54）并塌成全黑；
    # 只有 fp32 稳定。这里默认关闭混合精度，靠扩大步数而非降低精度换吞吐。
    amp_dtype = torch.bfloat16 if (args.amp and device == "cuda") else torch.float32
    use_amp = bool(args.amp and device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    start_iter = 0
    if args.resume and Path(args.resume).exists():
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        ema_model.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_iter = int(state["iter"]) + 1
        print(f"从 {args.resume} 恢复，起始 iter={start_iter}")

    # 预取真值并缓存输入图的基线预测（用于计算 OPR）
    records = load_gt_records(args.eval_records)
    attach_baseline_predictions(records)

    detector, det_config = build_detector()
    det_feat_loss = None
    if args.w_det_feat > 0:
        scale_weights = tuple(
            float(item) for item in str(args.det_feat_scales).replace(";", ",").split(",")
        )
        det_feat_loss = DetectorFeatureLoss(
            ROOT / args.det_weights, layer_weights=scale_weights,
            magnitude_weight=args.det_feat_magnitude_weight,
            confidence_monitor=confidence_enabled, confidence_grad=args.w_det_conf > 0,
            confidence_threshold=args.det_conf_threshold, confidence_topk=args.det_conf_topk,
        ).to(device)
        print(
            f"检测器特征损失已启用：权重 {args.w_det_feat}，"
            f"层 {det_feat_loss.layer_indices}，"
            f"尺度权重 {det_feat_loss.layer_weights}，从第 {args.det_feat_start} 步开始"
        )
    log_path = run_dir / "train_log.csv"
    new_log = not log_path.exists()
    log_file = log_path.open("a", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    if new_log:
        log_writer.writerow(["iter", "loss", "pixel", "ssim", "grad", "color", "lr", "sec"])

    best_map = -1.0
    confidence_totals = torch.zeros(7, dtype=torch.long, device=device)
    confidence_pairs = torch.zeros(7, dtype=torch.long, device=device)
    batch_trace = hashlib.sha256()
    history: list[dict] = []
    model.train()
    data_iter = iter(loader)
    t_start = time.time()

    end_iter = min(args.iters, start_iter + args.stop_after) if args.stop_after > 0 else args.iters
    if end_iter <= start_iter:
        raise ValueError("训练结束步数必须大于续训起点")
    for step in range(start_iter, end_iter):
        try:
            degraded, clean = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            degraded, clean = next(data_iter)
        batch_trace.update(degraded.numpy().tobytes())
        batch_trace.update(clean.numpy().tobytes())
        if step - start_iter < 2 or step % args.log_every == 0:
            (run_dir / "batch_trace.json").write_text(json.dumps({
                "batches": step - start_iter + 1, "sha256": batch_trace.hexdigest(),
            }), encoding="utf-8")
        degraded = degraded.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        lr = (
            args.lr * (step + 1) / args.warmup
            if step < args.warmup
            else args.min_lr
            + (args.lr - args.min_lr)
            * 0.5
            * (1.0 + math.cos(math.pi * (step - args.warmup) / max(args.iters - args.warmup, 1)))
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            pred = model(degraded)
            # 边缘加权只在开启时才构造，关掉时不产生任何额外开销
            edge_w = edge_weight_map(clean, args.edge_px) if args.edge_px > 0 else None
            l_pixel = charbonnier(pred, clean, weight=edge_w)
            l_ssim = 1.0 - ssim_index(pred, clean)
            l_grad = gradient_loss(pred, clean)
            l_color = color_constancy(pred, clean)
            loss = (
                args.w_pixel * l_pixel
                + args.w_ssim * l_ssim
                + args.w_grad * l_grad
                + args.w_color * l_color
            )
            l_delta = torch.zeros((), device=device)
            if args.w_delta > 0:
                # 模型是残差形式（输出 = 输入 + delta），这一项直接惩罚修改幅度。
                # 动机很直接：检测器在**原始退化图**上表现最好（0.2042），复原的目标
                # 是"别改坏"而不是"改得多"。推理时把输出与输入混合到 0.9 倍，
                # 在数学上等价于把 delta 缩小到 0.9 倍 —— 说明模型很可能改过头了。
                # 与其在推理时打折，不如在训练时就把这条约束写进目标。
                l_delta = (pred - degraded).abs().mean()
                loss = loss + args.w_delta * l_delta
            l_det_feat = torch.zeros((), device=device)
            l_det_conf = torch.zeros((), device=device)
            if det_feat_loss is not None and step >= args.det_feat_start:
                # 特征损失前先把图拉回 [0,1]：检测器是按该取值范围预训练的，
                # 直接喂未截断的输出会让特征尺度失真。
                pred_in = pred.float().clamp(0.0, 1.0)
                clean_in = clean.float()
                pred_in, clean_in = prepare_detector_view(
                    pred_in, clean_in, args.det_feat_crop, args.det_feat_view
                )
                reference = None
                if confidence_enabled:
                    reference, _ = prepare_detector_view(
                        degraded, degraded, args.det_feat_crop, args.det_feat_view
                    )
                l_det_feat = det_feat_loss(pred_in, clean_in, confidence_reference=reference)
                loss = loss + args.w_det_feat * l_det_feat
                if confidence_enabled:
                    l_det_conf = det_feat_loss.last_confidence_loss
                    confidence_totals += det_feat_loss.last_confidence_stats["selected_per_class"]
                    confidence_pairs += det_feat_loss.last_confidence_stats["class_pairs_per_class"]
                    if args.w_det_conf > 0:
                        loss = loss + args.w_det_conf * l_det_conf
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                ema_param.mul_(args.ema).add_(param.detach(), alpha=1 - args.ema)

        if step % args.log_every == 0:
            elapsed = time.time() - t_start
            log_writer.writerow(
                [
                    step,
                    f"{loss.item():.5f}",
                    f"{l_pixel.item():.5f}",
                    f"{l_ssim.item():.5f}",
                    f"{l_grad.item():.5f}",
                    f"{l_color.item():.5f}",
                    f"{lr:.2e}",
                    f"{elapsed:.1f}",
                ]
            )
            log_file.flush()
            print(
                f"iter {step:6d} loss {loss.item():.4f} "
                f"(px {l_pixel.item():.4f} ssim {l_ssim.item():.4f} "
                f"grad {l_grad.item():.4f} col {l_color.item():.4f}"
                + (f" delta {l_delta.item():.4f}" if args.w_delta > 0 else "")
                + (f" detfeat {l_det_feat.item():.4f}" if det_feat_loss is not None else "")
                + (f" detconf {l_det_conf.item():.6f}" if confidence_enabled else "")
                + f") lr {lr:.1e}",
                flush=True,
            )
            if confidence_enabled:
                names = ("person", "bicycle", "car", "motorcycle", "bus", "train", "truck")
                (run_dir / "confidence_monitor.json").write_text(json.dumps({
                    "iter": step, "weight": args.w_det_conf,
                    "loss_at_logged_step": float(l_det_conf.detach()),
                    "selected_anchors": dict(zip(names, confidence_totals.cpu().tolist())),
                    "supervised_image_class_pairs": dict(zip(names, confidence_pairs.cpu().tolist())),
                }, ensure_ascii=False, indent=2), encoding="utf-8")

        # 评测点按"本轮已完成的步数"计数，而不是绝对步数。
        # 否则续训时若起点恰好是 eval_det_every 的整数倍，
        # 第一个循环步就会立刻触发一次耗时十几分钟的 406 张全天气评测，
        # 看起来像训练卡死。done_steps 从 1 起算，保证首步不触发。
        done_steps = step - start_iter + 1
        if args.eval_det_every > 0 and done_steps % args.eval_det_every == 0:
            ema_model.eval()
            metrics = evaluate_detection(
                ema_model,
                records,
                detector,
                det_config,
                args.det_limit,
                device,
                pred_path=run_dir / f"preds_{step:06d}.json",
            )
            ema_model.train()
            metrics["iter"] = step
            history.append(metrics)
            (run_dir / "det_history.json").write_text(
                json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                f"  [评测 iter {step}] mAP50 {metrics['map50']:.4f} "
                f"Recall {metrics['recall']:.4f} OPR {metrics['opr']:.4f} "
                f"暗部 {metrics['dark_before']:.3f}->{metrics['dark_after']:.3f}",
                flush=True,
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "ema": ema_model.state_dict(),
                    "iter": step,
                    "metrics": metrics,
                },
                run_dir / f"iter_{step:06d}.pt",
            )
            if metrics["map50"] > best_map:
                best_map = metrics["map50"]
                torch.save(
                    {"ema": ema_model.state_dict(), "iter": step, "metrics": metrics},
                    run_dir / "best.pt",
                )
                print(f"  新最优 mAP50={best_map:.4f} -> best.pt", flush=True)

        if args.save_every > 0 and step > 0 and step % args.save_every == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "ema": ema_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "iter": step,
                },
                run_dir / "latest.pt",
            )

    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "iter": end_iter - 1,
        },
        run_dir / "latest.pt",
    )
    if confidence_enabled:
        names = ("person", "bicycle", "car", "motorcycle", "bus", "train", "truck")
        (run_dir / "confidence_monitor.json").write_text(json.dumps({
            "iter": end_iter - 1, "weight": args.w_det_conf,
            "loss_at_logged_step": float(l_det_conf.detach()),
            "selected_anchors": dict(zip(names, confidence_totals.cpu().tolist())),
            "supervised_image_class_pairs": dict(zip(names, confidence_pairs.cpu().tolist())),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "batch_trace.json").write_text(json.dumps({
        "batches": end_iter - start_iter, "sha256": batch_trace.hexdigest(),
    }), encoding="utf-8")
    log_file.close()
    print(f"训练结束，用时 {(time.time() - t_start) / 60:.1f} 分钟，最优 mAP50={best_map:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
