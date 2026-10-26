"""Histoformer 官方检查点加载与单图前向冒烟：验证尺寸、颜色与显存。

用法：
    python scripts/np2_histoformer_probe.py --weights <path> --image <path>
"""
from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
ARCH = ROOT / 'third_party/histoformer/basicsr/models/archs/histoformer_arch.py'
CFG = ROOT / 'third_party/histoformer/Allweather/Options/Allweather_Histoformer.yml'


def load_arch_module():
    """按文件路径加载官方 arch 模块，绕过 basicsr 包级 __init__ 的重依赖导入。"""
    spec = importlib.util.spec_from_file_location('histoformer_arch', ARCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_model(weights: Path, device: str = 'cuda'):
    module = load_arch_module()
    cfg = yaml.safe_load(CFG.read_text(encoding='utf-8'))
    kwargs = dict(cfg['network_g'])
    kwargs.pop('type')
    net = module.Histoformer(**kwargs)
    ckpt = torch.load(weights, map_location='cpu', weights_only=False)
    if 'params' not in ckpt:
        raise KeyError(f'检查点缺少 params 键：{weights}')
    # strict=True：缺键或多键都必须报错，禁止随机初始化继续跑
    net.load_state_dict(ckpt['params'], strict=True)
    return net.to(device).eval()


def restore_full(net, rgb: np.ndarray, device: str, factor: int = 8):
    """严格遵循官方 test_histoformer.py：RGB[0,1] → reflect pad 到 factor 倍数 → 前向 → 裁回。"""
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy())[None].to(device)
    h, w = tensor.shape[2], tensor.shape[3]
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    if padh or padw:
        tensor = F.pad(tensor, (0, padw, 0, padh), 'reflect')
    with torch.inference_mode():
        out = net(tensor)
    out = out[:, :, :h, :w]
    return out.clamp(0, 1), (h, w, padh, padw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    rgb = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32) / 255.0
    net = build_model(Path(args.weights), args.device)
    torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    out, geom = restore_full(net, rgb, args.device)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - begin
    out_np = out[0].permute(1, 2, 0).float().cpu().numpy()

    # 尺寸与数值范围核查
    print(f'输入尺寸={rgb.shape} 输出尺寸={out_np.shape} pad={geom[2:] if len(geom) == 4 else geom}')
    print(f'输出范围=[{out_np.min():.4f}, {out_np.max():.4f}] 有限={bool(np.isfinite(out_np).all())}')
    print(f'前向耗时={elapsed*1000:.1f}ms 峰值显存={torch.cuda.max_memory_allocated()/2**20:.0f}MiB')

    # 与原图的偏差：应当是小幅修正而非全图重置
    diff = np.abs(out_np - rgb)
    print(f'|输出-输入| 均值={diff.mean():.4f} 最大={diff.max():.4f}')
    psnr_in = -10 * np.log10(np.mean((out_np - rgb) ** 2) + 1e-12)
    print(f'输出相对输入的 PSNR={psnr_in:.3f}dB（越低表示改动越大，无真值时仅作幅度检查）')


if __name__ == '__main__':
    main()
