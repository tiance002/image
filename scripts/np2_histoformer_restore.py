"""NP2：官方预训练复原模型在 ACDC val 上的推理（Histoformer）。

要点：
1. 严格沿用官方 test_histoformer.py 的预处理：BGR→RGB、/255、reflect pad 到 8 的倍数、clamp[0,1]、回 BGR。
2. 1920x1080 全幅前向在本机 8GB 显存上会溢出到统一内存（实测单图 214s、峰值 14985MiB），
   因此改为固定参数的分块推理：块 640、重叠 64、汉宁窗加权融合。这一步是明确记录的适配。
3. 输出与原图同尺寸、同坐标，不做 resize，不影响下游 letterbox 与 GT 对应关系。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from scripts.np2_histoformer_probe import build_model  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_rgb(path: Path) -> np.ndarray:
    """按官方 util.load_img 的语义读图：BGR→RGB，保持 uint8。"""
    buf = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f'无法解码图像：{path}')
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_png(path: Path, rgb: np.ndarray) -> None:
    """cv2.imwrite 在含非 ASCII 的绝对路径下会静默失败，这里统一用 imencode。"""
    ok, buf = cv2.imencode('.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f'编码失败：{path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.tobytes())


def pad_to_factor(tensor: torch.Tensor, factor: int = 8):
    """官方 padding 规则：向右/下 reflect 填充到 factor 的整数倍。"""
    h, w = tensor.shape[-2], tensor.shape[-1]
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    if padh or padw:
        tensor = F.pad(tensor, (0, padw, 0, padh), 'reflect')
    return tensor, (h, w)


def forward_patch(net, patch: np.ndarray, device: str, use_bf16: bool) -> np.ndarray:
    """对单块做官方前向。patch 为 RGB [0,1] HxWx3，返回同尺寸 RGB。"""
    tensor = torch.from_numpy(patch.transpose(2, 0, 1).copy())[None].to(device)
    if use_bf16:
        tensor = tensor.to(torch.bfloat16)
    tensor, (h, w) = pad_to_factor(tensor)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
        out = net(tensor)
    out = out[:, :, :h, :w].clamp(0, 1)
    return out[0].float().permute(1, 2, 0).cpu().numpy()


def tile_positions(total: int, tile: int, stride: int):
    """返回覆盖 [0,total) 的起点列表，最后一块对齐到末尾，保证无遗漏。"""
    if total <= tile:
        return [0]
    positions = list(range(0, total - tile + 1, stride))
    if positions[-1] + tile < total:
        positions.append(total - tile)
    return positions


def hann_window(size: int, device: str) -> torch.Tensor:
    """无零端点的汉宁窗：取 size+2 点的对称汉宁窗（periodic=False）后去掉首尾两点。

    修复记录（NP2-R）：旧实现用 `hann_window(size)` 后 `clamp_min(1e-6)`，两维相乘后
    角点权重约 1e-12；融合时又把分母截到 `max(weight, 1e-8)`，等于把 1e-12 的权重
    除以 1e-8，输出被缩小 1e-4，恒等模型在四角会变黑。这里改用真正 >0 的窗口，
    并让分子、分母用同一个权重、分母完全不截断（见 restore_tiled 的全覆盖断言）。
    """
    return torch.hann_window(size + 2, periodic=False, device=device, dtype=torch.float32)[1:-1]


def restore_tiled(net, rgb: np.ndarray, device: str, tile: int, overlap: int, use_bf16: bool):
    """分块复原 + 归一化融合。

    融合约束（这是修复的核心）：
    1. 分子 `sum(out * w)` 与分母 `sum(w)` 使用**同一个** `w`；
    2. 分母不做任何截断，除法是逐像素精确的，因此恒等模型必须逐像素还原输入；
    3. 断言每个像素至少被一块覆盖（分母严格 > 0），否则说明块位置有遗漏，直接报错。
    """
    h, w = rgb.shape[:2]
    stride = tile - overlap
    ys, xs = tile_positions(h, tile, stride), tile_positions(w, tile, stride)
    acc = np.zeros_like(rgb, dtype=np.float64)
    weight = np.zeros((h, w, 1), dtype=np.float64)
    wy, wx = hann_window(tile, 'cpu').numpy(), hann_window(tile, 'cpu').numpy()
    window = (wy[:, None] * wx[None, :])[:, :, None]
    for y in ys:
        for x in xs:
            patch = rgb[y:y + tile, x:x + tile]
            ph, pw = patch.shape[:2]
            out = forward_patch(net, patch, device, use_bf16)
            win = window[:ph, :pw]
            acc[y:y + ph, x:x + pw] += out.astype(np.float64) * win
            weight[y:y + ph, x:x + pw] += win
    uncovered = int((weight <= 0).sum())
    if uncovered:
        raise RuntimeError(f'分块融合存在未覆盖像素 {uncovered} 个（tile={tile}, overlap={overlap}）')
    out = (acc / weight).clip(0, 1)
    return out.astype(np.float32), len(ys) * len(xs), float(weight.min())


def load_image_list(weathers, limit):
    payload = json.loads((ROOT / 'data/raw/ACDC_detection/gt_detection/instancesonly_val_gt_detection.json').read_text(encoding='utf-8'))
    images = sorted(payload['images'], key=lambda x: x['id'])
    images = [i for i in images if i['file_name'].split('/')[0] in weathers]
    if limit:
        images = images[:limit]
    return images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--name', required=True, help='输出子目录名，也是方法名')
    parser.add_argument('--weather', nargs='+', default=['rain', 'snow'])
    parser.add_argument('--tile', type=int, default=640)
    parser.add_argument('--overlap', type=int, default=64)
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--out-root', default='results/np2_revision/restored',
                        help='NP2-R 默认写入新目录，避免覆盖 np2_external_probe 的旧产物')
    args = parser.parse_args()

    images = load_image_list(set(args.weather), args.limit)
    out_dir = ROOT / args.out_root / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = Path(args.weights)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = build_model(weights, device)
    if args.bf16:
        net = net.half()
    params = sum(p.numel() for p in net.parameters())

    records, timings = [], []
    begin_all = time.perf_counter()
    for index, image in enumerate(images):
        src = ROOT / 'data/raw/ACDC_images/rgb_anon' / image['file_name']
        rgb = read_rgb(src).astype(np.float32) / 255.0
        torch.cuda.synchronize() if device == 'cuda' else None
        begin = time.perf_counter()
        restored, n_tiles, min_weight = restore_tiled(net, rgb, device, args.tile, args.overlap, args.bf16)
        torch.cuda.synchronize() if device == 'cuda' else None
        elapsed = time.perf_counter() - begin
        if restored.shape != rgb.shape:
            raise RuntimeError(f'尺寸不一致：{image["file_name"]} {restored.shape} vs {rgb.shape}')
        if not np.isfinite(restored).all():
            raise FloatingPointError(f'输出非有限：{image["file_name"]}')
        uint8 = (restored * 255).round().clip(0, 255).astype(np.uint8)
        rel = Path(image['file_name'])
        write_png(out_dir / rel, uint8)
        diff = np.abs(restored - rgb)
        records.append(dict(image_id=image['id'], file_name=image['file_name'],
                            weather=rel.parts[0], tiles=n_tiles,
                            diff_mean=float(diff.mean()), diff_max=float(diff.max()),
                            # 明确语义：逐像素取三通道绝对差的最大值，统计其 > 0.02 的像素占比
                            # （等价于任一通道超过阈值的像素比例；不是通道元素比例或平均通道改变量）。
                            changed_frac_maxchan_gt_0p02=float((diff.max(axis=2) > 0.02).mean()),
                            fusion_min_weight=min_weight,
                            restore_ms=elapsed * 1000))
        timings.append(elapsed * 1000)
        if (index + 1) % 10 == 0 or index + 1 == len(images):
            done = (time.perf_counter() - begin_all) / (index + 1)
            print(f'{args.name} {index+1}/{len(images)} 平均 {np.mean(timings):.0f}ms/图 预计剩余 {done*(len(images)-index-1)/60:.1f}min', flush=True)

    meta = dict(method=args.name, weights=str(weights), weights_sha256=sha256(weights),
                weights_bytes=weights.stat().st_size, restore_parameters=params,
                official_repo='https://github.com/sunshangquan/Histoformer',
                official_commit='1f045f06c03551c31504d8042dbe6ff9b9569108',
                config='third_party/histoformer/Allweather/Options/Allweather_Histoformer.yml',
                preprocessing='BGR->RGB, /255, reflect pad to multiple of 8, clamp[0,1], ->BGR uint8',
                adaptation=dict(tiling=True, tile=args.tile, overlap=args.overlap,
                                fusion='hann_no_zero_endpoint_exact_denominator',
                                fusion_version='np2r-1',
                                fusion_note='分子分母同权重，分母不截断；断言全覆盖；恒等模型逐像素还原（见 fusion_identity_check.json）',
                                dtype='bfloat16' if args.bf16 else 'float32',
                                reason='全幅 1920x1080 在 8GB 显存溢出到统一内存，实测 214s/图、峰值 14985MiB'),
                images=len(images), weathers=sorted(args.weather),
                mean_restore_ms=float(np.mean(timings)), median_restore_ms=float(np.median(timings)),
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2 ** 20 if device == 'cuda' else None,
                torch=torch.__version__, numpy=np.__version__, opencv=cv2.__version__,
                device=torch.cuda.get_device_name() if device == 'cuda' else 'cpu', per_image=records)
    (out_dir / 'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print('完成：', out_dir)


if __name__ == '__main__':
    main()
