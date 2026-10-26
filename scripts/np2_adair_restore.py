"""NP2：AdaIR-5D 官方检查点推理入口（当前权重不可得，脚本先就绪）。

官方权重仅托管在 Google Drive（README: https://drive.google.com/drive/folders/1x2LN4kWkO3S65jJlH-1INUFiYt8KFzPH）。
本机到 drive.google.com / docs.google.com / googleapis.com 全部连接失败（curl 返回 000，CONNECT 被拒/超时），
本脚本在缺少真实权重时**直接报错退出**，不会用随机初始化继续推理。

与官方的差异（记录为适配）：
- 官方测试集用 crop_img(base=16) 中心裁剪，会丢掉最多 15 像素边缘并改变坐标原点；
  这里改为 reflect pad 到 16 的倍数、前向后裁回原尺寸，保证输出与原图逐像素对齐。
- 官方入口依赖 pytorch-lightning（load_from_checkpoint）；这里直接构建 AdaIR 结构并 strict 加载 state_dict，
  只用到 torch + einops，不引入训练栈。

适用范围（预先固定，不看验证集挑权重）：fog / rain / night 用同一个 adair5d.ckpt；snow 明确不支持，标 N/A。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from scripts.np2_histoformer_restore import read_rgb, write_png  # 复用同一套中文路径安全 I/O

OFFICIAL_FOLDER = 'https://drive.google.com/drive/folders/1x2LN4kWkO3S65jJlH-1INUFiYt8KFzPH'
OFFICIAL_REPO = 'https://github.com/c-yn/AdaIR'
OFFICIAL_COMMIT = 'ccb8b98e49614e07badd0641e5163fa7635c2f02'
SUPPORTED = ('fog', 'rain', 'night')


def load_net(weights: Path, device: str):
    import importlib.util
    if not weights.is_file():
        raise FileNotFoundError(
            f'缺少 AdaIR 官方权重：{weights}\n'
            f'官方地址 {OFFICIAL_FOLDER}（本机对该域名连接失败，见 EXTERNAL_BASELINES.md 的阻断记录）\n'
            f'按任务要求不使用随机初始化代替真实权重。')
    spec = importlib.util.spec_from_file_location('adair_model', ROOT / 'third_party/adair/net/model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    net = module.AdaIR(inp_channels=3, out_channels=3, dim=48, num_blocks=[4, 6, 6, 8],
                       num_refinement_blocks=4, heads=[1, 2, 4, 8],
                       ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias', decoder=True)
    ckpt = torch.load(weights, map_location='cpu', weights_only=False)
    state = ckpt.get('state_dict', ckpt)
    # Lightning 保存的键可能带 net. / model. 前缀，先尝试直接加载，再尝试去前缀
    candidates = [state]
    for prefix in ('net.', 'model.', '_forward_module.'):
        if any(k.startswith(prefix) for k in state):
            candidates.append({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)})
    last_error = None
    for candidate in candidates:
        try:
            net.load_state_dict(candidate, strict=True)
            return net.to(device).eval(), sum(p.numel() for p in net.parameters())
        except Exception as error:  # noqa: BLE001 - 需要收集所有尝试的失败原因
            last_error = error
    raise RuntimeError(f'严格加载失败（不允许随机初始化继续）：{last_error}')


def restore_full(net, rgb: np.ndarray, device: str, base: int = 16):
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy())[None].to(device)
    h, w = tensor.shape[-2:]
    pad_h = (base - h % base) % base
    pad_w = (base - w % base) % base
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), 'reflect')
    with torch.inference_mode():
        out = net(tensor)
    return out[:, :, :h, :w].clamp(0, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--name', default='adair5d')
    parser.add_argument('--weather', nargs='+', default=list(SUPPORTED))
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--out-root', default='results/np2_external_probe/restored')
    args = parser.parse_args()

    unsupported = set(args.weather) - set(SUPPORTED)
    if unsupported:
        raise ValueError(f'AdaIR-5D 不包含这些天气，禁止冒充：{sorted(unsupported)}')

    weights = Path(args.weights)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net, params = load_net(weights, device)
    payload = json.loads((ROOT / 'data/raw/ACDC_detection/gt_detection/instancesonly_val_gt_detection.json').read_text(encoding='utf-8'))
    images = [i for i in sorted(payload['images'], key=lambda x: x['id']) if i['file_name'].split('/')[0] in args.weather]
    if args.limit:
        images = images[:args.limit]

    out_dir = ROOT / args.out_root / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, image in enumerate(images):
        rgb = read_rgb(ROOT / 'data/raw/ACDC_images/rgb_anon' / image['file_name']).astype(np.float32) / 255.0
        restored = restore_full(net, rgb, device)
        restored = restored[0].permute(1, 2, 0).float().cpu().numpy()
        write_png(out_dir / image['file_name'], (restored * 255).round().clip(0, 255).astype(np.uint8))
        records.append(dict(image_id=image['id'], file_name=image['file_name']))
        if (index + 1) % 20 == 0:
            print(f'{args.name} {index+1}/{len(images)}', flush=True)

    meta = dict(method=args.name, weights=str(weights),
                weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                official_repo=OFFICIAL_REPO, official_commit=OFFICIAL_COMMIT,
                official_weight_folder=OFFICIAL_FOLDER, restore_parameters=params,
                preprocessing='RGB [0,1]; reflect pad to multiple of 16 (官方为中心裁剪); clamp[0,1]',
                supported_weather=list(SUPPORTED), snow='N/A 官方五退化不含去雪',
                images=len(images), per_image=records)
    (out_dir / 'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print('完成：', out_dir)


if __name__ == '__main__':
    main()
