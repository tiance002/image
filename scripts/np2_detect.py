"""NP2：对已生成的复原图跑冻结 YOLOv8n，产出与 next_phase 完全同构的逐图预测。

设计上刻意只依赖项目原有环境（ultralytics/torch/cv2），不加载任何外部复原库，
避免 einops/basicsr 等依赖影响检测器行为。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import cv2
import numpy as np
import torch
import ultralytics
import yaml
from ultralytics import YOLO

from saferestore.metrics.next_phase import CLASSES


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_bgr(path: Path) -> np.ndarray:
    buf = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'无法解码：{path}')
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True, help='方法名，同时是预测输出目录名')
    parser.add_argument('--images', required=True, help='复原图根目录（与 ACDC val 相对路径同构）')
    parser.add_argument('--out-root', default='results/np2_external_probe/predictions')
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / 'configs/next_phase_evaluation.yaml').read_text(encoding='utf-8'))
    payload = json.loads((ROOT / config['annotations']).read_text(encoding='utf-8'))
    image_root = Path(args.images)
    # 只评估该目录真实存在的图，天气与 ID 严格取自官方 GT 清单
    images = [i for i in sorted(payload['images'], key=lambda x: x['id']) if (image_root / i['file_name']).is_file()]
    if not images:
        raise FileNotFoundError(f'{image_root} 下没有任何匹配的 val 图像')

    out = ROOT / args.out_root / args.name
    if out.exists():
        raise FileExistsError(f'不覆盖已有预测：{out}')
    out.mkdir(parents=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    weights = ROOT / config['model']
    detector = YOLO(str(weights))
    detector_params = sum(p.numel() for p in detector.model.parameters())

    all_rows, all_runtime, timing = [], [], []
    for index, image in enumerate(images):
        source = read_bgr(image_root / image['file_name'])
        if source.shape[1] != image['width'] or source.shape[0] != image['height']:
            raise ValueError(f'尺寸与标注不一致：{image["file_name"]} {source.shape} vs {image["width"]}x{image["height"]}')
        rows = []
        for confidence in (config['ap_confidence'], config['runtime_confidence']):
            if device == 'cuda':
                torch.cuda.synchronize()
            begin = time.perf_counter()
            result = detector.predict(source=source, imgsz=640, conf=confidence,
                iou=config['nms_iou'], agnostic_nms=False, max_det=config['max_det'],
                classes=list(CLASSES), device=0 if device == 'cuda' else 'cpu',
                rect=True, verbose=False)[0]
            if device == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin
            predictions = {}
            for box, score, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
                predictions.setdefault(str(int(cls)), []).append(dict(box=box.tolist(), confidence=float(score)))
            rows.append(dict(image_id=image['id'], file_name=image['file_name'],
                             weather=image['file_name'].split('/')[0], width=image['width'],
                             height=image['height'], predictions=predictions))
            if confidence == config['runtime_confidence']:
                timing.append(dict(image_id=image['id'], detect_runtime_ms=elapsed * 1000))
        all_rows.append(rows[0])
        all_runtime.append(rows[1])
        if (index + 1) % 25 == 0 or index + 1 == len(images):
            print(f'{args.name} detect {index+1}/{len(images)}', flush=True)

    for filename, rows in [('ap001.json', all_rows), ('runtime025.json', all_runtime)]:
        (out / filename).write_text(json.dumps(dict(protocol=config, per_image=rows), ensure_ascii=False), encoding='utf-8')
    times = timing[5:] if len(timing) > 5 else timing
    metadata = dict(method=args.name, restored_from=str(image_root),
        detector_sha256=sha(weights), annotation_sha256=sha(ROOT / config['annotations']),
        protocol=config, python=platform.python_version(), torch=torch.__version__,
        numpy=np.__version__, ultralytics=ultralytics.__version__, opencv=cv2.__version__,
        images=len(images), detector_parameters=detector_params,
        device=torch.cuda.get_device_name() if device == 'cuda' else 'cpu',
        mean_detect_runtime_ms=float(np.mean([t['detect_runtime_ms'] for t in times])),
        median_detect_runtime_ms=float(np.median([t['detect_runtime_ms'] for t in times])),
        per_image_timing=timing)
    (out / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print('完成：', out)


if __name__ == '__main__':
    main()
