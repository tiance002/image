"""固定每天气前10张，测含文件解码与CPU框回传的实际端到端延迟。"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.train_restorer import imread_rgb,restore_tiled
from saferestore.models.nafnet import build_nafnet
from scripts.evaluate_next_phase import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods',nargs='+',default=['input','v4','guard','C0','C1'])
    parser.add_argument('--per-weather',type=int,default=10)
    args=parser.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(1)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    payload=json.loads((ROOT/'data/raw/ACDC_detection/gt_detection/instancesonly_val_gt_detection.json').read_text(encoding='utf-8'))
    images=sorted(payload['images'],key=lambda i:i['file_name'])
    subset=[i for w in ('fog','rain','night','snow') for i in [v for v in images if v['file_name'].split('/')[0]==w][:args.per_weather]]
    # 所有方法都在相同已预读的文件集合上计时；读回/解码仍在pipeline计时内。
    for image in subset:
        (ROOT/'data/raw/ACDC_images/rgb_anon'/image['file_name']).read_bytes()
    paths=dict(input=None,v4='runs/real_v4_allweather/latest.pt',guard='runs/confidence_v8_20261025_guard/latest.pt',
        C0='runs/gt_next_phase_20261025_C0/latest.pt',C1='runs/gt_next_phase_20261025_C1/latest.pt')
    output=dict(selection='Each weather first10 paths lexicographically, GT-independent; five warmup images, then40 timed images; warm filesystem cache',
        included='file read/decode, RGB conversion, H2D, bf16 NAFNet tiled restoration, quantize/D2H, BGR conversion, FP32 YOLO640 .25 NMS .7 maxDet300, CPU boxes transfer',
        excluded='one-off model loading, weight hashing, GUI rendering, saving images or JSON',methods={})
    for name in args.methods:
        if name not in paths:raise ValueError(name)
        model=None
        if paths[name]:
            state=torch.load(ROOT/paths[name],map_location='cpu',weights_only=False)
            model=build_nafnet().to(device).eval();model.load_state_dict(state['ema']);del state
        weights=ROOT/'yolov8n.pt'
        if not weights.is_file():raise FileNotFoundError('缺少本地YOLO权重，禁止自动下载')
        detector=YOLO(str(weights))
        detector_parameters=sum(p.numel() for p in detector.model.parameters())
        def pipeline(image):
            rgb=imread_rgb(ROOT/'data/raw/ACDC_images/rgb_anon'/image['file_name'])
            if model is not None:
                tensor=torch.from_numpy(rgb.transpose(2,0,1).copy())[None].to(device)
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=device=='cuda'):
                    restored=restore_tiled(model,tensor)
                if not torch.isfinite(restored).all():raise FloatingPointError('计时推理输出非有限')
                rgb=(restored[0].float().clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
            else:rgb=(rgb*255).round().astype(np.uint8)
            result=detector.predict(source=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),imgsz=640,conf=.25,iou=.7,
                classes=[0,1,2,3,5,6,7],max_det=300,agnostic_nms=False,rect=True,verbose=False,
                device=0 if device=='cuda' else 'cpu')[0]
            return result.boxes.data.cpu().numpy()
        for image in subset[:5]:pipeline(image)
        if device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        records=[]
        for index,image in enumerate(subset):
            if device=='cuda':torch.cuda.synchronize()
            start=time.perf_counter();boxes=pipeline(image)
            if device=='cuda':torch.cuda.synchronize()
            records.append(dict(file_name=image['file_name'],weather=image['file_name'].split('/')[0],
                latency_ms=(time.perf_counter()-start)*1000,detections=len(boxes)))
        values=[r['latency_ms'] for r in records]
        output['methods'][name]=dict(n=len(records),mean_ms=float(np.mean(values)),median_ms=float(np.median(values)),p95_ms=float(np.quantile(values,.95)),
            restoration_parameters=sum(p.numel() for p in model.parameters()) if model else 0,detector_parameters=detector_parameters,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else None,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if device=='cuda' else None,
            checkpoint_sha256=sha(ROOT/paths[name]) if paths[name] else None,per_image=records)
        (ROOT/'results/next_phase/benchmark.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
        print(name,'benchmark n',len(values),'mean_ms',np.mean(values),flush=True)
        del model,detector
        if device=='cuda':torch.cuda.empty_cache()


if __name__=='__main__':main()
