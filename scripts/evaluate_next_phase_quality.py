"""仅在同一源图合成、像素对齐的 RGBv2 dev 上测 PSNR/SSIM。"""
from __future__ import annotations
import argparse
import csv
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.train_restorer import imread_rgb,restore_tiled
from saferestore.models.nafnet import build_nafnet


def quality_metrics(prediction,target):
    prediction=prediction.float();target=target.float()
    mse=(prediction-target).square().mean()
    # 11x11 sigma1.5 Gaussian，valid区域，RGB三通道等权，与常见SSIM约定一致。
    coords=torch.arange(11,device=prediction.device,dtype=torch.float32)-5
    weights=torch.exp(-coords.square()/(2*1.5**2));weights/=weights.sum()
    kernel=(weights[:,None]*weights[None,:]).view(1,1,11,11).repeat(3,1,1,1)
    blur=lambda t:F.conv2d(t,kernel,groups=3)
    mx,my=blur(prediction),blur(target)
    vx,vy=blur(prediction.square())-mx.square(),blur(target.square())-my.square()
    cov=blur(prediction*target)-mx*my
    ssim=((2*mx*my+.01**2)*(2*cov+.03**2)/((mx.square()+my.square()+.01**2)*(vx+vy+.03**2))).mean()
    return dict(psnr=float(-10*torch.log10(mse.clamp_min(1e-12))),ssim=float(ssim),mse=float(mse))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods',nargs='+',default=['input','v4','guard','C0','C1'])
    parser.add_argument('--manifest',default='data/real_weather_rgb_v2/manifests/dev.csv')
    parser.add_argument('--out',default='results/next_phase/image_quality.json')
    args=parser.parse_args()
    torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    with (ROOT/args.manifest).open(encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    checkpoints=dict(input=None,v4='runs/real_v4_allweather/latest.pt',guard='runs/confidence_v8_20261025_guard/latest.pt',
        C0='runs/gt_next_phase_20261025_C0/latest.pt',C1='runs/gt_next_phase_20261025_C1/latest.pt')
    result=dict(manifest=args.manifest,n=len(rows),metric_protocol='RGB float range0..1, PSNR allpixels, SSIM gaussian11 sigma1.5 valid-window; no extra image resize',
        evidence_scope='同一清晰源图合成退化的像素对齐dev（fog100/rain100），不是ACDC真实跨时刻weather/reference配对；不能推断真实夜雪画质',
        lpips=dict(status='BLOCKED',reason='本地没有LPIPS包及预训练感知权重；禁止自动下载，不用随机网络冒充'),methods={})
    for name in args.methods:
        if name not in checkpoints: raise ValueError(name)
        model=None
        if checkpoints[name]:
            path=ROOT/checkpoints[name]
            if not path.is_file(): raise FileNotFoundError(path)
            state=torch.load(path,map_location='cpu',weights_only=False)
            model=build_nafnet().to(device).eval();model.load_state_dict(state['ema']);del state
        scores=[]
        for index,row in enumerate(rows):
            source=torch.from_numpy(imread_rgb(ROOT/row['degraded_path']).transpose(2,0,1).copy())[None].to(device)
            target=torch.from_numpy(imread_rgb(ROOT/row['clean_path']).transpose(2,0,1).copy())[None].to(device)
            if source.shape!=target.shape: raise ValueError('非像素对齐尺寸')
            with torch.inference_mode():
                if model:
                    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device=='cuda'): prediction=restore_tiled(model,source)
                    if not torch.isfinite(prediction).all():raise FloatingPointError('画质评测输出非有限')
                    prediction=(prediction.float().clamp(0,1)*255).round()/255
                else: prediction=source
                values=quality_metrics(prediction,target)
            scores.append(dict(scene_id=row['scene_id'],weather=row['weather'],**values))
            if (index+1)%50==0:print(name,'quality',index+1,'/',len(rows),flush=True)
        summary={}
        for scope in ('all','fog','rain'):
            subset=scores if scope=='all' else [r for r in scores if r['weather']==scope]
            summary[scope]={key:float(np.mean([r[key] for r in subset])) for key in ('psnr','ssim')}
        result['methods'][name]=dict(checkpoint=checkpoints[name],summary=summary,per_image=scores)
        (ROOT/args.out).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        del model
        if device=='cuda': torch.cuda.empty_cache()


if __name__=='__main__':main()
