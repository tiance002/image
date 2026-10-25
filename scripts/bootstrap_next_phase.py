"""配对、按天气分层的图像 bootstrap：复用官方 COCO 逐图匹配。"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from saferestore.metrics.next_phase import normalize_ground_truth,coco_evaluate,resample_summary,runtime_counts


def image_runtime(evaluator, input_hits):
    ids=evaluator.params.imgIds
    rows={i:dict(tp=0,fp=0,gt=0,kept=0,denominator=0) for i in ids}
    input_gt_by_image={i:set() for i in ids}
    for annotation in evaluator.cocoGt.anns.values():
        if annotation['id'] in input_hits:
            input_gt_by_image[annotation['image_id']].add(annotation['id'])
    for item in evaluator.evalImgs:
        if item is None: continue
        valid=~item['dtIgnore'][0]
        matches=item['dtMatches'][0]
        bucket=rows[item['image_id']]
        bucket['tp']+=int(np.count_nonzero((matches>0)&valid))
        bucket['fp']+=int(np.count_nonzero((matches==0)&valid))
        bucket['gt']+=int(np.count_nonzero(item['gtIgnore']==0))
        bucket['kept']+=len(set(map(int,matches[(matches>0)&valid]))&input_hits)
    return np.array([[r['tp'],r['fp'],r['gt'],r['kept'],len(input_gt_by_image[i])] for i,r in rows.items()],dtype=float)


def runtime_summary(array,indices):
    tp,fp,gt,kept,denom=array[indices].sum(0)
    return dict(recall=tp/max(gt,1),fp_per_image=fp/len(indices),opr=kept/denom if denom else np.nan)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs',nargs='+',default=['v4:input','guard:input','C1:C0'])
    parser.add_argument('--resamples',type=int,default=1000)
    parser.add_argument('--out',default='results/next_phase/bootstrap.json')
    args=parser.parse_args()
    payload=json.loads((ROOT/'data/raw/ACDC_detection/gt_detection/instancesonly_val_gt_detection.json').read_text(encoding='utf-8'))
    images,annotations=normalize_ground_truth(payload)
    folders=ROOT/'results/next_phase/predictions'
    methods=sorted(set(m for pair in args.pairs for m in pair.split(':'))|{'input'})
    predictions={m:{kind:json.loads((folders/m/f'{kind}.json').read_text(encoding='utf-8'))['per_image'] for kind in ('ap001','runtime025')} for m in methods}
    result=dict(resamples=args.resamples,seed=20261025,unit='paired image, stratified by weather, percentile95',
        limitation='相邻帧相关；图像重采样不构成场景独立或多随机种子证据。区间跨零表示证据不足，不证明无损。无真值类别重采样记为NaN并报告有效次数。',scopes={})
    for scope in ('rain_fog200','all406'):
        subset=[i for i in images if scope=='all406' or i['file_name'].split('/')[0] in ('fog','rain')]
        ids={i['id'] for i in subset};gt=[a for a in annotations if a['image_id'] in ids]
        evaluators={m:{kind:coco_evaluate(subset,gt,[r for r in predictions[m][kind] if r['image_id'] in ids],.001 if kind=='ap001' else .25,100 if kind=='ap001' else 300) for kind in ('ap001','runtime025')} for m in methods}
        input_hits=runtime_counts(evaluators['input']['runtime025'])['hits']
        arrays={m:image_runtime(evaluators[m]['runtime025'],input_hits) for m in methods}
        strata=[[j for j,i in enumerate(subset) if i['file_name'].split('/')[0]==w] for w in ('fog','rain','night','snow')]
        strata=[s for s in strata if s]
        rng=np.random.default_rng(20261025)
        samples={pair:{} for pair in args.pairs}
        full_indices=np.arange(len(subset))
        points={m:{**resample_summary(evaluators[m]['ap001'],full_indices),**runtime_summary(arrays[m],full_indices)} for m in methods}
        for index in range(args.resamples):
            indices=np.concatenate([rng.choice(s,len(s),replace=True) for s in strata])
            metrics={m:{**resample_summary(evaluators[m]['ap001'],indices),**runtime_summary(arrays[m],indices)} for m in methods if any(m in p.split(':') for p in args.pairs)}
            for pair in args.pairs:
                candidate,reference=pair.split(':')
                for key in metrics[candidate]:
                    samples[pair].setdefault(key,[]).append(metrics[candidate][key]-metrics[reference][key])
            if (index+1)%100==0: print(scope,'bootstrap',index+1,'/',args.resamples,flush=True)
        result['scopes'][scope]={}
        for pair,values in samples.items():
            candidate,reference=pair.split(':');summary={}
            for key,values in values.items():
                finite=np.asarray(values)[np.isfinite(values)]
                lo,hi=np.quantile(finite,[.025,.975]) if len(finite) else (None,None)
                delta=points[candidate][key]-points[reference][key]
                summary[key]=dict(delta=float(delta) if np.isfinite(delta) else None,ci95=[float(lo),float(hi)] if lo is not None else None,valid_resamples=len(finite),sign_positive=bool(lo>0) if lo is not None else False)
            result['scopes'][scope][pair]=summary
        Path(ROOT/args.out).write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':main()
