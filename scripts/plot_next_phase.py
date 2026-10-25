"""用本地 matplotlib 导出可分享的 PR、训练损失和梯度曲线。"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/next_phase'


def main():
    data=json.loads((OUT/'metrics.json').read_text(encoding='utf-8'))
    plots=OUT/'plots';plots.mkdir(exist_ok=True)
    names=[n for n in ('input','v4','guard','C0','C1') if n in data['methods']]
    colors=dict(input='#222222',v4='#377eb8',guard='#984ea3',C0='#ff7f00',C1='#00876c')
    for scope in ('rain_fog200','all406'):
        fig,axes=plt.subplots(2,4,figsize=(15,7),sharex=True,sharey=True)
        for axis,cls in zip(axes.flat,('person','bicycle','car','motorcycle','bus','train','truck')):
            for name in names:
                row=data['methods'][name]['scopes'][scope]['standard']
                values=row['per_class'][cls]['pr50']
                if max(values)>=0:axis.plot(row['recall_grid'],values,label=name,color=colors[name],linewidth=1.4)
            count=data['methods']['input']['scopes'][scope]['standard']['per_class'][cls]['gt']
            axis.set_title(f'{cls} (GT={count})');axis.set_xlim(0,1);axis.set_ylim(0,1.02)
            axis.grid(alpha=.2);axis.set_xlabel('Recall');axis.set_ylabel('Precision')
        handles,labels=axes.flat[0].get_legend_handles_labels()
        axes.flat[-1].axis('off');axes.flat[-1].legend(handles,labels,loc='center',frameon=False)
        fig.suptitle(f'ACDC validation {scope}: COCO AP50 PR, conf>=0.001 (model selection, not independent test)')
        fig.tight_layout();fig.savefig(plots/f'pr50_{scope}.png',dpi=180);plt.close(fig)
    logs={}
    for name in ('C0','C1'):
        path=ROOT/f'runs/gt_next_phase_20261025_{name}/train_log.csv'
        if path.exists():
            with path.open(encoding='utf-8') as f:logs[name]=list(csv.DictReader(f))
    if logs:
        fig,axes=plt.subplots(2,2,figsize=(13,8))
        for name,rows in logs.items():
            x=np.array([int(r['step']) for r in rows]); stride=max(len(x)//400,1)
            for axis,key in zip(axes.flat,('restoration_total','task_total','grad_restoration_l2','grad_task_weighted_applied_l2')):
                values=np.array([float(r[key]) for r in rows])
                # 10步滑动平均仅用于展示，原始逐步数值仍保存在CSV。
                window=min(10,len(values)); smooth=np.convolve(values,np.ones(window)/window,mode='valid')
                axis.plot(x[window-1:][::stride],smooth[::stride],label=name,color=colors[name],linewidth=1)
                axis.set_title(key+' (10-step moving mean)');axis.set_xlabel('New optimization steps');axis.grid(alpha=.2)
                axis.legend()
        fig.suptitle('Matched C0/C1 training: native task gradients measured in both arms')
        fig.tight_layout();fig.savefig(plots/'training_losses_gradients.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
