"""只在完整公平续训成功后运行固定后评测，不选择权重、不续训或发布。"""
from __future__ import annotations
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/next_phase'


def main():
    training=json.loads((ROOT/'runs/gt_next_phase_20261025/comparison_plan.json').read_text(encoding='utf-8'))
    required=('config_match','batch_match','initialization_match','detector_unchanged')
    if training['status']!='completed' or not all(training.get(k) for k in required):
        raise RuntimeError('两组4000步与公平性核验尚未完成，拒绝后评测')
    if any(r['steps']!=4000 for r in training['arm_results']):raise RuntimeError('预算不是每组4000步')
    evaluation=dict(status='running',fixed_checkpoint='EMA latest@51999, no selection by validation',completed=[])
    def save():
        (OUT/'evaluation_plan.json').write_text(json.dumps(evaluation,ensure_ascii=False,indent=2),encoding='utf-8')
    save()
    def run(key,arguments):
        evaluation['current']=key;save();print('START',key,flush=True)
        with (OUT/(key+'.log')).open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-u',*arguments],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        evaluation['completed'].append(key);save();print('DONE',key,flush=True)
    protocol=yaml.safe_load((ROOT/'configs/next_phase_evaluation.yaml').read_text(encoding='utf-8'))
    for arm in ('C0','C1'):
        checkpoint=f'runs/gt_next_phase_20261025_{arm}/latest.pt'
        folder=OUT/'predictions'/arm
        if folder.exists():
            metadata=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
            digest=hashlib.sha256((ROOT/checkpoint).read_bytes()).hexdigest()
            if metadata['images']!=406 or metadata['checkpoint_sha256']!=digest or metadata['protocol']!=protocol:
                raise RuntimeError(f'{arm}已有预测与计划不一致；不覆盖')
            for filename in ('ap001.json','runtime025.json'):
                rows=json.loads((folder/filename).read_text(encoding='utf-8'))['per_image']
                if len(rows)!=406 or len({r['image_id'] for r in rows})!=406:raise RuntimeError('已有预测不完整')
            print('REUSE',arm,'verified complete predictions',flush=True)
            evaluation['completed'].append('eval_'+arm)
            evaluation.setdefault('reused_predictions',[]).append(arm)
            save()
        else:
            run('eval_'+arm,['scripts/evaluate_next_phase.py','--name',arm,'--checkpoint',checkpoint])
    methods=['input','v4','guard','C0','C1']
    run('score_final',['scripts/evaluate_next_phase.py','--score',*methods])
    run('failure_final',['scripts/analyze_next_phase_failures.py','--methods',*methods])
    # CPU bootstrap与只测质量的GPU推理独立。计时留到bootstrap结束后，避免后台CPU负载影响延迟。
    bootstrap_arguments=['scripts/bootstrap_next_phase.py','--pairs','C1:C0','C1:input','C0:input','C1:v4',
        '--resamples','1000','--out','results/next_phase/bootstrap_gt.json']
    with (OUT/'bootstrap_gt.log').open('w',encoding='utf-8') as bootstrap_log:
        background=subprocess.Popen([sys.executable,'-u',*bootstrap_arguments],cwd=ROOT,stdout=bootstrap_log,stderr=subprocess.STDOUT)
        evaluation['background_bootstrap']=True;save();print('START bootstrap_gt (CPU)',flush=True)
        try:
            run('quality_final',['scripts/evaluate_next_phase_quality.py','--methods',*methods])
            code=background.wait()
            if code:raise subprocess.CalledProcessError(code,bootstrap_arguments)
        except BaseException:
            if background.poll() is None:background.terminate();background.wait()
            raise
        evaluation['background_bootstrap']=False;evaluation['completed'].append('bootstrap_gt');save()
        print('DONE bootstrap_gt',flush=True)
    run('benchmark_final',['scripts/benchmark_next_phase.py','--methods',*methods])
    run('reports_final',['scripts/report_next_phase.py','--training'])
    run('plots_final',['scripts/plot_next_phase.py'])
    evaluation['status']='completed';evaluation.pop('current',None);save()
    print('All fixed post-training evaluations completed; external baselines remain BLOCKED. No push/deploy.',flush=True)


if __name__=='__main__':main()
