"""Generate the six-section NP2-R report only after all real inputs are complete.

Read-only source evidence; writes only a NEW REVISION_REPORT.md. No inference,
training, network, or modification of historical np2_external_probe artifacts.
The final single-experiment recommendation must be supplied after results review.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/np2_revision'
LEGACY = ROOT / 'results/np2_external_probe'
FIX = 'histoformer_real_fusionfix'
OLD = 'histoformer_real_legacy_invalidated'
NAMES = {'0':'person','1':'bicycle','2':'car','3':'motorcycle','5':'bus','6':'train','7':'truck'}
METHODS = ('input','v4','C0','C1',FIX,OLD)
LABELS = {'input':'input','v4':'v4','C0':'C0','C1':'C1',FIX:'Histoformer修复后',OLD:'Histoformer旧缺陷版（无效）'}


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1 << 20),b''):
            value.update(chunk)
    return value.hexdigest()


def num(value, digits=6):
    if value is None:
        return 'N/A'
    if isinstance(value,bool):
        return '通过' if value else '未通过'
    if not math.isfinite(float(value)):
        raise ValueError(f'Nonfinite report value: {value}')
    return f'{value:.{digits}f}'


def table(headers, rows):
    def safe(value):
        return str(value).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(safe,headers))+' |',
                     '| '+' | '.join(['---']*len(headers))+' |',
                     *['| '+' | '.join(map(safe,row))+' |' for row in rows]])+'\n'


def ci(value):
    return num(value['delta'])+' ['+', '.join(num(v) for v in value['ci95'])+']' if value['ci95'] else 'N/A'


def evidence():
    paths = {name:OUT/filename for name,filename in {
        'supervision':'supervision_probe.json','metrics':'metrics.json','bootstrap':'bootstrap.json',
        'scale':'scale_order_probe.json','quality':'quality_lpips_fusionfix.json',
        'tiling':'tiling_consistency_fusionfix.json','identity':'fusion_identity_check.json',
        'preflight':'preflight.json','zero':'scale_zero_bus_anchor_centers.json',
        'checks':'scale_probe_cpu_checks.json','png_changes':'fusion_png_changes.json',
        'numerics':'supervision_tf32_diagnosis.json','failed':'supervision_failed_attempt.json'}.items()}
    missing=[str(p.relative_to(ROOT)) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError('Incomplete NP2-R inputs: '+', '.join(missing))
    data={key:load(path) for key,path in paths.items()}
    sup=data['supervision']; scale=data['scale']; quality=data['quality']
    assert sup['status']=='completed' and set(sup['models'])=={'v4','C0','C1'}
    assert sup['manifest']['counts']['total']==64 and sup['manifest']['legacy_manifest_exact_match']
    assert sup['manifest']['precision']['backward_only_tf32_disabled']
    assert sup['manifest']['precision']['validation_tolerance']==1e-4
    for model in sup['models'].values():
        assert model['batch_consistency']['batches']==16 and model['batch_consistency']['all_batches_pass']
        assert model['gradient']['batches']==64 and model['gradient']['batch_size']==1 and model['gradient']['all_pass']
        assert all(model['invariants'][k] for k in ('model_unchanged','detector_including_bn_unchanged','checkpoint_file_unchanged','detector_file_unchanged','detector_has_no_grad'))
    assert scale['all_completed'] and set(scale['models'])=={'v4','C0','C1'} and scale['sample']['count']==24
    for model in scale['models'].values():
        assert len(model['per_image'])==24 and model['weights_and_bn_unchanged'] and model['detector_weights_and_bn_unchanged']
        assert {'D_fp32','A_fp32','B_fp32','D_u8','A_u8','B_u8'}<=set(model['summaries'])
    assert quality['status']=='completed' and quality['n']==len(quality['per_image'])==200
    assert data['tiling']['status']=='completed' and len(data['tiling']['cases'])==2
    assert data['png_changes']['n']==len(data['png_changes']['per_image'])==200
    assert data['identity']['identity_all_ok'] and len(data['identity']['identity_cases'])==12
    assert data['bootstrap']['resamples']==1000 and set(data['bootstrap']['scopes'])=={'rain','snow'}
    assert data['metrics']['scopes']['rain']['n_images']==100 and data['metrics']['scopes']['snow']['n_images']==100
    assert data['metrics']['scopes']['rain_snow200']['n_images']==200
    assert all(set(scope['methods'])==set(METHODS) for scope in data['metrics']['scopes'].values())
    metadata=load(OUT/'restored'/FIX/'metadata.json')
    detector=load(OUT/'predictions'/FIX/'metadata.json')
    assert metadata['images']==detector['images']==200
    assert metadata['adaptation']['fusion_version']=='np2r-1'
    assert metadata['adaptation']['dtype']=='float32'
    assert len(list((OUT/'restored'/FIX).rglob('*.png')))==200
    assert quality['fingerprint']['weights_sha256']==metadata['weights_sha256']
    assert data['tiling']['weights_sha256']==metadata['weights_sha256']
    pre=data['preflight']
    assert pre['review_baseline']=='3d53ca0dbd9902c86de6953de791a73c557dd2ef' and pre['no_training']
    # Verify actual retained resources and all old evidence, not merely a declaration.
    for rel,expected in pre['frozen_resources'].items():
        assert sha(ROOT/rel)==expected['sha256'],f'Frozen resource changed: {rel}'
    for rel,expected in pre['preexisting_external_probe_files'].items():
        legacy_file=ROOT/rel
        actual=legacy_file.stat()
        assert actual.st_size==expected['size'],f'Legacy evidence size changed: {rel}'
        assert actual.st_mtime_ns==expected['mtime_ns'],f'Legacy evidence timestamp changed: {rel}'
        if 'sha256' in expected:
            assert sha(legacy_file)==expected['sha256'],f'Legacy evidence SHA256 changed: {rel}'
    matrix=list(csv.DictReader((OUT/'bus_train_gt_matrix.csv').open(encoding='utf-8')))
    assert len({(r['image_id'],r['gt_id'],r['protocol']) for r in matrix})==len(matrix)
    rows2730=[r for r in matrix if r['gt_id']=='2730']
    assert len(rows2730)==2 and all(r['source_gt_id']=='2729' for r in rows2730)
    assert next(r for r in rows2730 if r['protocol']=='ap001')['v4']=='hit'
    assert next(r for r in rows2730 if r['protocol']=='runtime025')['v4']=='miss'
    data.update(restoration=metadata,detector=detector,matrix=matrix,
                old_supervision=load(LEGACY/'supervision_probe.json'),
                old_quality=load(LEGACY/'quality_lpips.json'),
                old_scale=load(LEGACY/'scale_order_probe.json'),
                old_tiling=load(LEGACY/'tiling_consistency.json'))
    data['evidence_sha256']={str(p.relative_to(ROOT)):sha(p) for p in paths.values()}
    return data


def render(data,recommendation):
    sup,metrics,boot,scale,quality=(data[k] for k in ('supervision','metrics','bootstrap','scale','quality'))
    old=data['old_supervision']; lines=['# SafeRestore-RF NP2-R 修订报告\n']
    def add(text):lines.append(text+'\n')
    add('审查基线：`tiance002/image` / `next-phase-gt-20261025` / `3d53ca0dbd9902c86de6953de791a73c557dd2ef`。运行区为真实科研工作区；本报告只汇总已完成的修订产物。旧 `np2_external_probe` 全部保留，受影响结论标为“旧版本待修订”。本轮训练步数为0，未执行C2，未替换默认权重。')
    add('## 1. 错误位置、最小复现和修复')
    add(table(['位置','错误及最小复现','修复与验收'],[
        ['旧监督脚本手动box/DFL与gradient_probe','pred_bboxes/anchor_points为网格坐标，target_bboxes为像素；交接针对性反例7.217→13.441（约+86%）','target_bboxes/stride_tensor后与原生BboxLoss一致，保留原生联合分配'],
        ['旧监督脚本weight形状','[N,1]×[N]广播为[N,N]；交接反例7.217→53.764（约+644%）','weight.unsqueeze(-1)，IoU/DFL均保持[N,1]'],
        ['旧reproduction_abs_diff','只校验原生总量，未覆盖手动分解和梯度向量','每个统计批和每个梯度批验证七类之和，比较梯度向量relative L2/cosine/max_abs，超阈值立即报错'],
        ['np2_histoformer_restore.restore_tiled','二维角点权重约1e-12，分母截到1e-8；常量0.5的角点仅约0.00005','无零端点Hann，同权分子/分母且不截分母，全覆盖断言；12个恒等用例逐像素最大差0'],
        ['旧scale A/B','只有B量化、A复原灰边，精度/上下文也不同，不能声称仅改变顺序','主对照共同FP32/INTER_LINEAR/坐标/padding；量化、灰边、bf16和实际rect管线独立命名']]))
    legacy=data['identity']['legacy_repro'][0]
    add(f"恒等复现的原始数值：期望{num(legacy['expected'])}，旧角点{num(legacy['corner_tl'],9)}，比值{num(legacy['corner_ratio'],9)}。修复前局部最大差混入了融合错误，撤回直接归因于上下文的解释。交接已完成的7/7监督针对性测试和12个融合恒等用例沿用；本轮不重复颜色审计、数据生成或C0/C1训练。")
    add('完整GPU诊断首次运行在第一个梯度批触发严格检查：标量误差2.384e-7，向量相对L2误差约0.000674，超过1e-4。失败日志及初始JSON作为旧数值版本保留，没有放宽阈值。固定同一图的受控复现如下：')
    add(table(['前向cuDNN TF32','反向cuDNN TF32','原生损失','分量向量和对原生相对L2','同手动图分量和对总量相对L2'],[
        [case['forward_cudnn_tf32'],case['backward_global_cudnn_tf32'],num(case['native_loss'],10),num(case['native_vs_sum_class_gradients']['relative_l2_error'],10),num(case['same_manual_graph_sumclass_vs_total_gradient']['relative_l2_error'],10)] for case in data['numerics']['cases']]))
    add('只关闭反向卷积TF32，保留cuDNN enabled/benchmark/deterministic设置和历史前向精度，可保持前向损失及分配不变，将误差降至约1e-6。修正版逐批使用这项数值策略；它是诊断反传的精度说明，不宣称复现历史TF32训练梯度的每一位。另修复报告路径变量被batch结果覆盖的写盘问题，并用单图端到端保存验收后才重跑完整诊断。')
    add(table(['固定1024裁剪','旧融合最大差（无效上下文证据）','修复后最大差','修复后均值差','接缝/非接缝均值差'],[
        [case['weather'],num(next(r['max_abs_diff'] for r in data['old_tiling']['cases'] if r['weather']==case['weather'])),num(case['max_abs_diff']),num(case['mean_abs_diff']),num(case['seam_mean_abs_diff'])+' / '+num(case['interior_mean_abs_diff'])]
        for case in data['tiling']['cases']]))
    add('以上是真实网络固定1024裁剪的单块/640分块比较，不是1920全幅等价性证明。修复后残余差可含上下文差，仍不能将任一局部差自动定性为模型效果。')
    add('## 2. 哪些旧结果仍有效，哪些数字被更新')
    add(table(['旧证据','处理','本轮替代'],[
        ['input/v4/C0/C1正式val预测、内部FP32画质','保留且直接复用，不重推理','同子集metrics和quality.references'],
        ['原生联合分配正anchor覆盖、原生总损失','可保留；逐GT/逐批重新核对','修订supervision_probe.json'],
        ['旧逐类box/DFL、grad_norm及充分性推断','旧版本待修订，不可作为有效归因','修正坐标/形状后的损失及向量校验'],
        ['旧Histoformer rain/snow200预测/指标/区间/GT变化和画质','融合缺陷版数字失效；只作修复前后历史参照','fusionfix 200张、metrics/bootstrap/quality_lpips_fusionfix'],
        ['旧尺度A/B结果','仅保留为混杂历史管线，不能隔离顺序','固定24张主D/A/B与独立扰动'],
        ['旧CSV内部模型的运行时hit/miss','conf=.25仍有效，不能混用AP候选口径','双协议矩阵、统一评估ID和source_gt_id']]))
    add(f"`preflight.json`记录的{len(data['preflight']['preexisting_external_probe_files'])}个历史文件的大小/mtime在生成本报告时核对一致；其中记录了SHA256的文件还核对内容哈希一致；{len(data['preflight']['frozen_resources'])}项权重资源逐一SHA256核对未变。64张固定train样本与旧manifest完全相同；24张尺度样本也逐文件复用。检查点及检测器参数/BN的前后不变校验通过，未创建优化器。")
    add('## 3. 修正前后逐类损失与梯度')
    add('下表在同一固定64张、统计batch=4下比较。旧box/DFL列是有坐标与广播缺陷的原始“未乘gain”数；新box/DFL除以对应gain后列出，以统一显示单位。分类正/负BCE均为除以同批target_scores_sum的未乘gain量。它们是16个批次标量之和，不是64图均值。旧数只用于暴露误差，不能作训练监督强度的有效证据。')
    rows=[]
    for name,new in sup['models'].items():
        for code,label in NAMES.items():
            a=old['models'][name]['per_class'][code];b=new['per_class'][code]
            rows.append([name,label,num(a['cls_pos_contribution'])+' → '+num(b['cls_pos_ungained_legacy_units']),num(a['cls_neg_contribution'])+' → '+num(b['cls_neg_ungained_legacy_units']),num(a['box_contribution'])+' → '+num(b['box_contribution']/new['gain']['box']),num(a['dfl_contribution'])+' → '+num(b['dfl_contribution']/new['gain']['dfl'])])
    add(table(['模型','类别','分类正：旧→新','分类负：旧→新','box未乘gain：旧→新','DFL未乘gain：旧→新'],rows))
    rows=[]
    for name,new in sup['models'].items():
        for code,label in NAMES.items():
            c=new['per_class'][code];parts=[c[k] for k in ('cls_pos_contribution','cls_neg_contribution','box_contribution','dfl_contribution')]
            rows.append([name,label,*[num(v) for v in parts],num(sum(parts)),f"{c['gt']}/{c['positives']}/{c['zero']}",num(c['target_weight_total'])])
    add('以下是实际原生gain下的修正版分量，可在同批求和恢复原生标量。正anchor来自全部GT一次联合分配；没有删类重分配。')
    add(table(['模型','类别','分类正×gain','分类负×gain','box×gain','DFL×gain','合计','GT/正anchor/零正GT','target_scores总权重'],rows))
    add(table(['模型','原生总量','七类分量总量','统计批最大误差','梯度批数','向量相对L2最大','余弦最小','正anchor逐GT等于旧记录'],[
        [n,num(m['native_loss_total_sum']),num(m['reported_class_component_total']),num(m['batch_consistency']['max_abs_diff'],9),m['gradient']['batches'],num(m['gradient']['grad_relative_l2_max'],9),num(m['gradient']['grad_cosine_min'],9),num(m['legacy_positive_coverage_exact_match'])] for n,m in sup['models'].items()]))
    add('旧grad_norm记录错误手动分解所得的批梯度范数累加，缺少逐批向量；其中all也不是原生总损失梯度，all/bus/train均受错误box/DFL影响。修正版完整64个单图批次同时记录“范数之和”和“范数均值”；两者都不是整批数据联合梯度向量的范数。类别范数/原生范数必须命名“梯度范数占比（不可加和）”。')
    add(table(['模型','对象','旧错误手动损失范数和（all非原生）','新64批范数和','新64批范数均值','新逐批范数比均值（不可加和）'],[
        [n,label,num(old['models'][n]['grad_norm'].get('all' if label=='native' else label)),num(m['gradient']['native_sum_of_batch_norms'] if label=='native' else m['gradient']['per_class_sum_of_batch_norms'][label]),num(m['gradient']['native_mean_norm'] if label=='native' else m['gradient']['per_class_mean_norm'][label]),'1.000000' if label=='native' else num(m['gradient']['per_class_mean_norm_ratio_nonadditive'][label])]
        for n,m in sup['models'].items() for label in ('native','bus','train')]))
    add(table(['模型','类别','新64批范数和','新64批范数均值','新逐批范数比均值（不可加和）'],[
        [n,label,num(m['gradient']['per_class_sum_of_batch_norms'][label]),num(m['gradient']['per_class_mean_norm'][label]),num(m['gradient']['per_class_mean_norm_ratio_nonadditive'][label])]
        for n,m in sup['models'].items() for label in NAMES.values()]))
    add('逐类梯度还包含该类负分类BCE；某张图没有该类正GT时也可能有非零类梯度。正监督覆盖必须结合逐GT正anchor和target_scores解释，不能凭类梯度非零判定监督充分。')
    add('每批图像名、GT原始ID、box/DFL分量、梯度向量相对误差/最大元素差均在JSON保留。固定64张中bus/train存在正anchor和任务梯度，只能证明这些诊断样本中有监督，不能推出整个训练期监督充分；也不能据此排除监督覆盖问题或直接提高全局损失权重。')
    zeros=[r for m in sup['models'].values() for r in m['per_gt'] if r['category']==5 and r['positives']==0]
    add(f"修正版共记录{len(zeros)}条‘模型×零正bus’记录。逐GT忽略区候选/正anchor交集已保存；IgnoreRegionBCE保留正监督，只去掉相应负监督。掩码比例与正样本交集应结合本样本边界解读，不能外推全部训练帧。")
    add('## 4. 修正后的Histoformer检测及GT变化')
    meta=data['restoration']
    add(f"同一官方`net_g_real.pth`权重，SHA256 `{meta['weights_sha256']}`，官方代码版本`{meta['official_commit']}`，FP32、tile640/overlap64、修正融合。实际重推理200张：rain100+snow100。该权重是真实雪版本，snow是对应任务，rain是跨任务迁移压力测试；不能冒充`net_g_best.pth` Allweather，也没有fog/night纯模型成绩。")
    png=data['png_changes']
    add(f"200张均用同权重重新前向后融合，不是放大旧量化PNG。新旧PNG平均绝对差{num(png['mean_abs_u8'])}/255、最大差{png['maximum_abs_u8']}/255。`changed_frac_maxchan_gt_0p02`准确含义是‘逐像素三个通道改变量的最大值超过0.02的像素比例’，修复后200张中位为{num(png['changed_frac_maxchan_gt_0p02_median'])}，不是平均通道差，也不是模型支持范围的证明。")
    examples=[next(r for r in png['per_image'] if r['weather']==w) for w in ('rain','snow')]
    add(table(['天气','image_id','旧四角RGB（缺陷版）','新四角RGB'],[[r['weather'],r['image_id'],str(r['old_corners_rgb']),str(r['new_corners_rgb'])] for r in examples]))
    rows=[]
    for scope,scope_data in metrics['scopes'].items():
        for name in METHODS:
            r=scope_data['methods'][name];a=r['standard'];rt=r['runtime']
            rows.append([scope,LABELS[name],scope_data['n_images'],num(a['map50']),num(a['map50_95']),num(rt['recall']),num(rt['fp_per_image']),num(rt['opr'])])
    add(table(['子集','方法','图数','mAP50','mAP50:95','Recall(.25)','FP/图(.25)','OPR(.25)'],rows))
    add('AP采用conf=.001、COCO IoU.50:.95/maxDets100；运行时采用conf=.25、IoU.50/maxDets300；相同YOLOv8n FP32、类内NMS .7、全局预测maxDet300、七类映射与crowd规则。rain_snow200是明确的适用/迁移联合子集，不是406张全天气纯模型分数。')
    add(table(['子集','方法',*NAMES.values()],[
        [scope,LABELS[name],*[num(bucket['methods'][name]['standard']['per_class'][label]['ap50']) for label in NAMES.values()]] for scope,bucket in metrics['scopes'].items() for name in METHODS]))
    add(table(['天气','修复后−对照','ΔmAP50 [95%CI]','ΔmAP50:95 [95%CI]','Δbus AP50 [95%CI]','Δtrain AP50 [95%CI]','Δbicycle AP50 [95%CI]'],[
        [weather,LABELS[pair.split(':')[1]],*[ci(values[key]) for key in ('map50','map50_95','bus','train','bicycle')]] for weather,pairs in boot['scopes'].items() for pair,values in pairs.items()]))
    add('1000次配对图像percentile bootstrap，seed20261025；少数重采样中某类没有GT，该类有效次数见JSON的valid字段。区间包含零（包括端点为零）不构成等价性或“证明不更差”。帧与同序列相关性仍存在；两个天气独立区间不能拼成200张联合区间。')
    add(table(['子集','修复后Histoformer：相对input新增GT数/ID (.25)','丢失GT数/ID (.25)'],[
        [scope,str(len(bucket['methods'][FIX]['runtime']['gained_input_gt']))+' / '+str(bucket['methods'][FIX]['runtime']['gained_input_gt']),str(len(bucket['methods'][FIX]['runtime']['lost_input_gt']))+' / '+str(bucket['methods'][FIX]['runtime']['lost_input_gt'])] for scope,bucket in metrics['scopes'].items()]))
    add('雪天全类别运行时实际丢失3个输入GT，恢复3个；旧文字“丢失1个”不正确。下表再单独列bus/train；全类别OPR不能误当成bus或train保留率。')
    add(table(['子集','方法','类别','GT','检出(.25)','相对input新增检出GT','相对input丢失GT'],[
        [scope,LABELS[name],label,row['gt'],row['detected'],str(row['recovered']),str(row['lost'])] for scope,bucket in metrics['scopes'].items() for name in METHODS for label,row in bucket['methods'][name]['bus_train_gt'].items()]))
    matrix={}
    for row in data['matrix']:
        matrix.setdefault((row['image_id'],row['gt_id']),{})[row['protocol']]=row
    cases=[]
    for (image_id,gid),protocols in matrix.items():
        ap,rt=protocols['ap001'],protocols['runtime025']
        changed=len({rt[n] for n in METHODS})>1 or ap[FIX]!=ap[OLD] or rt[FIX]!=rt[OLD]
        if changed or gid in ('1269','1673','2730'):
            cases.append([rt['weather'],image_id,gid,rt['source_gt_id'],rt['category'],*[ap[n]+'/'+rt[n] for n in METHODS]])
    add('下表每行按image_id+评估GT_id去重，同一目标各方法并排；每个单元格为“AP conf=.001 / 运行时conf=.25”。评估GT_id=原始source_gt_id+1。全部目标保留在双协议CSV，不按方法重复制作失败样例。')
    add(table(['天气','image_id','评估GT_id','原始GT_id','类别',*[LABELS[n] for n in METHODS]],cases))
    pair2730=matrix[('395','2730')]
    add(f"GT2730（image395，原始标注ID2729）在v4的AP协议conf=.001/maxDets100下为`{pair2730['ap001']['v4']}`，运行时conf=.25/maxDets300下为`{pair2730['runtime025']['v4']}`。修正版Histoformer分别为`{pair2730['ap001'][FIX]}`/`{pair2730['runtime025'][FIX]}`；不能再将AP命中陈述与运行时CSV直接比较。")
    rain=metrics['scopes']['rain']['methods']; snow=metrics['scopes']['snow']['methods']
    recovered=[gid for gid in ('1269','1673') if matrix[next(key for key in matrix if key[1]==gid)]['runtime025'][FIX]=='hit']
    add(f"雨天两个历史原图正确、内部复原丢失的bus目标（评估ID1269、1673），修正版在conf=.25实际恢复{len(recovered)}/2，恢复ID为{recovered}。train/bicycle的逐类变化应按上表和区间判断，不以总体mAP掩盖单类退化。")
    add('有效对齐质量只在固定合成dev fog100/rain100上计算。真实雪没有有效对齐参考；fog属于该真实雪权重的任务外压力测试。旧内部参考是FP32浮点输出，新外部float_mean用于同精度对照；新外部PNG等价量化分数单列，不能混写。')
    rows=[]
    for name,ref in quality['references'].items():
        rows.append([name,'FP32旧有效结果复用',*[num(ref['mean'][k]) for k in ('psnr','ssim','lpips')]])
    oldq=data['old_quality']['methods']['histoformer_real']['mean']
    rows.extend([['Histoformer旧缺陷版（无效）','旧uint8输出',*[num(oldq[k]) for k in ('psnr','ssim','lpips')]],['Histoformer修复后','FP32浮点',*[num(quality['float_mean'][k]) for k in ('psnr','ssim','lpips')]],['Histoformer修复后','uint8 PNG等价',*[num(quality['mean'][k]) for k in ('psnr','ssim','lpips')]]])
    add(table(['方法','输出协议','PSNR↑','SSIM↑','LPIPS↓'],rows))
    rows=[]
    for name,ref in quality['references'].items():
        for weather,row in ref['by_weather'].items():
            rows.append([weather,name,'FP32参考复用',*[num(row[k]) for k in ('psnr','ssim','lpips')]])
    for precision,buckets in [('FP32',quality['float_by_weather']),('uint8',quality['by_weather'])]:
        for weather,row in buckets.items():
            rows.append([weather,'Histoformer修复后',precision,*[num(row[k]) for k in ('psnr','ssim','lpips')]])
    add(table(['天气','方法','输出协议','PSNR↑','SSIM↑','LPIPS↓'],rows))
    add(f"修复后200张真实图的复原均值{num(meta['mean_restore_ms'],2)}ms/图、中位{num(meta['median_restore_ms'],2)}ms/图、峰值allocated {num(meta['peak_allocated_mib'],2)}MiB；冻结检测runtime均值{num(data['detector']['mean_detect_runtime_ms'],2)}ms/图。合成dev复原均值{num(quality['mean_restore_ms'],2)}ms/图。复原计时与检测计时各有边界，不相加伪称实测端到端；也不与不同精度/不同分辨率历史计时作公平加速结论。")
    add('dev质量作业在1024裁剪检查后显存占满，前20图复原约26秒/图；保存原始记录后用同一未改动脚本的quality-only独立进程续跑，恢复约3.4秒/图。已保存20图原样保留，权重/清单/LPIPS指纹和数值精度不变，裁剪检查未重复。dev平均耗时混合两次进程的显存状态，只作本次作业记录，不用于公平效率比较；续跑进程峰值也不是整个跨进程作业峰值。证据见quality_process_resume.json与质量两份日志。')
    add('## 5. 清理变量后的尺度对照')
    add('固定旧24张train帧，检查点v4/C0/C1均不更新。主D=直接输入；A=FP32 INTER_LINEAR缩放→无灰边小图复原→共同检测padding；B=原尺寸FP32复原→相同缩放→相同padding。主对照均关闭autocast/TF32，同样坐标反变换、640方形检测画布、原生联合GT损失、NMS .7和AP/runtime阈值。')
    add('仍保留的混杂：A只看无灰边360×640上下文，B看1920×1080并用1024/64四块；上下文/分块差异尚未被隔离。GT坐标一致只证明几何对齐，不能证明唯一变量已经隔离。此处为见过的train诊断子集AP，不是独立test或正式val成绩。')
    rows=[]
    for name,model in scale['models'].items():
        for path in ('D_fp32','A_fp32','B_fp32'):
            r=model['summaries'][path];rt=r['runtime'];a=r['ap']
            rows.append([name,path,num(a['map50']),num(a['map50_95']),num(rt['recall']),num(rt['fp_per_image']),num(rt['opr_to_D_fp32']),str(rt['gained_vs_D_fp32']),str(rt['lost_vs_D_fp32']),rt['new_fp_vs_D_fp32_count'],num(model['losses_mean'][path])])
    add(table(['模型','路径','AP50','AP50:95','Recall(.25)','FP/图(.25)','OPR对D','相对D新增GT','相对D丢失GT','新增FP','原生损失均值'],rows))
    add(table(['模型','路径',*NAMES.values()],[[name,path,*[num(model['summaries'][path]['ap']['per_class'][label]['ap50']) for label in NAMES.values()]] for name,model in scale['models'].items() for path in ('D_fp32','A_fp32','B_fp32')]))
    rows=[]
    for name,model in scale['models'].items():
        for path in ('D_u8','A_u8','B_u8','A_pad_historical_fp32','B_bf16_float','B_restoration_bf16_prequant_square'):
            r=model['summaries'][path]
            rows.append([name,path,num(r['ap']['map50']),num(r['runtime']['recall']),num(r['runtime']['fp_per_image']),str(r['runtime']['gained_vs_D_fp32']),str(r['runtime']['lost_vs_D_fp32'])])
        for path,r in model['actual_production_rect']['summaries'].items():
            rt=r['runtime'];rows.append([name,path,num(r['ap']['map50']),num(rt['recall']),num(rt['fp_per_image']),str(rt['gained_vs_D_original_u8_rect']),str(rt['lost_vs_D_original_u8_rect'])])
    add('独立扰动表：D/A/B_u8都在同一640画布位置round→uint8→float；A_pad是复原灰边的历史管线；B_bf16_float只改变复原精度；B_restoration_bf16_prequant_square额外使用全分辨率量化和uint8缩放；两个rect路径才走实际YOLO.predict自动矩形预处理。rect行的GT增减以其D_original_u8_rect为基准，其余以D_fp32为基准。')
    add(table(['模型','独立路径','AP50','Recall(.25)','FP/图(.25)','新增GT','丢失GT'],rows))
    rows=[]
    for name,model in scale['models'].items():
        for path in ('A_fp32','B_fp32'):
            for label in ('bus','train'):
                c=model['summaries'][path]['mean_native_loss_components_by_class'][label]
                rows.append([name,path,label,*[num(c[k]) for k in ('cls','box','dfl','total')]])
    add(table(['模型','路径','类别','原生分类均值','box均值','DFL均值','合计'],rows))
    rows=[]
    for name,model in scale['models'].items():
        summary=model['summaries']
        for space in ('original_size','detector_size'):
            buckets=summary['D_fp32']['gt_size_bins_all_classes'][space]
            for bucket,counts in buckets.items():
                rows.append([name,space,bucket,counts['gt'],*[str(summary[p]['gt_size_bins_all_classes'][space][bucket]['matched_ap001'])+'/'+str(summary[p]['gt_size_bins_all_classes'][space][bucket]['matched_runtime025']) for p in ('D_fp32','A_fp32','B_fp32')]])
    add('真实GT尺寸分桶使用原图和检测画布两套宽高/面积：bbox面积按COCO式32²/96²阈值分桶（非多边形标注area），最短边以8/16/32为界。下表所有类别共同统计，每格为AP .001命中数 / runtime .25命中数。每个GT还在JSON保留原始尺寸，避免用bicycle等类别名替代“小目标”。')
    add(table(['模型','尺寸空间','尺寸桶','GT数','D命中(.001/.25)','A命中(.001/.25)','B命中(.001/.25)'],rows))
    rows=[]
    for name,model in scale['models'].items():
        for image in model['per_image']:
            paths=image['paths'];by_path={p:{r['gt_id']:r for r in paths[p]['per_gt']} for p in ('D_fp32','A_fp32','B_fp32')}
            for gid,gt in by_path['D_fp32'].items():
                if gt['category'] not in (5,6):continue
                values=[by_path[p][gid] for p in ('D_fp32','A_fp32','B_fp32')]
                if len({v['post_nms_runtime025']['matched'] for v in values})<2:continue
                cells=[]
                for v in values:
                    raw=v['highest_target_class_score_with_iou_ge_0p5'];hit=v['post_nms_runtime025']
                    cells.append(('hit' if hit['matched'] else 'miss')+'; raw score '+num(raw['score'])+'; raw IoU '+num(raw['iou']))
                size=gt['original_size'];rows.append([name,image['image_id'],gid,gt['class_name'],num(size['width'],1)+'×'+num(size['height'],1),*cells])
    add('以下仅列主路径运行时匹配发生变化的bus/train目标，按模型+image_id+GT_id一行并排。raw score是在IoU≥.5候选中该类最高分，raw IoU为该候选IoU；它是NMS前诊断量，不能替代conf=.25检出判断。完整低阈值候选和NMS后匹配均保留。')
    add(table(['模型','image_id','GT_id','类别','原图W×H','D(.25状态;raw)','A(.25状态;raw)','B(.25状态;raw)'],rows))
    add(table(['image_id','评估GT_id/原始GT_id','640检测框','stride8框内中心','stride16','stride32'],[
        [r['image_id'],str(r['gt_id'])+'/'+str(r['source_gt_id']),str(r['detector_xyxy']),r['coverage']['8']['strict_inside'],r['coverage']['16']['strict_inside'],r['coverage']['32']['strict_inside']] for r in data['zero']['actual_zero_positive_bus']]))
    add('两个零正bus的中心必须逐框实算：其中一个虽最短边小于8像素，仍有一个stride8中心落入框内；另一个三层都没有严格框内中心。因此撤回“小于8像素必然无anchor”。有框内中心却无最终正样本，还可能涉及联合TAL打分、top-k和冲突消解；本轮没有凭中心计数就排除这些因素。')
    add('## 6. 下一轮唯一最值得进行的实验及理由')
    add(table(['旧结论','修订'],[
        ['bus/train梯度占15%–16%，监督已充分','撤回充分性结论；改名梯度范数占比（不可加和），只描述固定诊断帧'],
        ['已经排除监督覆盖问题','撤回；64张train不能代表整个训练期'],
        ['已证明主要是小目标被抹掉','降级为待验证；本轮提供真实尺寸桶观察，不能由类别名称直接推断因果'],
        ['一个检查点近似恒等，所以排除更强预训练路线','撤回；一个真实雪检查点及其迁移范围不足以外推成熟模型路线'],
        ['区间跨零，所以证明不更差','撤回；区间跨零不是等价性检验'],
        ['裁剪C2失败，所以尺度顺序不是主因','撤回；未启用相同GT梯度的裁剪组不能隔离处理路径，本轮也未运行C2']]))
    add('**唯一建议：'+recommendation['title']+'。**')
    add('依据：'+'；'.join(recommendation['basis'])+'。')
    add('实验设计：'+'；'.join(recommendation['design'])+'。')
    add('本轮只提出该项，不自动启动。若该项涉及处理路径训练，两组必须实际启用相同GT任务梯度，同初始化、数据/优化预算、损失系数和配置；真正的处理路径对齐与裁剪采样分别命名，不能拿C0零任务梯度组当尺度控制组。一次短实验不能排除整个方向。')
    add('证据边界：val已经用于历史模型选择，帧与序列相关；train诊断样本也来自历史训练资源。对这些帧重新划分不使现有检查点变成独立测试。单seed、单检查点和少量bus/train目标限制推断强度。没有依据时不提高总体GT权重、不立即换骨干。')
    add('原始证据入口：[监督](supervision_probe.json)、[尺度](scale_order_probe.json)、[修复检测](metrics.json)、[配对区间](bootstrap.json)、[双协议GT矩阵](bus_train_gt_matrix.csv)、[画质](quality_lpips_fusionfix.json)、[全幅裁剪/分块](tiling_consistency_fusionfix.json)、[历史有效性登记](LEGACY_RESULT_STATUS.md)、[预检与冻结资源](preflight.json)。')
    add('本报告读取的文件SHA256：\n\n'+table(['文件','SHA256'],data['evidence_sha256'].items()))
    return '\n'.join(lines).rstrip()+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recommendation-file',default='results/np2_revision/next_experiment.json')
    parser.add_argument('--out',default='results/np2_revision/REVISION_REPORT.md')
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    target=(ROOT/args.out).resolve()
    assert target.is_relative_to(OUT.resolve())
    if target.exists() and not args.validate_only:
        raise FileExistsError('Preserve existing report: '+str(target))
    data=evidence()
    recommendation=load(ROOT/args.recommendation_file)
    assert isinstance(recommendation['title'],str) and recommendation['title'].strip()
    assert recommendation['basis'] and recommendation['design'] and recommendation['not_executed'] is True
    assert not isinstance(recommendation.get('experiments'),list), 'Exactly one main recommendation'
    report=render(data,recommendation)
    assert sum(line.startswith('## ') for line in report.splitlines())==6
    if args.validate_only:
        print('All real inputs complete; six-section report renders successfully without writing')
        return
    with target.open('x',encoding='utf-8',newline='\n') as stream:stream.write(report)
    assert target.read_text(encoding='utf-8')==report
    print('Wrote and verified',target)


if __name__=='__main__':
    main()
