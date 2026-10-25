"""从实际 JSON/CSV 生成下一阶段报告；缺失项明确写未完成。"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/next_phase'
sys.path.insert(0,str(ROOT/'src'))
from saferestore.statistics import paired_bootstrap_difference


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(name,lines):
    (OUT/name).write_text('\n'.join(lines)+'\n',encoding='utf-8')


def f(value):
    return '—' if value is None else f'{value:.6f}'


def metric_tables(data,names,scopes=('rain_fog200','all406','fog','rain','night','snow')):
    lines=[]
    for scope in scopes:
        lines += [f'## {scope}', '', '| 方法 | AP50(.001) | AP50:95(.001) | 去train六类AP50 | Recall(.25) | FP/图(.25) | OPR(.25) |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for name in names:
            row=data['methods'][name]['scopes'][scope];s,r=row['standard'],row['runtime']
            lines += [f"| {name} | {f(s['map50'])} | {f(s['map50_95'])} | {f(s['map50_without_train'])} | {f(r['recall'])} | {f(r['fp_per_image'])} | {f(r['opr'])} |"]
        lines += ['', '| 类别 | 非crowd GT | '+' | '.join(n+' AP50' for n in names)+' |', '|---|---:|'+'---:|'*len(names)]
        classes=data['methods']['input']['scopes'][scope]['standard']['per_class']
        for cls,value in classes.items():
            lines += [f"| {cls} | {value['gt']} | "+' | '.join(f(data['methods'][n]['scopes'][scope]['standard']['per_class'][cls]['ap50']) for n in names)+' |']
        lines += ['']
    return lines


def additional_results(data,names):
    lines=[]
    quality=read(OUT/'image_quality.json') if (OUT/'image_quality.json').exists() else None
    benchmark=read(OUT/'benchmark.json') if (OUT/'benchmark.json').exists() else None
    if quality:
        lines += ['## 有效像素对齐数据的画质', '',quality['evidence_scope'], '',
            '下面为每图指标再平均；RGB范围0..1，PSNR全像素，SSIM高斯11/sigma1.5的valid窗口。LPIPS因本地包及预训练权重缺失而BLOCKED。没有把ACDC跨时刻weather/reference直接计算成配对画质。', '',
            '| 方法 | dev200 PSNR(dB) | dev200 SSIM | fog PSNR / SSIM | rain PSNR / SSIM |', '|---|---:|---:|---|---|']
        for name in names:
            if name not in quality['methods']:continue
            s=quality['methods'][name]['summary']
            lines += [f"| {name} | {s['all']['psnr']:.6f} | {s['all']['ssim']:.6f} | {s['fog']['psnr']:.4f} / {s['fog']['ssim']:.6f} | {s['rain']['psnr']:.4f} / {s['rain']['ssim']:.6f} |"]
        q_ci={}
        for candidate,reference in (('C1','C0'),('C1','v4'),('C0','v4')):
            if candidate not in quality['methods'] or reference not in quality['methods']:continue
            a={r['scene_id']:r for r in quality['methods'][candidate]['per_image']}
            b={r['scene_id']:r for r in quality['methods'][reference]['per_image']}
            if set(a)!=set(b) or any(a[k]['weather']!=b[k]['weather'] for k in a):raise ValueError('画质记录不严格配对')
            q_ci[candidate+':'+reference]={}
            for scope in ('all','fog','rain'):
                ids=sorted(k for k in a if scope=='all' or a[k]['weather']==scope)
                q_ci[candidate+':'+reference][scope]={key:paired_bootstrap_difference([a[k][key] for k in ids],[b[k][key] for k in ids],resamples=1000,seed=20261025) for key in ('psnr','ssim')}
        if q_ci:
            (OUT/'quality_bootstrap.json').write_text(json.dumps(dict(unit='paired synthetic image, percentile95, shared project statistics helper',pairs=q_ci),ensure_ascii=False,indent=2),encoding='utf-8')
            lines += ['', '| 画质差值(dev200) | 均值差 | 配对bootstrap95%区间 |', '|---|---:|---|']
            for pair,scopes in q_ci.items():
                for key,v in scopes['all'].items():lines += [f"| {pair} {key} | {v['mean']:.6f} | [{v['ci_lower']:.6f}, {v['ci_upper']:.6f}] |"]
            lines += ['', '画质区间基于对齐合成dev图像的配对重采样；不替代真实天气主观质量、LPIPS、多seed或场景独立评测。', '']
    if benchmark:
        lines += ['## 完整端到端实测', '',benchmark['selection'], '',benchmark['included'], '',
            '| 方法 | 复原参数 / 检测参数 | 平均ms | 中位ms | P95 ms | 峰值allocated MiB |', '|---|---:|---:|---:|---:|---:|']
        for name in names:
            if name not in benchmark['methods']:continue
            b=benchmark['methods'][name]
            lines += [f"| {name} | {b['restoration_parameters']} / {b['detector_parameters']} | {b['mean_ms']:.2f} | {b['median_ms']:.2f} | {b['p95_ms']:.2f} | {b['peak_allocated_mib']:.1f} |"]
        lines += ['', '模型载入、权重哈希、写图和GUI不计时。全部本地NAFNet候选结构相同，延迟差异不应解释成骨干加速；CPU bootstrap已结束后才计时。最初全406评估metadata另有不含读图的计时，两者边界不同，不能混用。', '']
    for path in (OUT/'bootstrap_p0.json',OUT/'bootstrap_gt.json'):
        if not path.exists():continue
        ci=read(path)
        lines += [f"## 检测配对区间：{path.name}", '',ci['unit'], '',ci['limitation'], '',
            '| 集合 | 差值(candidate:reference) | 指标 | 点差 | 95%区间 | 有效重采样 |', '|---|---|---|---:|---|---:|']
        for scope,pairs in ci['scopes'].items():
            for pair,values in pairs.items():
                for key in ('map50','map50_95','bus','train','truck','recall','fp_per_image','opr'):
                    v=values[key];bounds=v['ci95']
                    lines += [f"| {scope} | {pair} | {key} | {f(v['delta'])} | {('['+f(bounds[0])+', '+f(bounds[1])+']') if bounds else '—'} | {v['valid_resamples']} |"]
        lines += ['']
    return lines


def report_p0(data):
    names=[n for n in ('input','v4','guard') if n in data['methods']]
    lines=['# P0 正确性核查与统一协议重评', '',
        '核查日期：2026-10-25。远端 `tiance002/image/main` 与本地公开副本起点均为 `396af4a50a2512b7dabb00a69b3ccfded403dea9`。实际科研工作区无Git历史；本轮以现存源码、运行配置、权重与原始标注为准，最终提交到已有本地公开副本，不自动push。', '',
        '## 数据颜色与历史影响', '',
        '历史 scene 800/800 唯一目标 JPEG 经实物逐字节重编码确认为R/B交换；首个night退化输入亦精确复演。weather 800/800清晰目标正确；当前生成器BGR标定用于RGB的契约错误已修，但六个历史weather退化图的seed42复演不能吻合，历史配方仍未完整溯源。不能将颜色问题认定为bus掉分根因。', '',
        '已新建 weather/train2400对、weather/dev200对、scene/train800对，共5200个唯一JPEG；全量解码、960×540尺寸和样本顺序核验通过。旧数据、CSV、标定、历史权重保留。v4初始化仍属于历史数据分布。细节、样例、版本和哈希见 `COLOR_AUDIT.md` / `color_audit.json`。', '',
        '## 新协议与标注处理', '',
        '- 检测器为本地YOLOv8n，SHA256 `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`；imgsz640，class-aware NMS IoU0.7，全局maxDet300，FP32检测。七类COCO零基ID为0/1/2/3/5/6/7；修复detector.yaml漏列train的问题。',
        '- AP独立预测conf0.001；复用COCO官方bbox匹配、crowd忽略、每图每类maxDets100、IoU.50:.05:.95和101点PR。官方源码/许可证在 `src/saferestore/metrics/coco_official/`；仅将移除的np.float别名换为同义float64，bbox IoU适配支持crowd交集/检测面积。没有安装pycocotools。',
        '- 运行时另跑conf0.25，读取同一官方匹配的IoU0.5计数、maxDet300；OPR使用原图已检出非crowd GT的一对一保持率。不能把低阈值Recall冒充运行时Recall。',
        '- GT来自原始combined val标注，406图/2794条原始框；七类非crowd共2682框。crowd按官方规则允许重复匹配并忽略；rider25不映射person，未标注的其余COCO类别不评估。原始文件无独立ignore字段，若未来出现未定义ignore会拒绝而非静默丢弃。',
        '- 这是项目规定的七类映射子集，不等于ACDC原生八类排行榜分数。rider不合并person，也不跨类别建立评估ignore区，因此person覆盖rider可能被算FP；训练B流则保守忽略rider区域负监督，二者规则已区分。逐天气没有GT的类别AP记为—，COCO均值只统计有GT类；200/406两个主汇总七类均有GT。',
        '- 复原EMA、完整原图1024tile/64overlap、bf16推理、RGB→BGR后交YOLO；每图每类预测及focus bus/train pre-NMS候选保存在 `predictions/<方法>/`，PR数组在metrics.json。pre-NMS反变换存在约2像素舍入误差，NMS诊断保留证据限制。',
        '- 固定Python3.11.15、torch2.6.0+cu124、Ultralytics8.4.104、NumPy2.4.4；每份metadata.json保存权重/标注哈希、版本、延迟和显存。',
        '- 本轮406与200均为ACDC val模型选择集，不是独立test。train/val帧文件名无交集，但15个weather/sequence键重合；不作场景独立、跨域泛化或完整D1宣称。', '',
        '另将1600张train原图的预注册SHA256与406张val原图逐一比对，文件内容完全相同的跨划分重复为0；标注ID重复、非正框尺寸、引用缺失图像均为0。证据见annotation_integrity.json。这不排除同序列相邻帧相关性。', '',
        '## AP协议差异', '',
        '下表用于展示协议差异，不属于模型改进。历史列引用旧推理产物和自定义评估，只作参考，不能将其与本轮差值全部归因于评估器。后两列基于本轮同一源图与推理链路、COCO官方匹配，能直接比较阈值截断的影响。', '',
        '| 方法/雨雾200 | 历史自定义AP50@.25 | 本轮COCO AP50@.25 | 本轮标准AP50@.001 |', '|---|---:|---:|---:|']
    historical=dict(input=.204221,v4=.203157,guard=.200764)
    for n in names:
        row=data['methods'][n]['scopes']['rain_fog200']
        lines += [f"| {n} | {historical[n]:.6f} | {row['legacy_threshold_same_coco']['map50']:.6f} | {row['standard']['map50']:.6f} |"]
    corrected_input=data['methods']['input']['scopes']['rain_fog200']['standard']['map50']
    lines += ['', f'原图标准AP50由旧0.2042变为{corrected_input:.6f}，主要说明旧高阈值截断了PR曲线，不能据此宣称模型提升。新协议下原图motorcycle AP50=0.032247，旧“motorcycle 在基线下的 AP 为 0.0000，属检测器能力上限而非复原引入的退化。”只反映旧阈值数字，其“能力上限”归因不再成立；对应验收文档已增加勘误。', '',
        'ACDC原始annotation ID从0开始，而官方COCO dtMatches以0作为未匹配哨兵。本轮适配统一将annotation ID加1并保存source_annotation_id，避免首个GT被误记为FP；图像ID仍可为0。零ID、重复ID等5项评估专项测试通过。适配前的临时汇总已标记INVALIDATED并移入audit_history，全部汇总及bootstrap从原预测重新计算；训练配方、预测框和GT内容未变。', '']
    lines += ['原图视觉复核还确认一个重要边界：雨天`GOPR0402_frame_000120`左侧是被画面边缘截断的铰接公交，input/v4/guard都将其识别为truck（置信度约.622/.539/.593，IoU约.970/.958/.969）。这是原图已有的类别错误，不能归因于复原新增损伤；见MANUAL_CASE_REVIEW.md。', '']
    lines += metric_tables(data,names)
    lines += ['## 正确性检查与后续', '',
        '直接相关测试共27项：10颜色契约、8真实GT冻结损失、5COCO阈值/crowd/零标注ID/缺图及重复预测、4NAFNet非整除尺寸/残差梯度测试。另用重复图像构造独立重评核对bootstrap复用匹配结果，AP50及AP50:95一致。未扩展到无关UI测试、未安装依赖。', '',
        '后续画质评估还发现并修复既有NAFNet残差尺寸错误：960×540补齐后x为544高，输出裁为540高，原代码无法相加。最小修复只将残差裁回原高宽，不改层结构、参数或权重。实际C0 EMA在320/640 FP32及1024 bf16下新旧前向逐位一致，差值0；全部406图均1920×1080，已完成检测只使用1024图块，故原预测有效。见padding_equivalence.json。', '',
        'P0失败类型与真实样例见 `FAILURE_ANALYSIS.md` / `failure_cases.json`。P0完成后按已预注册配置自动进入C0/C1，不依据validation结果调整检测权重或延长预算。']
    save('P0_REPORT.md',lines)


def training_report(data):
    run=ROOT/'runs/gt_next_phase_20261025'
    plan=read(run/'comparison_plan.json')
    calibration=read(run/'calibration.json') if (run/'calibration.json').exists() else None
    lines=['# 真实GT检测监督与公平续训', '',f"计划状态：`{plan['status']}`。", '',
        'NAFNet骨干保持现有结构。借鉴UniRestore冻结检测器对复原图施加真实框任务监督的思想；原生YOLO损失、TaskAlignedAssigner、box回归及DFL来自已安装Ultralytics8.4.104。没有迁移扩散框架，没有修改生产冻结检测器。', '',
        'A流：修复RGBv2配对，crop320/batch4、extra_ratio0.6，原像素/SSIM/梯度/颜色/残差/检测特征损失全部保留；特征中心256和尺度2/1/.75，置信度guard关闭。B流：1600张官方真实天气train帧，完整视野letterbox640/microbatch1，真实同帧9607框，七类映射，crowd/rider528框用于保守负样本忽略；未标注73类不计算分类负监督。', '',
        'YOLO权重requires_grad=False且eval固定BN；学生forward保留图像梯度，A/B顺序提取对NAFNet参数的梯度后合成。直接复用原生cls/box/DFL，不改任务分配。忽略框训练策略与COCO评估crowd规则不同，具体见GT_READINESS.md。', '',
        '两组均恢复v4@47999 model/EMA/AdamW moments，相同seed42、A/B样本顺序与内容、FP32、AdamW lr2e-5常数/wd1e-4/betas.9/.99、EMA.999，预算各4000步；保存+2000/+4000。C0也跑B流及原生损失梯度监测，唯一核心差异为C1将新增梯度加入更新。', '',
        '最初8个固定train批次、不更新模型/优化器、按0.2×A梯度范数/B梯度范数的中位数定一次权重；500步线性启用。该20%是工程初始值，未宣称最优；不看val调参。', '',
        'v4历史初始化经历了已证实scene颜色交换及未完整溯源的weather配方。本轮C0/C1均用同一新分布；相对v4的收益包含修复数据及普通续训，新增GT机制只由C1−C0比较。', '',
        f"预注册固定{len(plan['inputs_sha256'])}个输入文件、{len(plan['source_sha256'])}个源码文件哈希；保存完整逐步tensor+GT+样本ID流哈希。", '']
    if calibration:
        lines += [f"实际固定检测权重：**{calibration['weight']:.9g}**；{calibration['batches']}个校准批次，模型/检测器状态不变。", '']
    for arm in ('C0','C1'):
        directory=ROOT/f'runs/gt_next_phase_20261025_{arm}'
        evidence_path=directory/'training_evidence.json'
        if not evidence_path.exists():
            lines += [f'## {arm}', '', '尚无完整结束证据，不填造训练步数或结果。', ''];continue
        evidence=read(evidence_path)
        with (directory/'train_log.csv').open(encoding='utf-8') as handle: rows=list(csv.DictReader(handle))
        lines += [f'## {arm}', '',f"实训{evidence['steps']}步；用时{evidence['elapsed_seconds']/60:.2f}分钟；峰值allocated显存{evidence['peak_gpu_allocated_bytes']/2**20:.1f}MiB。",f"完整批次SHA256：`{evidence['stream_sha256']}`。检测器/BN前后状态哈希一致：{evidence['detector_unchanged']}。", '',
            '| 逐步量 | 前100步均值 | 后100步均值 |', '|---|---:|---:|']
        for key in ('restoration_total','pixel','ssim','grad','color','delta','detector_feature','task_total','box_loss','cls_loss','dfl_loss','grad_restoration_l2','grad_task_l2','grad_task_weighted_applied_l2','grad_total_preclip_l2','ignored_anchor_fraction'):
            lines += [f"| {key} | {np.mean([float(r[key]) for r in rows[:100]]):.6g} | {np.mean([float(r[key]) for r in rows[-100:]]):.6g} |"]
        ratios=np.array([float(r['grad_task_weighted_applied_l2'])/max(float(r['grad_restoration_l2']),1e-12) for r in rows])
        clipped=np.mean([float(r['grad_total_preclip_l2'])>1 for r in rows])
        lines += ['',f'训练全程实际加权B/A梯度范数比中位数={np.median(ratios):.6g}、P95={np.quantile(ratios,.95):.6g}；触发norm1裁剪的步骤比例={clipped:.2%}。初始校准20%不保证每张图或整个训练期都维持20%，未使用动态权重。']
        lines += ['', '`grad_total_postclip_l2`为min(preclip,1)的裁剪上界，实际PyTorch稳定项使范数略低，不声称独立测量了精确裁剪后范数。所有主损失和A/B梯度范数逐步保存，缺失梯度、NaN/Inf、冻结状态变化会终止。', '']
    if plan['status']=='completed':
        lines += ['## 公平性结束核验', '',f"配置一致={plan.get('config_match')}；完整流一致={plan.get('batch_match')}；model/EMA初始化一致={plan.get('initialization_match')}；检测器冻结={plan.get('detector_unchanged')}。训练后资源/源码再次核验。两组同优化步数和数据预算，新增反传的实耗时间另列，不声称墙钟时间相同。",'']
        lines += ['结束核验之后，为完成960×540画质评估修复了NAFNet残差padding裁剪。训练时精确6份源码已按原哈希归档于runs/gt_next_phase_20261025/training_sources/；320/640训练及1024检测图块的新旧前向逐位一致（max差0），所以没有重训或重跑已保存检测。当前推理源码SHA与历史训练源码SHA不同，必须按上述版本说明理解。', '']
    if (OUT/'gt_coverage.json').exists():
        coverage=read(OUT/'gt_coverage.json')['arms']['C1']
        lines += ['## 真实GT输入覆盖', '',f"两组B流均4000次、1600个唯一真实train图像；bus框输入{coverage['gt_box_input_draws']['bus']}次，来自{coverage['unique_images_by_class']['bus']}个唯一图像；train框输入{coverage['gt_box_input_draws']['train']}次，来自{coverage['unique_images_by_class']['train']}个唯一图像。", '这些是真实标签输入次数，不是独立场景数，也不等价于原生assigner实际正anchor数；后者未逐类记录。完整统计见gt_coverage.json。', '']
    names=[n for n in ('input','v4','C0','C1','guard') if n in data['methods']]
    lines += metric_tables(data,names)
    extra=additional_results(data,names)
    lines += extra
    save('TRAINING_REPORT.md',lines)
    if 'C0' in data['methods'] and 'C1' in data['methods']:
        baseline_path=OUT/'BASELINE_COMPARISON.md'
        marker='\n<!-- LOCAL_MEASURED_RESULTS -->\n'
        original=baseline_path.read_text(encoding='utf-8').split(marker)[0]
        local=['# 本轮本地模型实测对照', '', '外部两基线仍为BLOCKED，不填写或推断其数值。以下统一新协议本地数值可追溯metrics.json；完整逐天气/训练/区间见TRAINING_REPORT.md。', '']
        local+=metric_tables(data,names,scopes=('rain_fog200','all406'))
        local+=extra
        baseline_path.write_text(original+marker+'\n'.join(local)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training',action='store_true')
    args=parser.parse_args()
    data=read(OUT/'metrics.json')
    report_p0(data)
    if args.training: training_report(data)


if __name__=='__main__':main()
