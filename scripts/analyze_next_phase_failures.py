"""依据官方 COCO 逐图匹配及保存的 pre-NMS 证据分析 bus/train。

此脚本只读取既有预测，不运行检测器或复原模型。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from saferestore.metrics.next_phase import CLASSES, coco_evaluate, normalize_ground_truth

REASONS = {
    'confidence_decrease': '正确类别但置信度降低并跨过运行时阈值',
    'wrong_class_candidate': '与 GT 重叠的其他类别候选',
    'localization_candidate': '正确类别高分框定位不足',
    'nms_suppression_evidence': '满足候选和抑制框条件的 NMS 证据',
    'original_input_miss': '原始输入已漏检',
    'new_high_ranking_false_positive': '新增且排在真实 TP 前面的 FP',
    'unresolved': '现有证据无法区分',
}


def iou(first, second):
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    wh = np.maximum(0.0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    inter = float(np.prod(wh))
    union = float(np.prod(np.maximum(a[2:]-a[:2], 0)) + np.prod(np.maximum(b[2:]-b[:2], 0)) - inter)
    return inter / union if union > 0 else 0.0


def xyxy(annotation):
    x, y, w, h = annotation['bbox']
    return [x, y, x+w, y+h]


def flat_predictions(row):
    return [dict(category_id=int(cid), box=p['box'], confidence=float(p['confidence']))
            for cid, predictions in row['predictions'].items() for p in predictions]


def box_evidence(prediction, target_box):
    if prediction is None:
        return None
    return dict(category_id=prediction['category_id'], category=CLASSES[prediction['category_id']],
                confidence=prediction['confidence'], box=prediction['box'], iou=iou(prediction['box'], target_box))


def match_index(evaluator):
    """仅从官方 COCOeval 读一对一匹配，不另造 TP/FP 匹配器。"""
    gt_matches, false_positives, true_positives = {}, [], []
    for result in evaluator.evalImgs:
        if result is None:
            continue
        for gid, did, ignored in zip(result['gtIds'], result['gtMatches'][0], result['gtIgnore']):
            if not ignored and did:
                dt = evaluator.cocoDt.anns[int(did)]
                gt_matches[int(gid)] = dict(category_id=dt['category_id'], confidence=float(dt['score']), box=xyxy(dt))
        for did, gid, ignored in zip(result['dtIds'], result['dtMatches'][0], result['dtIgnore'][0]):
            if ignored:
                continue
            dt = evaluator.cocoDt.anns[int(did)]
            item = dict(image_id=dt['image_id'], category_id=dt['category_id'],
                        confidence=float(dt['score']), box=xyxy(dt), matched_gt_id=int(gid) if gid else None)
            (true_positives if gid else false_positives).append(item)
    return gt_matches, false_positives, true_positives


def best_prediction(predictions, target_box, categories, minimum_score=0.001, minimum_iou=0.0):
    candidates = [p for p in predictions if p['category_id'] in categories
                  and p['confidence'] >= minimum_score and iou(p['box'], target_box) >= minimum_iou]
    return max(candidates, key=lambda p: (iou(p['box'], target_box), p['confidence'])) if candidates else None


def nms_evidence(row, target_box, category_id, threshold=.7):
    """保存候选-抑制框证据；排除被当作同一 survivor 的近似相同框。"""
    candidates = [p for p in row.get('pre_nms', []) if p['category_id'] == category_id
                  and p['confidence'] >= .25 and iou(p['box'], target_box) >= .5]
    survivors = [p for p in flat_predictions(row) if p['category_id'] == category_id]
    for candidate in sorted(candidates, key=lambda p: -p['confidence']):
        equivalent_survivor = any(abs(p['confidence']-candidate['confidence']) < 1e-5
                                  and iou(p['box'], candidate['box']) > .98 for p in survivors)
        if equivalent_survivor:
            continue
        for survivor in survivors:
            overlap = iou(candidate['box'], survivor['box'])
            if survivor['confidence'] > candidate['confidence'] and overlap > threshold:
                return dict(candidate=box_evidence(candidate, target_box),
                            suppressor=box_evidence(survivor, target_box), candidate_suppressor_iou=overlap,
                            qualification='Decoded-coordinate evidence under configured class-aware NMS; original indices are not saved.')
    return None


def method_gt_evidence(gt, ap_row, runtime_match, ap_match):
    target_box, cid, gid = xyxy(gt), gt['category_id'], gt['id']
    predictions = flat_predictions(ap_row)
    correct = best_prediction(predictions, target_box, {cid}, minimum_iou=.5)
    nearest = best_prediction(predictions, target_box, {cid})
    high_score = best_prediction(predictions, target_box, {cid}, minimum_score=.25)
    high_correct = best_prediction(predictions, target_box, {cid}, minimum_score=.25, minimum_iou=.5)
    other = best_prediction(predictions, target_box, set(CLASSES)-{cid}, minimum_score=.25, minimum_iou=.5)
    return dict(runtime_hit=gid in runtime_match, runtime_match=box_evidence(runtime_match.get(gid), target_box),
                ap50_match=box_evidence(ap_match.get(gid), target_box),
                best_same_class_iou=box_evidence(nearest, target_box),
                high_score_same_class=box_evidence(high_score, target_box),
                same_class_iou50_above_runtime=box_evidence(high_correct, target_box),
                best_same_class_iou50=box_evidence(correct, target_box),
                other_class_iou50=box_evidence(other, target_box),
                pre_nms_available='pre_nms' in ap_row)


def classify_gt(reference, candidate, row, gt, nms_threshold):
    reasons, evidence = [], {}
    if not reference['runtime_hit']:
        reasons.append('original_input_miss')
    if candidate['runtime_hit']:
        return reasons, evidence
    low = candidate['best_same_class_iou50']
    reference_correct = reference['runtime_match']
    if (reference_correct is not None and low is not None and .001 <= low['confidence'] < .25
            and candidate['same_class_iou50_above_runtime'] is None):
        reasons.append('confidence_decrease')
        evidence['confidence_decrease'] = dict(input=reference_correct, candidate=low)
    wrong = candidate['other_class_iou50']
    if wrong is not None:
        reasons.append('wrong_class_candidate')
        evidence['wrong_class_candidate'] = wrong
    loc = candidate['high_score_same_class']
    if loc is not None and loc['confidence'] >= .25 and .1 <= loc['iou'] < .5:
        reasons.append('localization_candidate')
        evidence['localization_candidate'] = loc
    nms = nms_evidence(row, xyxy(gt), gt['category_id'], nms_threshold)
    if nms:
        reasons.append('nms_suppression_evidence')
        evidence['nms_suppression_evidence'] = nms
    if not any(r != 'original_input_miss' for r in reasons):
        reasons.append('unresolved')
    return reasons, evidence


def load_rows(path, expected_ids):
    payload = json.loads(path.read_text(encoding='utf-8'))
    rows = payload['per_image']
    ids = [r['image_id'] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != expected_ids:
        raise ValueError(f'预测图像 ID 缺失/重复/多余：{path}')
    return {r['image_id']: r for r in rows}


def format_detection(item):
    if item is None:
        return '无匹配'
    return f"{item['category']} / {item['confidence']:.4f} / IoU {item['iou']:.4f}"


def render_report(payload):
    methods = payload['methods']
    lines = ['# bus/train 逐例失败分析', '', '本报告只读取已保存预测；不重新推理、不使用验证 GT 训练。'
             '“命中/FP”来自官方 COCOeval 在 IoU=0.5 的逐图一对一匹配，crowd 匹配被忽略。', '',
             '## 证据口径', '',
             '- 运行时以独立 conf=0.25 输出、maxDets=300 判定检出；AP/PR 以 conf=0.001、maxDets=100 统计。',
             '- GT 原始输入漏检与复原新增损失分开；分类允许多标签，分类计数不能相加作为唯一目标数。',
             '- 置信度下降须看到同类 IoU≥0.5 的低分框；空匹配本身不能证明置信度下降。',
             '- 类别错误与定位类别是有阈值证据的候选解释，不证明预测框与该物体具有唯一身份；重复框或邻近物体仍可能造成歧义。',
             '- NMS 证据须存在同类 pre-NMS 高分正确候选及更高分 post-NMS 抑制框，二者 IoU>配置阈值；排除近似同框 survivor。'
             '保存的是 decoded 坐标，缺少原始 NMS 索引；边界坐标舍入/letterbox 逆变换可有小偏差，因此称“可复核 NMS 证据”，不声称已追踪算子内部抑制索引。',
             '- pre-NMS 只保存最大类别候选，未保留每个 anchor 所有非最大类别 logits；不能由它排除全部类别竞争。',
             '- 新增高排序 FP 必须被官方匹配判为非忽略 FP、在 input 中无同类 IoU≥0.5 的对应 FP，并在该天气/类别排序中排在至少一个 TP 前面；并不证明删掉单个框后的 AP 反事实增量。', '',
             '## 全部非 crowd bus/train GT 的运行时检出', '',
             '| 范围 | 类别 | GT | '+' | '.join(methods)+' |',
             '|---|---|---:|'+'|'.join(['---:']*len(methods))+'|']
    for scope, classes in payload['hit_counts'].items():
        for cls, counts in classes.items():
            lines.append(f"| {scope} | {cls} | {counts['gt']} | " + ' | '.join(str(counts[m]) for m in methods)+' |')
    lines.extend(['', '## 解释标签数量（多标签，可重叠）', '', '| 方法 | 范围 | 类别 | 标签 | 数量 |', '|---|---|---|---|---:|'])
    for row in payload['reason_counts']:
        lines.append(f"| {row['method']} | {row['scope']} | {row['class']} | {REASONS[row['reason']]} | {row['count']} |")
    lines.extend(['', '## AP50 新协议实测', '', '同一新协议下比较，不能直接与历史 0.25 截断 AP 的差值混用。', '', '| 方法 | 范围 | bus AP50 | train AP50 | truck AP50 |', '|---|---|---:|---:|---:|'])
    for method, scopes in payload.get('ap_metrics', {}).items():
        for scope, values in scopes.items():
            vals = [f"{values.get(c):.6f}" if values.get(c) is not None else '未提供' for c in ('bus','train','truck')]
            lines.append('| '+method+' | '+scope+' | '+' | '.join(vals)+' |')
    for reason, examples in payload['examples'].items():
        lines.extend(['', '## '+REASONS[reason], '', f"最多展示 3 个真实样例，优先雨天 bus。可用样例 {len(examples)} 个。"])
        if not examples:
            lines.append('当前保存证据中没有满足该定义的样例，不能编造或从空匹配强行归因。')
        for case in examples:
            lines.extend(['', f"- `{case['file_name']}`；image_id={case['image_id']}，GT/候选类别={case['class']}，分析方法={case['method']}，GT id={case.get('gt_id', '无（FP）')}。", '', '| 方法 | 运行时同类匹配：类别 / 置信度 / IoU | 低阈值最佳同类重叠框 | 其他类别高分重叠框 |', '|---|---|---|---|'])
            for method, detail in case.get('comparison', {}).items():
                lines.append(f"| {method} | {format_detection(detail.get('runtime_match'))} | {format_detection(detail.get('best_same_class_iou'))} | {format_detection(detail.get('other_class_iou50'))} |")
            lines.extend(['', '证据：`'+json.dumps(case['evidence'], ensure_ascii=False)+'`'])
    lines.extend(['', '## 可复现性与限制', '',
                  '`failure_cases.json` 保存全部 bus/train GT 对照、所有分类标签和原始证据、官方匹配的 FP、输入文件 SHA256。'
                  '统计支持应读取主流程配对 bootstrap 报告；本脚本不将单例恢复解释为显著总体提升。', '',
                  '输入中的原始框坐标不用于训练；失配未解决时保留 unresolved。NMS 与定位边界案例应对照坐标和原图复核。', ''])
    return '\n'.join(lines)


def analyze(methods, predictions_root, annotation_path, metrics_path, config_path, output_dir):
    if 'input' not in methods or len(methods) != len(set(methods)):
        raise ValueError('--methods 必须包含 input，且不能重复')
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    images, gt = normalize_ground_truth(json.loads(annotation_path.read_text(encoding='utf-8')))
    ids = {i['id'] for i in images}
    image_lookup = {i['id']: i for i in images}
    inputs = [annotation_path, config_path]
    data = {}
    for method in methods:
        ap_path, runtime_path = predictions_root/method/'ap001.json', predictions_root/method/'runtime025.json'
        inputs.extend([ap_path, runtime_path])
        ap, runtime = load_rows(ap_path, ids), load_rows(runtime_path, ids)
        for image in images:
            if ap[image['id']]['file_name'] != image['file_name'] or runtime[image['id']]['file_name'] != image['file_name']:
                raise ValueError('预测图像路径与 GT 不一致')
        ap_eval = coco_evaluate(images, gt, list(ap.values()), .001, 100)
        rt_eval = coco_evaluate(images, gt, list(runtime.values()), .25, 300)
        ap_match, fp, tp = match_index(ap_eval)
        runtime_match, _, _ = match_index(rt_eval)
        data[method] = dict(ap=ap, runtime=runtime, ap_match=ap_match, runtime_match=runtime_match, fp=fp, tp=tp)
    focus_gt = [a for a in gt if a['category_id'] in (5,6) and not a.get('iscrowd', 0)]
    cases, counters, hit_counts = [], Counter(), {}
    examples = {reason: [] for reason in REASONS}
    scopes = {'all406':lambda w:True, 'rain_fog200':lambda w:w in ('rain','fog'), 'rain':lambda w:w=='rain'}
    for scope, condition in scopes.items():
        hit_counts[scope] = {}
        for cid in (5,6):
            selected = [a for a in focus_gt if condition(image_lookup[a['image_id']]['file_name'].split('/')[0]) and a['category_id']==cid]
            hit_counts[scope][CLASSES[cid]] = dict(gt=len(selected), **{m:sum(a['id'] in data[m]['runtime_match'] for a in selected) for m in methods})
    for target in focus_gt:
        image_id, gid, cid = target['image_id'], target['id'], target['category_id']
        image = image_lookup[image_id]
        weather = image['file_name'].split('/')[0]
        comparison = {m:method_gt_evidence(target, data[m]['ap'][image_id], data[m]['runtime_match'], data[m]['ap_match']) for m in methods}
        for method in methods:
            if method == 'input':
                continue
            reasons, evidence = classify_gt(comparison['input'], comparison[method], data[method]['ap'][image_id], target, float(config['nms_iou']))
            case = dict(gt_id=gid, source_gt_id=target.get('source_annotation_id',gid), image_id=image_id, file_name=image['file_name'], weather=weather, category_id=cid,
                        **{'class':CLASSES[cid]}, box=xyxy(target), method=method, reasons=reasons,
                        lost_vs_input=comparison['input']['runtime_hit'] and not comparison[method]['runtime_hit'],
                        gained_vs_input=not comparison['input']['runtime_hit'] and comparison[method]['runtime_hit'],
                        comparison=comparison, evidence=evidence)
            cases.append(case)
            for reason in reasons:
                for scope, condition in scopes.items():
                    if condition(weather): counters[(method,scope,CLASSES[cid],reason)] += 1
                examples[reason].append(dict(case, evidence=evidence.get(reason, {'input_hit':comparison['input']['runtime_hit'],'candidate_hit':comparison[method]['runtime_hit']})))
    fp_records = {}
    for method in methods:
        local = []
        for fp in data[method]['fp']:
            cid, image_id = fp['category_id'], fp['image_id']
            if cid not in (5,6):
                continue
            image = image_lookup[image_id]
            weather = image['file_name'].split('/')[0]
            equivalent_input_fp = any(p['image_id']==image_id and p['category_id']==cid and iou(p['box'], fp['box'])>=.5 for p in data['input']['fp'])
            ranked_before_tp = any(p['category_id']==cid and image_lookup[p['image_id']]['file_name'].split('/')[0]==weather and p['confidence']<fp['confidence'] for p in data[method]['tp'])
            high_new = method!='input' and not equivalent_input_fp and fp['confidence']>=.25 and ranked_before_tp
            nearby = [a for a in focus_gt if a['image_id']==image_id and a['category_id']==cid]
            nearest = max(nearby, key=lambda a:iou(xyxy(a),fp['box'])) if nearby else None
            target_box = xyxy(nearest) if nearest else fp['box']
            record = dict(fp, file_name=image['file_name'], weather=weather, **{'class':CLASSES[cid]},
                          nearest_gt_id=nearest['id'] if nearest else None,
                          nearest_gt_iou=iou(target_box,fp['box']) if nearest else None,
                          equivalent_input_fp=equivalent_input_fp, ranked_before_tp=ranked_before_tp, new_high_ranking_fp=high_new)
            local.append(record)
            if high_new:
                reason = 'new_high_ranking_false_positive'
                for scope, condition in scopes.items():
                    if condition(weather): counters[(method,scope,CLASSES[cid],reason)] += 1
                comp = {m:method_gt_evidence(nearest, data[m]['ap'][image_id], data[m]['runtime_match'], data[m]['ap_match']) for m in methods} if nearest else {}
                examples[reason].append(dict(image_id=image_id, file_name=image['file_name'], weather=weather, **{'class':CLASSES[cid]},method=method,gt_id=record['nearest_gt_id'], comparison=comp,
                                             evidence=record))
        fp_records[method] = local
    for reason in examples:
        examples[reason].sort(key=lambda c:(not(c['weather']=='rain' and c['class']=='bus'), c['weather']!='rain', c['class']!='bus', c['image_id'], c['method']))
        examples[reason] = examples[reason][:3]
    ap_metrics = {}
    if metrics_path.exists():
        inputs.append(metrics_path)
        metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
        for method in methods:
            if method in metrics.get('methods', {}):
                ap_metrics[method] = {scope:{cls:values['standard']['per_class'][cls]['ap50'] for cls in ('bus','train','truck')}
                                      for scope,values in metrics['methods'][method]['scopes'].items()}
    payload = dict(methods=methods, protocol=config, classification_policy='Multilabel, no unique causal root assumed',
                   gt_scope='Noncrowd bus/train GT; all406, rain_fog200 and rain summaries',
                   hit_counts=hit_counts, reason_counts=[dict(method=m,scope=s,**{'class':c},reason=r,count=n) for (m,s,c,r),n in sorted(counters.items())],
                   examples=examples, gt_cases=cases, ap50_official_false_positives=fp_records, ap_metrics=ap_metrics,
                   input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir/'failure_cases.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    (output_dir/'FAILURE_ANALYSIS.md').write_text(render_report(payload),encoding='utf-8')
    print(json.dumps(dict(gt_objects=len(focus_gt), method_gt_cases=len(cases), methods=methods,
                          example_counts={r:len(v) for r,v in examples.items()}, output=str(output_dir)),ensure_ascii=False))
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods',nargs='+',default=['input','v4','guard'])
    parser.add_argument('--predictions-root',type=Path,default=ROOT/'results/next_phase/predictions')
    parser.add_argument('--annotations',type=Path,default=ROOT/'data/raw/ACDC_detection/gt_detection/instancesonly_val_gt_detection.json')
    parser.add_argument('--metrics',type=Path,default=ROOT/'results/next_phase/metrics.json')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/next_phase_evaluation.yaml')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results/next_phase')
    args = parser.parse_args()
    analyze(args.methods,args.predictions_root,args.annotations,args.metrics,args.config,args.output_dir)


if __name__=='__main__':
    main()
