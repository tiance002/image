"""统一 COCO bbox AP 与运行时计数；保留官方 crowd/maxDets/PR 逻辑。"""
from __future__ import annotations

import contextlib
import copy
import io
from collections import defaultdict

import numpy as np

from .coco_official.cocoeval import COCOeval

CLASSES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 6: 'train', 7: 'truck'}
ACDC_TO_COCO = {24: 0, 33: 1, 26: 2, 32: 3, 28: 5, 31: 6, 27: 7}


class BboxCOCO:
    """仅提供官方 evaluator 所需索引接口；不实现匹配或 AP。"""
    def __init__(self, images, annotations):
        self.images = images
        self.anns = {a['id']: copy.deepcopy(a) for a in annotations}
        self.imgToAnns = defaultdict(list)
        for annotation in self.anns.values():
            self.imgToAnns[annotation['image_id']].append(annotation)

    def getImgIds(self):
        return [i['id'] for i in self.images]

    def getCatIds(self):
        return list(CLASSES)

    def getAnnIds(self, imgIds=(), catIds=()):
        images, categories = set(imgIds), set(catIds)
        return [a['id'] for a in self.anns.values()
                if (not images or a['image_id'] in images) and (not categories or a['category_id'] in categories)]

    def loadAnns(self, ids):
        return [self.anns[i] for i in ids]


def normalize_ground_truth(payload):
    images = sorted(payload['images'], key=lambda x: x['id'])
    names = [i['file_name'] for i in images]
    if len(names) != len(set(names)) or len({i['id'] for i in images}) != len(images):
        raise ValueError('重复验证图像ID/路径')
    if any('/val/' not in name.replace('\\', '/') for name in names):
        raise ValueError('评估只接受明确的 val 天气图像')
    annotations = []
    source_ids = [int(a['id']) for a in payload['annotations']]
    if len(source_ids) != len(set(source_ids)) or any(i < 0 for i in source_ids):
        raise ValueError('原始标注ID必须唯一且非负')
    for source in payload['annotations']:
        if source['category_id'] not in ACDC_TO_COCO:
            continue
        annotation = copy.deepcopy(source)
        annotation.pop('segmentation', None)
        # ACDC原始ann id从0开始；COCOeval dtMatches中的0是未匹配哨兵。
        # 图像ID可为0，annotation ID必须正值。保留来源ID便于追溯。
        annotation['source_annotation_id'] = int(source['id'])
        annotation['id'] = int(source['id']) + 1
        annotation['category_id'] = ACDC_TO_COCO[source['category_id']]
        # 当前官方标注只有 iscrowd，无显式 ignore；不静默支持另一套 ignore 定义。
        if annotation.get('ignore', 0) and not annotation.get('iscrowd', 0):
            raise ValueError('发现额外 ignore 标注，需核验其官方语义后评估')
        annotations.append(annotation)
    return images, annotations


def detections_from_rows(rows, threshold):
    annotations = []
    for row in rows:
        for class_id, predictions in row['predictions'].items():
            if int(class_id) not in CLASSES:
                continue
            for prediction in predictions:
                if prediction['confidence'] < threshold:
                    continue
                x1, y1, x2, y2 = prediction['box']
                annotations.append(dict(id=len(annotations) + 1, image_id=row['image_id'],
                    category_id=int(class_id), bbox=[x1, y1, x2-x1, y2-y1],
                    score=prediction['confidence'], area=max(0, x2-x1)*max(0, y2-y1), iscrowd=0))
    return annotations


def coco_evaluate(images, ground_truth, rows, threshold=0.001, max_dets=100):
    if len(images) != len(rows) or {i['id'] for i in images} != {r['image_id'] for r in rows}:
        raise ValueError('评估图像与预测必须严格一一对应')
    if any(a['id'] <= 0 for a in ground_truth) or len({a['id'] for a in ground_truth}) != len(ground_truth):
        raise ValueError('COCO匹配要求正值且唯一的annotation ID；请先normalize_ground_truth')
    evaluator = COCOeval(BboxCOCO(images, ground_truth), BboxCOCO(images, detections_from_rows(rows, threshold)), 'bbox')
    evaluator.params.areaRng = [[0, 1e10]]
    evaluator.params.areaRngLbl = ['all']
    evaluator.params.maxDets = [max_dets]
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.evaluate()
        evaluator.accumulate()
    return evaluator


def precision_summary(evaluator, include_pr=False):
    precision = evaluator.eval['precision'][:, :, :, 0, 0]
    def mean(values):
        valid = values[values >= 0]
        return float(valid.mean()) if valid.size else None
    per_class = {}
    for k, cid in enumerate(evaluator.params.catIds):
        count = sum(not a.get('iscrowd', 0) for a in evaluator.cocoGt.anns.values() if a['category_id'] == cid and a['image_id'] in set(evaluator.params.imgIds))
        per_class[CLASSES[cid]] = dict(gt=count, ap50=mean(precision[0, :, k]), ap50_95=mean(precision[:, :, k]))
        if include_pr:
            per_class[CLASSES[cid]]['pr50'] = precision[0, :, k].tolist()
            per_class[CLASSES[cid]]['score_at_recall50'] = evaluator.eval['scores'][0, :, k, 0, 0].tolist()
    six = [v['ap50'] for k, v in per_class.items() if k != 'train' and v['ap50'] is not None]
    return dict(map50=mean(precision[0]), map50_95=mean(precision),
                map50_without_train=float(np.mean(six)) if six else None, per_class=per_class,
                recall_grid=evaluator.params.recThrs.tolist() if include_pr else None)


def runtime_counts(evaluator):
    """从官方 IoU=.5 一对一匹配读计数，忽略 crowd 匹配。"""
    hits = set()
    tp = fp = gt = 0
    class_counts = {CLASSES[c]: dict(tp=0, fp=0, gt=0) for c in evaluator.params.catIds}
    for result in evaluator.evalImgs:
        if result is None:
            continue
        matches = result['dtMatches'][0]
        ignored = result['dtIgnore'][0]
        valid = ~ignored
        local_tp = int(np.count_nonzero((matches > 0) & valid))
        local_fp = int(np.count_nonzero((matches == 0) & valid))
        local_gt = int(np.count_nonzero(result['gtIgnore'] == 0))
        hits.update(int(i) for i in matches[(matches > 0) & valid])
        tp += local_tp; fp += local_fp; gt += local_gt
        bucket = class_counts[CLASSES[result['category_id']]]
        bucket['tp'] += local_tp; bucket['fp'] += local_fp; bucket['gt'] += local_gt
    return dict(recall=tp/max(gt, 1), precision=tp/max(tp+fp, 1), fp_per_image=fp/max(len(evaluator.params.imgIds),1),
                tp=tp, fp=fp, gt=gt, hits=hits, per_class=class_counts)


def resample_summary(evaluator, indices):
    """复用官方逐图匹配结果，按配对图像有放回重采样；不重跑检测。"""
    sampled = copy.copy(evaluator)
    sampled.params = copy.deepcopy(evaluator.params)
    sampled._paramsEval = copy.deepcopy(evaluator._paramsEval)
    ids = list(range(len(indices)))
    sampled.params.imgIds = ids
    sampled._paramsEval.imgIds = ids
    n = len(evaluator.params.imgIds)
    sampled.evalImgs = [evaluator.evalImgs[k*n + int(i)] for k in range(len(CLASSES)) for i in indices]
    with contextlib.redirect_stdout(io.StringIO()):
        sampled.accumulate()
    p = sampled.eval['precision'][:, :, :, 0, 0]
    valid = p >= 0
    return dict(map50=float(p[0][valid[0]].mean()), map50_95=float(p[valid].mean()),
        **{CLASSES[cid]: (float(p[0,:,k][p[0,:,k]>=0].mean()) if np.any(p[0,:,k]>=0) else np.nan)
           for k,cid in enumerate(evaluator.params.catIds)})
