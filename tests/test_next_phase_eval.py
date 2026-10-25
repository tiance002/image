import numpy as np
import pytest
from saferestore.metrics.coco_official.mask import iou
from saferestore.metrics.next_phase import coco_evaluate, precision_summary, runtime_counts, resample_summary, normalize_ground_truth


def test_coco_crowd_ioa_and_perfect_detection():
    assert iou([[10,10,5,5]], [[0,0,100,100]], [1])[0,0] == 1
    images=[dict(id=0,file_name='fog/val/a.png')]
    gt=[dict(id=1,image_id=0,category_id=5,bbox=[0,0,10,10],area=100,iscrowd=0),
        dict(id=2,image_id=0,category_id=5,bbox=[20,20,100,100],area=10000,iscrowd=1)]
    rows=[dict(image_id=0,predictions={'5':[dict(box=[0,0,10,10],confidence=.9),dict(box=[25,25,30,30],confidence=.8)]})]
    e=coco_evaluate(images,gt,rows)
    assert precision_summary(e)['map50_95']==pytest.approx(1)
    counts=runtime_counts(e)
    assert (counts['tp'],counts['fp'],counts['gt'],counts['hits'])==(1,0,1,{1})
    assert resample_summary(e,[0,0])['bus']==pytest.approx(1)


def test_low_conf_tp_is_ap_only_and_iou_thresholds_differ():
    images=[dict(id=0,file_name='fog/val/a.png')]
    gt=[dict(id=1,image_id=0,category_id=6,bbox=[0,0,10,10],area=100,iscrowd=0)]
    rows=[dict(image_id=0,predictions={'6':[dict(box=[0,0,8,10],confidence=.1)]})]
    low=coco_evaluate(images,gt,rows)
    high=coco_evaluate(images,gt,rows,.25)
    assert precision_summary(low)['map50']==pytest.approx(1)
    assert 0<precision_summary(low)['map50_95']<1
    assert precision_summary(high)['map50']==0
    assert runtime_counts(high)['recall']==0


def test_pairing_rejects_missing_image():
    with pytest.raises(ValueError,match='一一对应'):
        coco_evaluate([dict(id=0)],[],[])


def test_pairing_rejects_duplicate_prediction_image():
    with pytest.raises(ValueError,match='一一对应'):
        coco_evaluate([dict(id=0)],[],[dict(image_id=0,predictions={}),dict(image_id=0,predictions={})])


def test_acdc_zero_based_annotation_id_is_not_unmatched_sentinel():
    images,gt=normalize_ground_truth(dict(images=[dict(id=0,file_name='fog/val/a.png')],
        annotations=[dict(id=0,image_id=0,category_id=24,bbox=[0,0,10,10],area=100,iscrowd=0)]))
    rows=[dict(image_id=0,predictions={'0':[dict(box=[0,0,10,10],confidence=.9)]})]
    assert gt[0]['id']==1 and gt[0]['source_annotation_id']==0
    e=coco_evaluate(images,gt,rows)
    assert runtime_counts(e)['tp']==1
    assert precision_summary(e)['map50_95']==pytest.approx(1)
    with pytest.raises(ValueError,match='正值'):
        coco_evaluate(images,[{**gt[0],'id':0}],rows)
