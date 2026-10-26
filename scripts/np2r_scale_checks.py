"""CPU-only regression checks for NP2-R scale order geometry and evaluation.

Use --out with a new path under results/np2_revision to save evidence without
replacing an earlier check record. No training or GPU is used. The optional
--production-rect check uses the existing local YOLO checkpoint.
"""
import argparse
import copy
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import numpy as np
import torch
from scripts import np2_scale_order_probe as m


def checks(production_rect=False):
    records = []
    for shape in [(1080, 1920), (1920, 1080), (641, 961), (100, 130)]:
        rgb = np.random.default_rng(42).random((*shape, 3), dtype=np.float32)
        canvas, geom = m.letterbox_float(rgb)
        assert canvas.shape == (640, 640, 3) and canvas.dtype == np.float32
        boxes = [[0, 0, shape[1], shape[0]], [3, 7, 25, 31]]
        _, reference, _ = m.letterbox_rgb((rgb * 255).round().astype(np.uint8), boxes, [], 640, False)
        xyxy = m.transform_boxes(boxes, geom) / 640
        xywh = xyxy.copy()
        xywh[:, :2] = (xyxy[:, :2] + xyxy[:, 2:]) / 2
        xywh[:, 2:] = xyxy[:, 2:] - xyxy[:, :2]
        np.testing.assert_allclose(xywh, reference.numpy(), atol=1e-7, rtol=0)
        # An identity restorer must commute exactly with resize/common padding.
        a = m.common_pad(m.resize_float(rgb, geom), geom)
        b = m.letterbox_float(rgb.copy(), geom)[0]
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(m.quantize_canvas(a), m.quantize_canvas(b))
        records.append(dict(test='geometry_identity_symmetric_quantization', shape=list(shape),
                            max_abs_diff=float(np.max(np.abs(a-b)))))
    small = m.anchor_centers_in_box([1, 1, 7, 19])
    assert small['8']['strict_inside'] == 2
    records.append(dict(test='narrower_than_stride_can_have_centers', box=[1,1,7,19],
                        stride8_centers=small['8']['strict_centers']))
    assigned = dict(pred_bboxes=torch.tensor([[[10.,10.,30.,30.], [11.,11.,31.,31.], [100.,100.,120.,120.]]]),
                    stride_tensor=torch.ones(3,1), pred_scores=torch.full((1,3,80), -20.))
    assigned['pred_scores'][0,0,5] = 4
    assigned['pred_scores'][0,1,5] = 3
    assigned['pred_scores'][0,2,6] = 2
    row, indices = m.nms_rows(assigned, None, m.geometry(640,640),
        dict(id=1, file_name='rain/train/x/a.png', width=640, height=640), .25)
    assert sum(len(v) for v in row['predictions'].values()) == 2
    assert indices == [0,2]
    records.append(dict(test='same_native_nms_class_aware_025', indices=indices))
    images = [dict(id=0,file_name='rain/train/x/a.png',width=100,height=100),
              dict(id=1,file_name='snow/train/x/b.png',width=100,height=100)]
    anns = [dict(id=1,source_annotation_id=0,image_id=0,category_id=5,bbox=[10,10,20,20],area=400,iscrowd=0),
            dict(id=2,source_annotation_id=1,image_id=1,category_id=6,bbox=[10,10,20,20],area=400,iscrowd=0)]
    rows = [dict(image_id=i['id'],file_name=i['file_name'],
                 predictions={str(anns[j]['category_id']):[dict(box=[10,10,30,30],confidence=.9)]})
            for j,i in enumerate(images)]
    preds = {p:dict(ap001=copy.deepcopy(rows), runtime025=copy.deepcopy(rows)) for p in m.PATHS}
    preds['A_fp32']['runtime025'][0]['predictions'] = {}
    preds['A_fp32']['ap001'][0]['predictions'] = {}
    preds['B_fp32']['runtime025'][1]['predictions']['5'] = [dict(box=[70,70,90,90],confidence=.8)]
    result = m.evaluate_paths(images, anns, preds)
    assert np.isclose(result['D_fp32']['ap']['map50'], 1., atol=1e-12, rtol=0)
    assert result['A_fp32']['runtime']['lost_vs_D_fp32'] == [1]
    assert result['B_fp32']['runtime']['gained_vs_A_fp32'] == [1]
    assert result['B_fp32']['runtime']['new_fp_vs_D_fp32_count'] == 1
    assert result['B_fp32']['runtime025_gt']['1']['matched']
    assert result['D_fp32']['runtime']['hits'] == [1,2]
    records.append(dict(test='official_train_cohort_scoring_positive_ids_protocol_and_fp',
                        perfect_ap50=result['D_fp32']['ap']['map50'], lost_A_vs_D=[1],
                        gained_B_vs_A=[1], new_fp_B_vs_D=1))
    if production_rect:
        from ultralytics import YOLO
        torch.set_num_threads(4)
        detector = YOLO(str(m.ROOT / 'yolov8n.pt'))
        detector.model.to('cpu').float().eval().fuse(verbose=False)
        detector.model.requires_grad_(False)
        before = {k:v.detach().clone() for k,v in detector.model.state_dict().items()}
        image = dict(id=0,file_name='rain/train/synthetic/constant.png',width=1920,height=1080)
        _, shapes = m.actual_rect_predictions(detector,np.full((1080,1920,3),114,dtype=np.uint8),image,'cpu')
        assert shapes['ap001'] == [1,3,384,640] and shapes['runtime025'] == [1,3,384,640]
        assert all(torch.equal(before[k],v) for k,v in detector.model.state_dict().items())
        one_gt = [dict(id=1,image_id=0,category_id=5,bbox=[10,10,20,20],area=400,iscrowd=0)]
        base = dict(image_id=0,file_name=image['file_name'],predictions={'5':[dict(box=[10,10,30,30],confidence=.9)]})
        rect_preds = {key:{protocol:[copy.deepcopy(base)] for protocol in ('ap001','runtime025')}
                      for key in ('D_original_u8_rect','B_bf16_prequant_rect')}
        result = m.evaluate_paths([image],one_gt,rect_preds,direct_key='D_original_u8_rect',
                                 comparisons=[('B_bf16_prequant_rect','D_original_u8_rect')])
        assert result['B_bf16_prequant_rect']['runtime']['opr_to_D_original_u8_rect'] == 1.
        assert result['B_bf16_prequant_rect']['runtime']['lost_vs_D_original_u8_rect'] == []
        records.append(dict(test='actual_production_rect_predict_native_yolo_cpu',input_shape=shapes,
                            detector_state_unchanged=True,rect_baseline_fields_correct=True))
    return dict(all_pass=True, checks=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out')
    parser.add_argument('--production-rect', action='store_true')
    args = parser.parse_args()
    result = checks(args.production_rect)
    if args.out:
        target = (m.ROOT / args.out).resolve()
        assert target.is_relative_to((m.ROOT/'results/np2_revision').resolve())
        if target.exists():
            raise FileExistsError(target)
        m.write_json(target, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
