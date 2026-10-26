"""Offline NP2-R evidence audit. No model creation, forward, downloads or optimizers."""
from pathlib import Path
import csv, hashlib, json, math
from collections import Counter, defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
import sys
sys.path.insert(0,str(ROOT/'src'))
from saferestore.metrics.next_phase import CLASSES, ACDC_TO_COCO, coco_evaluate, normalize_ground_truth, precision_summary, runtime_counts
METHODS=('input','v4','C0','C1')

def load(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
def save(name,value):
 p=OUT/name
 with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 assert json.loads(p.read_text(encoding='utf-8'))==value

def table(name,rows):
 with (OUT/name).open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 assert len(list(csv.DictReader((OUT/name).open(encoding='utf-8-sig'))))==len(rows)

def xyxy(a):
 x,y,w,h=a['bbox'];return [x,y,x+w,y+h]
def iou(a,b):
 a,b=np.asarray(a,float),np.asarray(b,float);z=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]));inter=float(z.prod());den=float(np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()-inter);return inter/den if den else 0.
def flat(row):return [dict(category_id=int(c),confidence=p['confidence'],box=p['box']) for c,ps in row['predictions'].items() for p in ps]
def brief(p,box):return None if p is None else dict(**p,iou=iou(p['box'],box))
def best(ps,box,code,score=.001,min_iou=0,other=False):
 ps=[p for p in ps if (p['category_id']!=code if other else p['category_id']==code) and p['confidence']>=score and iou(p['box'],box)>=min_iou]
 return brief(max(ps,key=lambda p:(p['confidence'],iou(p['box'],box))) if ps else None,box)
def matches(ev):
 out={}
 for r in ev.evalImgs:
  if r is None:continue
  for gid,did,ignored in zip(r['gtIds'],r['gtMatches'][0],r['gtIgnore']):
   if did and not ignored:
    p=ev.cocoDt.anns[int(did)];out[int(gid)]={'category_id':p['category_id'],'confidence':p['score'],'box':xyxy(p),'dt_id':int(did)}
 return out

def supervision():
 d=load('results/np2_revision/supervision_probe.json');old=load('results/np2_external_probe/supervision_probe.json');a=load(d['manifest']['annotation'])
 assert d['status']=='completed' and d['manifest']['images']==old['manifest']['images']
 manifest=d['manifest']['images'];names=[r['file_name'] for r in manifest];assert len(names)==len(set(names))==64
 images={r['id']:r for r in a['images']};annotations={r['id']:r for r in a['annotations']};ids={r['image_id'] for r in manifest}
 gt_rows=[];image_rows=[];batch_rows=[];grad_rows=[];loss_rows=[];summary={};checks=[]
 for model,m in d['models'].items():
  pergt=m['per_gt'];assert len(pergt)==431
  actual={r['source_annotation_id'] for r in pergt};expected={r['id'] for r in a['annotations'] if r['image_id'] in ids and r['category_id'] in ACDC_TO_COCO and not r.get('iscrowd') and not r.get('ignore')}
  assert actual==expected and len(actual)==431
  for r in pergt:
   src=annotations[r['source_annotation_id']];assert ACDC_TO_COCO[src['category_id']]==r['category'] and src['image_id']==r['image_id']
   assert images[r['image_id']]['file_name']==r['image'] and np.allclose(xyxy(src),r['original_box_xyxy'],rtol=0,atol=1e-5)
   gt_rows.append(dict(model=model,source_gt_id=r['source_annotation_id'],eval_gt_id=r['source_annotation_id']+1,image_id=r['image_id'],file_name=r['image'],class_name=CLASSES[r['category']],positives=r['positives'],target_weight=r['weight'],positive_in_ignore=r['positive_anchors_in_ignore'],candidate_in_ignore=r['candidate_centers_in_ignore'],centers_stride8=r['anchor_centers_inside_by_stride']['8'],centers_stride16=r['anchor_centers_inside_by_stride']['16'],centers_stride32=r['anchor_centers_inside_by_stride']['32'],original_width=r['original_width'],original_height=r['original_height'],detector_width=r['detector_width'],detector_height=r['detector_height']))
  by_image=defaultdict(list)
  for r in pergt:by_image[r['image']].append(r)
  assert set(by_image)<=set(names)
  gb=m['gradient']['per_batch'];sb=m['batch_consistency']['per_batch']
  assert len(gb)==64 and [g['file_names'][0] for g in gb]==names and [f for b in sb for f in b['file_names']]==names
  for b in sb:
   assert abs(b['native_loss']-b['decomposed_loss'])<1e-4 and abs(sum(b['parts'].values())-b['native_loss'])<1e-4
   expected_ids=[r['source_annotation_id'] for f in b['file_names'] for r in by_image[f]];assert expected_ids==b['gt_ids']
   checks.append(dict(model=model,kind='saved_statistics',batch=b['batch'],file_names=b['file_names'],native_loss=b['native_loss'],decomposed_loss=b['decomposed_loss'],scalar_abs_diff=b['abs_diff'],native_box_cls_dfl_sum_abs_diff=abs(sum(b['parts'].values())-b['native_loss']),**{f'native_{k}':v for k,v in b['parts'].items()}))
   for code,cname in CLASSES.items():
    rr=[r for f in b['file_names'] for r in by_image[f] if r['category']==code]
    batch_rows.append(dict(model=model,statistics_batch=b['batch'],batch_images=len(b['file_names']),file_names=' | '.join(b['file_names']),class_name=cname,gt=len(rr),images_with_gt=len({r['image'] for r in rr}),positive_anchors=sum(r['positives'] for r in rr),zero_positive_gt=sum(r['positives']==0 for r in rr),target_weight=sum(r['weight'] for r in rr),class_loss_per_batch='NOT_SAVED',class_gradient_per_statistics_batch='NOT_SAVED'))
  for b in gb:
   assert b['loss_abs_diff']<1e-4 and b['grad']['relative_l2_error']<1e-4
   f=b['file_names'][0];assert b['gt_ids']==[r['source_annotation_id'] for r in by_image[f]]
   checks.append(dict(model=model,kind='saved_gradient',batch=b['batch'],file_names=b['file_names'],native_loss=b['native_loss'],decomposed_loss=b['decomposed_loss'],scalar_abs_diff=b['loss_abs_diff'],**b['grad']))
   for code,cname in CLASSES.items():
    rr=[r for r in by_image[f] if r['category']==code]
    row=dict(model=model,gradient_batch=b['batch'],image_id=next(r['image_id'] for r in manifest if r['file_name']==f),file_name=f,class_name=cname,gt=len(rr),positive_anchors=sum(r['positives'] for r in rr),zero_positive_gt=sum(r['positives']==0 for r in rr),target_weight=sum(r['weight'] for r in rr),positive_in_ignore=sum(r['positive_anchors_in_ignore'] for r in rr),native_gradient_norm=b['grad']['norm_native'],class_total_gradient_norm=b['class_grad_norm'][str(code)],class_norm_ratio_nonadditive=b['class_grad_norm_ratio_nonadditive'][str(code)],component_gradients='NOT_SAVED',gradient_attribution='class total over this image; not individual GT')
    image_rows.append(row)
    grad_rows.append(dict(**row,relative_l2_error=b['grad']['relative_l2_error'],cosine=b['grad']['cosine'],max_abs_diff=b['grad']['max_abs_diff']))
  cs={}
  for code,cname in CLASSES.items():
   rr=[r for r in pergt if r['category']==code];stored=m['per_class'][str(code)]
   assert len(rr)==stored['gt'] and sum(r['positives'] for r in rr)==stored['positives'] and sum(r['positives']==0 for r in rr)==stored['zero']
   cs[cname]=dict(unique_images_with_gt=len({r['image'] for r in rr}),unique_gt=len(rr),positive_anchors=sum(r['positives'] for r in rr),zero_positive_gt=sum(r['positives']==0 for r in rr),zero_ratio=stored['zero_ratio'],target_weight=sum(r['weight'] for r in rr),positive_in_ignore=sum(r['positive_anchors_in_ignore'] for r in rr),candidate_in_ignore=sum(r['candidate_centers_in_ignore'] for r in rr),loss_cls_pos=stored['cls_pos_contribution'],loss_cls_neg=stored['cls_neg_contribution'],loss_box=stored['box_contribution'],loss_dfl=stored['dfl_contribution'],gradient_class_total_mean_norm=m['gradient']['per_class_mean_norm'][cname],gradient_norm_mean_ratio_nonadditive=m['gradient']['per_class_mean_norm_ratio_nonadditive'][cname],gradient_when_no_class_gt_mean=float(np.mean([r['class_total_gradient_norm'] for r in image_rows if r['model']==model and r['class_name']==cname and r['gt']==0])) if any(r['model']==model and r['class_name']==cname and r['gt']==0 for r in image_rows) else None)
   loss_rows.append(dict(model=model,class_name=cname,**cs[cname]))
  summary[model]=dict(unique_images=64,unique_gt=431,statistics_batches=16,gradient_batches=64,classes=cs,max_vector_relative_l2=max(b['grad']['relative_l2_error'] for b in gb),min_vector_cosine=min(b['grad']['cosine'] for b in gb),max_vector_absolute_error=max(b['grad']['max_abs_diff'] for b in gb),native_loss_sum16=m['native_loss_total_sum'],class_components_sum16=m['reported_class_component_total'])
  assert abs(sum(sum(cs[c][k] for k in ['loss_cls_pos','loss_cls_neg','loss_box','loss_dfl']) for c in cs)-m['native_loss_total_sum'])<1e-4
 ignored=[r for r in a['annotations'] if r['image_id'] in ids and (r.get('iscrowd') or r.get('ignore') or r['category_id'] not in ACDC_TO_COCO)]
 mask_audit=dict(ignored_regions=len(ignored),rider=sum(r['category_id']==25 for r in ignored),crowd=sum(bool(r.get('iscrowd')) for r in ignored),explicit_ignore=sum(bool(r.get('ignore')) for r in ignored),bus_train_excluded_by_annotation_flags=[r['id'] for r in ignored if r['category_id'] in (28,31)],mask_rule='(~mask[...,None]) OR (target_scores>0): positive soft targets always preserved; unsupervised73 classes multiply zero',zero_bus_mask_overlap=[dict(model=r['model'],source_gt_id=r['source_gt_id'],candidate_in_ignore=r['candidate_in_ignore']) for r in gt_rows if r['class_name']=='bus' and r['positives']==0],limits='No image-semantic audit of the official crowd flags. Saved target weights do not store full anchor indices or per-anchor target_scores.')
 save('supervision_audit.json',dict(status='EXISTING_EVIDENCE_VALIDATED',models=summary,mask_audit=mask_audit,loss_units='sum of16 native mean-batch4 gained scalars; not mean of64 images',gradient_units='mean of64 batch1 parameter-vector norms; not norm of full dataset gradient',missing=['individual class-term gradients','per-class per-batch4 loss scalar breakdown','full native assignment tensor/index trace']))
 save('batch_vector_checks.json',checks);table('supervision_per_gt.csv',gt_rows);table('supervision_per_image.csv',image_rows);table('supervision_per_batch4.csv',batch_rows);table('supervision_gradients_per_batch1.csv',grad_rows);table('class_loss_gradient_summary.csv',loss_rows)
 return d,summary

def prediction_audit(sup):
 a=load('data/raw/ACDC_detection/gt_detection/instancesonly_val_gt_detection.json');images,anns=normalize_ground_truth(a);noncrowd=[a for a in anns if a['category_id'] in (5,6) and not a.get('iscrowd')];im={i['id']:i for i in images}
 assert len(noncrowd)==95
 rows={m:{p:{r['image_id']:r for r in load(f'results/next_phase/predictions/{m}/{p}.json')['per_image']} for p in ['ap001','runtime025']} for m in METHODS}
 for m in METHODS:
  for p in rows[m]:assert set(rows[m][p])==set(im)
 old=load('results/next_phase/failure_cases.json');old_index={(r['image_id'],r['gt_id'],r['method']):r for r in old['gt_cases']}
 official={};scores={}
 for m in METHODS:
  official[m]={}
  for p,conf,det in [('ap001',.001,100),('runtime025',.25,300)]:
   ev=coco_evaluate(images,anns,list(rows[m][p].values()),conf,det);official[m][p]=matches(ev)
   if p=='ap001':scores[m]=precision_summary(ev)
  print('offline COCO matched',m,flush=True)
 output=[];wide=[];case_rows=[];nms_rows=[];legacy_changes=[]
 for gt in noncrowd:
  gid,cid,iid=gt['id'],gt['category_id'],gt['image_id'];box=xyxy(gt);entry=dict(image_id=iid,gt_id=gid,source_gt_id=gt['source_annotation_id'],file_name=im[iid]['file_name'],weather=im[iid]['file_name'].split('/')[0],class_name=CLASSES[cid],original_box=box,original_width=gt['bbox'][2],original_height=gt['bbox'][3],methods={})
  w={k:entry[k] for k in ['image_id','gt_id','source_gt_id','file_name','weather','class_name','original_width','original_height']}
  for m in METHODS:
   ps=flat(rows[m]['ap001'][iid]);rt=brief(official[m]['runtime025'].get(gid),box);ap=brief(official[m]['ap001'].get(gid),box);correct=best(ps,box,cid,min_iou=.5);high=best(ps,box,cid,score=.25);other=best(ps,box,cid,score=.25,min_iou=.5,other=True)
   pre=rows[m]['ap001'][iid].get('pre_nms',[])
   candidates=[p for p in pre if p['category_id']==cid and p['confidence']>=.25 and iou(p['box'],box)>=.5]
   possible=[]
   for p in candidates:
    equivalent=[q for q in ps if q['category_id']==cid and abs(q['confidence']-p['confidence'])<1e-7 and iou(q['box'],p['box'])>.98]
    if equivalent:continue
    for q in ps:
     if q['category_id']==cid and q['confidence']>p['confidence'] and iou(p['box'],q['box'])>.7:
      shifts=[iou([p['box'][0]+dx,p['box'][1]+dy,p['box'][2]+dx,p['box'][3]+dy],box) for dx in [-2,0,2] for dy in [-2,0,2]]
      possible.append(dict(candidate=brief(p,box),survivor=brief(q,box),pair_iou=iou(p['box'],q['box']),candidate_gt_iou_translation_2px_range=[min(shifts),max(shifts)],status='UNRESOLVED_NATIVE_NMS_TRACE',reason='No original anchor/NMS kept/suppressed indices; coordinates are decoded and clipped; confidence association is not operator trace.'))
      break
   obs=[]
   input_hit=gid in official['input']['runtime025'];own=rt is not None
   if input_hit and not own and correct and correct['confidence']<.25:obs.append('same_class_postNMS_IoU50_score_below_runtime_threshold')
   if not own and other:obs.append('high_score_other_class_overlap_candidate_not_unique_object_identity')
   if not own and high and .1<=high['iou']<.5:obs.append('high_score_same_class_below_IoU50_candidate')
   if not input_hit:obs.append('input_runtime_miss')
   if possible:obs.append('decoded_NMS_pair_observed_operator_cause_UNRESOLVED')
   if not own and not obs:obs.append('UNRESOLVED_NO_DISCRIMINATING_SAVED_CANDIDATE')
   ev=dict(runtime_hit=own,ap001_iou50_hit=ap is not None,runtime_match=rt,ap001_match=ap,best_score_same_class_IoU50_postNMS=correct,highest_score_same_class_postNMS=high,highest_score_other_class_IoU50_postNMS=other,pre_nms_available=bool(pre),pre_nms_high_correct_count=len(candidates),nms_status='UNRESOLVED_NATIVE_NMS_TRACE' if not own else 'not_needed_for_hit',nms_pairs=possible[:3],observations=obs,causal_root='UNRESOLVED')
   entry['methods'][m]=ev;w[m+'_runtime025']='hit' if own else 'miss';w[m+'_ap001_iou50']='hit' if ap else 'miss';w[m+'_best_IoU50_score']=None if correct is None else correct['confidence'];w[m+'_best_IoU50_iou']=None if correct is None else correct['iou']
   case_rows.append(dict(image_id=iid,gt_id=gid,source_gt_id=gt['source_annotation_id'],method=m,class_name=CLASSES[cid],weather=entry['weather'],runtime025_hit=own,ap001_iou50_hit=ap is not None,lost_vs_input=input_hit and not own,gained_vs_input=not input_hit and own,best_same_IoU50_score=None if correct is None else correct['confidence'],best_same_IoU50_iou=None if correct is None else correct['iou'],other_class=None if other is None else CLASSES[other['category_id']],other_score=None if other is None else other['confidence'],other_iou=None if other is None else other['iou'],observations=' | '.join(obs),nms_status=ev['nms_status'],root_cause='UNRESOLVED'))
   if m!='input':
    prev=old_index[(iid,gid,m)]['comparison'][m]
    assert prev['runtime_hit']==own
    assert (prev['ap50_match'] is not None)==(ap is not None)
   for pair in possible[:3]:nms_rows.append(dict(image_id=iid,gt_id=gid,method=m,**pair))
  for m in METHODS[1:]:
   cur=entry['methods'][m]['best_score_same_class_IoU50_postNMS'];base=entry['methods']['input']['best_score_same_class_IoU50_postNMS']
   entry['methods'][m]['score_delta_vs_input']=cur['confidence']-base['confidence'] if cur and base else None
  output.append(entry);wide.append(w)
 assert len({(r['image_id'],r['gt_id']) for r in output})==95
 # Strictly compare saved NP2-R dual-protocol matrix for rain/snow.
 import csv as csv_module
 previous=list(csv_module.DictReader((ROOT/'results/np2_revision/bus_train_gt_matrix.csv').open(encoding='utf-8')))
 lookup={(r['image_id'],r['gt_id']):r for r in output}
 for r in previous:
  now=lookup[(int(r['image_id']),int(r['gt_id']))]
  for m in METHODS:assert r[m]==('hit' if now['methods'][m]['runtime_hit' if r['protocol']=='runtime025' else 'ap001_iou50_hit'] else 'miss')
 counts={}
 for scope,weathers in [('all406',{'fog','rain','night','snow'}),('fog',{'fog'}),('rain',{'rain'}),('night',{'night'}),('snow',{'snow'})]:
  counts[scope]={}
  for cname in ['bus','train']:
   rr=[r for r in output if r['class_name']==cname and r['weather'] in weathers]
   counts[scope][cname]={'GT':len(rr),'unique_images':len({r['image_id'] for r in rr}),'methods':{}}
   for m in METHODS:
    own={r['gt_id'] for r in rr if r['methods'][m]['runtime_hit']};base={r['gt_id'] for r in rr if r['methods']['input']['runtime_hit']}
    counts[scope][cname]['methods'][m]=dict(runtime025_hits=len(own),ap001_iou50_hits=sum(r['methods'][m]['ap001_iou50_hit'] for r in rr),lost_vs_input=sorted(base-own),gained_vs_input=sorted(own-base))
 save('validation_cases.json',dict(scope='all406 validation;95 unique noncrowd bus/trainGT',protocol=dict(ap='.001 COCO IoU.50:.95 maxDets100; AP hit field at IoU.50',runtime='.25 COCO IoU.50 maxDets300; saved separate NMS outputs',nms='class-aware IoU.7; global maxDet300'),cases=output,counts=counts,official_metrics=scores,limitations=['PostNMS candidates have no persistent anchor identity.','Stored preNMS only max class per anchor, no 80class logits or original anchor/NMS indices.','No validation backward or model forward performed.']))
 save('nms_evidence_unresolved.json',nms_rows);table('bus_train_gt_matrix_all406.csv',wide);table('validation_case_observations.csv',case_rows)
 return output,counts

def links(sup,cases):
 scale=load('results/np2_revision/scale_order_probe.json');links=[];pergt={m:{(r['image_id'],r['source_annotation_id']):r for r in d['per_gt']} for m,d in sup['models'].items()}
 for model,md in scale['models'].items():
  gradients={b['file_names'][0]:b for b in sup['models'][model]['gradient']['per_batch']}
  for row in md['per_image']:
   for gt in row['paths']['A_fp32']['per_gt']:
    if gt['category'] not in (0,2,5,6):continue
    key=(row['image_id'],gt['source_gt_id']);r=pergt[model][key];b=gradients[row['file_name']]
    states={p:next(g for g in row['paths'][p]['per_gt'] if g['source_gt_id']==gt['source_gt_id']) for p in ['D_fp32','A_fp32','B_fp32']}
    links.append(dict(model=model,image_id=row['image_id'],source_gt_id=gt['source_gt_id'],eval_gt_id=gt['gt_id'],class_name=gt['class_name'],file_name=row['file_name'],positive_anchors_batch4=r['positives'],target_weight_batch4=r['weight'],class_total_gradient_norm_image=b['class_grad_norm'][str(gt['category'])],class_gradient_not_per_gt=True,original_size=gt['original_size'],detector_size=gt['detector_size'],D=states['D_fp32']['post_nms_runtime025'],A=states['A_fp32']['post_nms_runtime025'],B=states['B_fp32']['post_nms_runtime025'],evidence='DIRECT_SAME_TRAIN_FRAME_AND_GT; observed association, not causal gradient ablation',precision_limit='supervision historical forward TF32 vs scale TF32off; not same exact graph'))
 save('same_train_gt_evidence_links.json',links)
 manifest=sup['manifest']['images'];seq=lambda s:(s.split('/')[0],s.split('/')[2]);pop=load(sup['manifest']['annotation']);alltrain={seq(r['file_name']) for r in pop['images']};fixed={seq(r['file_name']) for r in manifest}
 vl=[]
 for r in cases:
  key=seq(r['file_name']);vl.append(dict(image_id=r['image_id'],gt_id=r['gt_id'],class_name=r['class_name'],file_name=r['file_name'],same_image_in_fixed64=any(i['file_name']==r['file_name'] for i in manifest),same_sequence_in_fixed64=key in fixed,same_sequence_in_train1600=key in alltrain,linked_train_images=[i['file_name'] for i in manifest if seq(i['file_name'])==key],causal_link='UNRESOLVED: shared class/weather/sequence does not identify same physical object or historical gradient'))
 save('validation_training_links.json',vl)
 print('same-trainGT links',len(links),'val95 seq links',sum(r['same_sequence_in_fixed64'] for r in vl),flush=True)

if __name__=='__main__':
 sup,_=supervision();cases,_=prediction_audit(sup);links(sup,cases)
 save('offline_checks.json',dict(status='PASS',native_statistics_batches=48,native_gradient_batches=192,unique_train_images=64,unique_train_GT=431,model_GT_rows=1293,unique_val_bus_train_GT=95,model_val_GT_rows=380,new_model_forwards=0,optimizer_steps=0,downloads=0,legacy_match_status='All existing formal hits and corrected rain/snow matrix match recomputed official matching.'))
 print('offline evidence saved and readback verified',flush=True)