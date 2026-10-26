"""Read-only delivery validation. Does not import torch or run any model."""
from pathlib import Path
import csv,hashlib,json,math
from statistics import mean
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def run():
 pre=read(BASE/'preflight.json');final=read(BASE/'final_checks_v2.json');g=BASE/'fp64_component_gradients';summary=read(g/'component_gradient_summary.json')
 assert summary['status']=='completed' and final['status']=='PASS'
 assert summary['counters']['optimizer_steps']==summary['counters']['prediction_generation']==summary['counters']['validation_forwards']==0
 assert sha(BASE/'BUS_TRAIN_DIAGNOSIS_REPORT_v2.md')==final['report_sha256']
 for rel,v in pre['inputs'].items():assert sha(ROOT/rel)==v['sha256'],rel
 for row in read(BASE/'sample_image_hashes.json')['images']:assert sha(ROOT/'data/raw/ACDC_images/rgb_anon'/row['file_name'])==row['sha256']
 supervision=read(ROOT/'results/np2_revision/supervision_probe.json')
 geometry=read(BASE/'native_candidate_geometry_audit.json');native_masks={(r['model'],r['image_id'],r['source_gt_id']):set(r['native_expanded_candidate_indices']) for r in geometry['per_gt']}
 assert geometry['status']=='PASS_NATIVE_GEOMETRY'
 for path,value in read(BASE/'preflight_native_source_addendum.json')['inputs'].items():assert sha(Path(path))==value['sha256']
 count={'batches':0,'actual_native_assigner_calls':0,'vector_checks':0,'anchor_slots_checked':0,'GT_rows':0}
 for model in ['v4','C0','C1']:
  m=read(g/summary['models'][model]['result_file']);assert len(m['batches'])==64
  oldgt={(r['image_id'],r['source_annotation_id']):r for r in supervision['models'][model]['per_gt']}
  for b in m['batches']:
   indices=[idx for gt in b['GT'] for idx in gt['positive_anchor_indices']]
   assert len(indices)==len(set(indices)), 'Joint assignment double-counted an anchor'
   assert b==read(g/f'component_batches_{model}'/f"batch_{b['batch']:03d}.json")
   count['batches']+=1;count['actual_native_assigner_calls']+=b['assignment_calls'];assert b['assignment_calls']==1
   assert abs(sum(v[t]['loss'] for v in b['classes'].values() for t in ['cls_pos','cls_neg','box','dfl'])-b['native_loss'])<1e-4
   for v in b['vector_checks'].values():
    assert v['pass'] and v['relative_l2_error']<1e-4 and -1<=v['cosine']<=1 and math.isfinite(v['max_abs_diff'])
    count['vector_checks']+=1
   for gt in b['GT']:
    count['GT_rows']+=1;assert gt['positive_anchors']==len(gt['positive_anchor_indices'])==len(set(gt['positive_anchor_indices']))==len(gt['positive_anchor_centers'])==len(gt['positive_target_scores'])
    assert gt['same_as_saved_batch4'] and abs(sum(gt['positive_target_scores'])-gt['target_weight'])<1e-5
    for index,stride,center in zip(gt['positive_anchor_indices'],gt['positive_anchor_strides'],gt['positive_anchor_centers']):
     assert 0<=index<8400
     if index<6400:offset,cols,ss=0,80,8
     elif index<8000:offset,cols,ss=6400,40,16
     else:offset,cols,ss=8000,20,32
     i=index-offset;expected=[(i%cols+.5)*ss,(i//cols+.5)*ss]
     assert index in native_masks[(model,gt['image_id'],gt['source_annotation_id'])], 'Anchor outside actual native candidate rule'
     assert stride==ss and center==expected,(index,center,expected)
     count['anchor_slots_checked']+=1
  inv=m['invariants'];assert inv['model_tensor_before']==inv['model_tensor_after'] and inv['detector_before']==inv['detector_after'] and inv['checkpoint_before']==inv['checkpoint_after']
 assert count['batches']==192 and count['actual_native_assigner_calls']==192 and count['vector_checks']==768 and count['GT_rows']==1293
 cases=read(BASE/'validation_cases.json')['cases'];assert len(cases)==len({(r['image_id'],r['gt_id']) for r in cases})==95
 matrix=list(csv.DictReader((BASE/'bus_train_gt_matrix_all406.csv').open(encoding='utf-8-sig')));assert len(matrix)==95
 index={(r['image_id'],r['gt_id']):r for r in cases}
 for row in matrix:
  c=index[(int(row['image_id']),int(row['gt_id']))];assert int(row['source_gt_id'])+1==int(row['gt_id'])
  for model in ['input','v4','C0','C1']:
   assert row[model+'_runtime025']==('hit' if c['methods'][model]['runtime_hit'] else 'miss')
   assert row[model+'_ap001_iou50']==('hit' if c['methods'][model]['ap001_iou50_hit'] else 'miss')
 comp=list(csv.DictReader((BASE/'component_gradients_per_image.csv').open(encoding='utf-8-sig')));assert len(comp)==5376
 for row in comp:
  b=read(g/f"component_batches_{row['model']}"/f"batch_{int(row['batch']):03d}.json");r=b['classes'][row['class_name']][row['term']]
  assert float(row['gradient_norm_FP64'])==r['gradient_norm'] and float(row['gained_loss'])==r['loss']
 proposal=read(BASE/'next_minimal_experiment_v2.json');assert proposal['status']=='PROPOSED_ONLY_NOT_EXECUTED' and proposal['training_budget']==0 and proposal['optimizer']=='None'
 print(json.dumps({'status':'PASS','scope':'Offline consistency/hash/geometry validation; does not recompute raw autograd vectors or NMS','checks':count,'formal_prediction_inferences':0,'training_steps':0},ensure_ascii=False))
if __name__=='__main__':run()