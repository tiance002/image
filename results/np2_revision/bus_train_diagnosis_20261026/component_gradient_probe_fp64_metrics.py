"""Fixed64 no-optimizer component-gradient audit; only training diagnostic forwards.
Native assigner is called ONCE per detector forward; hooks capture that actual result.
No validation images, NMS, prediction generation, training or parameter update.
"""
from pathlib import Path
import contextlib, hashlib, json, sys, time, os
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[3];BASE=Path(__file__).resolve().parent;OUT=BASE/'fp64_component_gradients'
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
from torch.utils.data import DataLoader
from saferestore.data.detection_gt import ACDCDetectionDataset,collate_detection_batch
from saferestore.losses.task_detection import FrozenYoloTaskLoss
from saferestore.models.nafnet import build_nafnet
from np2r_supervision_probe import class_components,compare_vectors,state_tensor_sha256,sha256_file,build_gt_metadata
from ultralytics.utils.tal import make_anchors
CODES=(0,1,2,3,5,6,7);NAMES={0:'person',1:'bicycle',2:'car',3:'motorcycle',5:'bus',6:'train',7:'truck'};TERMS=('cls_pos','cls_neg','box','dfl')

def write_new(path,value):
 with path.open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 assert json.loads(path.read_text(encoding='utf-8'))==value

def tensor_hash(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def gradient(term,params,zero):
 if not term.requires_grad:return torch.zeros_like(zero),False
 grads=torch.autograd.grad(term,params,retain_graph=True,allow_unused=True)
 return torch.cat([(g.detach().reshape(-1).cpu() if g is not None else torch.zeros(p.numel())) for p,g in zip(params,grads)]).float(),True

def dot64(a,b):
 return float(np.einsum('i,i->',a.detach().cpu().numpy(),b.detach().cpu().numpy(),dtype=np.float64,optimize=False))
def norm64(a):return dot64(a,a)**.5
def compare_vectors(a,b):
 diff=a-b;na,nb=norm64(a),norm64(b);raw=dot64(a,b)/(na*nb) if na and nb else 1.
 if abs(raw)>1+1e-10:raise RuntimeError('Invalid FP64 cosine '+str(raw))
 return dict(norm_sum=na,norm_native=nb,relative_l2_error=norm64(diff)/max(nb,1e-12),cosine=min(1.,max(-1.,raw)),cosine_raw_fp64=raw,max_abs_diff=float(diff.abs().max()))

def check(a,b):
 result=compare_vectors(a,b)
 # Near-zero cases use absolute error; regular vectors use unchanged1e-4 relative L2.
 result['pass']=bool(result['relative_l2_error']<1e-4 or (result['norm_native']<1e-10 and result['max_abs_diff']<1e-7))
 if not result['pass']:raise RuntimeError('Vector mismatch '+str(result))
 return result

def run():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 torch.manual_seed(42);np.random.seed(42)
 device='cuda' if torch.cuda.is_available() else 'cpu'
 original=json.loads((ROOT/'results/np2_revision/supervision_probe.json').read_text(encoding='utf-8'))
 expected=json.loads((BASE/'preflight.json').read_text(encoding='utf-8'))
 names=[r['file_name'] for r in original['manifest']['images']]
 dataset=ACDCDetectionDataset(ROOT/original['manifest']['annotation'],ROOT/'data/raw/ACDC_images/rgb_anon',image_size=640,seed=42)
 idx={r['file_name']:i for i,r in enumerate(dataset.records)};metadata=build_gt_metadata(ROOT/original['manifest']['annotation'],dataset)
 loader=DataLoader(torch.utils.data.Subset(dataset,[idx[n] for n in names]),batch_size=1,shuffle=False,num_workers=0,collate_fn=collate_detection_batch)
 loss_fn=FrozenYoloTaskLoss(ROOT/'yolov8n.pt',device=device)
 detector_initial=state_tensor_sha256(loss_fn.detector)
 counters={'restoration_forward':0,'detector_forward':0,'native_assigner_calls':0,'autograd_calls':0,'optimizer_created':0,'optimizer_steps':0,'prediction_generation':0,'validation_forwards':0}
 capture={}
 def detector_hook(module,args,output):capture['output']=output;counters['detector_forward']+=1
 def assign_hook(module,args,output):capture['assign_args']=args;capture['assign_result']=output;counters['native_assigner_calls']+=1
 dh=loss_fn.detector.register_forward_hook(detector_hook);ah=loss_fn.criterion.assigner.register_forward_hook(assign_hook)
 native_get=loss_fn.criterion.get_assigned_targets_and_loss
 def record_native(*args,**kwargs):
  value=native_get(*args,**kwargs);capture['native_parts']=value[1];return value
 loss_fn.criterion.get_assigned_targets_and_loss=record_native
 report={'status':'running','manifest_source_sha256':sha256_file(ROOT/'results/np2_revision/supervision_probe.json'),'fixed_manifest':original['manifest']['images'],'batch_size':1,'precision':{'forward_cudnn_tf32':torch.backends.cudnn.allow_tf32,'backward_cudnn_tf32':False,'matmul_tf32':False,'autocast':False},'attribution':'Four terms per class on the EXACT same captured native joint-GT assignment. Classification positive means target_scores>0; negative means zero target for this class. Other class GT anchors can have negative BCE for this class.','gradient_space':'All restoration network parameters in state order; mean norms nonadditive; vectors not interpreted as causal effects.','models':{},'no_optimizer_created':True,'optimizer_steps':0,'user_authorized_train_diagnostic_forwards':True,'metric_reduction':'CPU FP64 einsum for dot and squared-norm; model/gradients remain FP32','storage_revision':'parameter gradient vectors accumulated on CPU float32; native/sum checks also CPU; old completed batch0 retained; interrupted partial work disclosed separately'}
 write_new(OUT/'gradient_run_started_fp64.json',{'source_sha256':sha256_file(Path(__file__)),'fixed_images':64,'models':list(original['models']),'scope':'Necessary autograd only; no new predictions','resource_hashes_prevalidated':True})
 for model_name,prior in original['models'].items():
  result_path=OUT/f'component_gradients_{model_name}.json';assert not result_path.exists()
  checkpoint=ROOT/prior['checkpoint'];before_hash=sha256_file(checkpoint);assert before_hash==prior['checkpoint_sha256']==expected['inputs'][prior['checkpoint']]['sha256']
  state=torch.load(checkpoint,map_location='cpu',weights_only=False);model=build_nafnet().to(device).float().eval();model.load_state_dict(state['ema'],strict=True);del state
  params=[p for p in model.parameters() if p.requires_grad];before_state=state_tensor_sha256(model);batches=[];start=time.perf_counter()
  oldgt={(r['image_id'],r['source_annotation_id']):r for r in prior['per_gt']};oldgrad={b['file_names'][0]:b for b in prior['gradient']['per_batch']}
  directory=OUT/f'component_batches_{model_name}';directory.mkdir(exist_ok=True)
  completed=sorted(directory.glob('batch_*.json'))
  batches=[json.loads(p.read_text(encoding='utf-8')) for p in completed]
  if batches:
   assert [b['batch'] for b in batches]==list(range(len(batches)))
   counters.update(batches[-1]['counters_after'])
   print('Resume',model_name,'preserving',len(batches),'completed batches',flush=True)
  for bi,batch in enumerate(loader):
   if bi<len(completed):continue
   f=batch['file_names'][0];capture.clear();counts_before=dict(counters)
   images=batch['img'].to(device);restored=model(images);counters['restoration_forward']+=1
   native,native_stats=loss_fn(restored,batch)
   assert counters['native_assigner_calls']-counts_before['native_assigner_calls']==1
   criterion=loss_fn.criterion;raw=criterion.parse_output(capture['output']);pred_distri=raw['boxes'].permute(0,2,1).contiguous();pred_scores=raw['scores'].permute(0,2,1).contiguous()
   anchors,strides=make_anchors(raw['feats'],criterion.stride,.5)
   labels,target_boxes,target_scores,fg,gtidx=capture['assign_result']
   gtlabels,gtboxes=capture['assign_args'][3:5]
   assigned=dict(pred_distri=pred_distri,pred_scores=pred_scores,anchor_points=anchors,stride_tensor=strides,pred_bboxes=criterion.bbox_decode(anchors,pred_distri),target_labels=labels,target_bboxes=target_boxes,target_scores=target_scores,fg_mask=fg,target_gt_idx=gtidx)
   bce=criterion.bce(pred_scores,target_scores.to(pred_scores.dtype))*criterion.class_weights
   pc=class_components(criterion,assigned,bce);tss=target_scores.sum().clamp_min(1)
   parts={}
   for code in CODES:
    positive=fg & (target_scores[...,code]>0)
    cp=bce[...,code].masked_select(positive).sum()/tss*criterion.hyp.cls
    cn=bce[...,code].masked_select(~positive).sum()/tss*criterion.hyp.cls
    parts[code]=dict(cls_pos=cp,cls_neg=cn,box=pc[code]['box'],dfl=pc[code]['dfl'])
   native_parts=capture['native_parts'];scalar_parts={'box':sum(p['box'] for p in parts.values()),'cls':sum(p['cls_pos']+p['cls_neg'] for p in parts.values()),'dfl':sum(p['dfl'] for p in parts.values())}
   scalar_diff={k:abs(float(scalar_parts[k].detach())-float(native_parts[j].detach())) for j,k in enumerate(['box','cls','dfl'])}
   assert max(scalar_diff.values())<1e-4 and abs(sum(float(v.detach()) for v in scalar_parts.values())-float(native.detach()))<1e-4
   gtrows=[]
   for local,meta in enumerate(metadata[f]):
    picked=fg[0] & (gtidx[0]==local);code=int(batch['cls'][local]);pos_count=int(picked.sum());scores=target_scores[0][picked].sum()
    prior_gt=oldgt[(meta['image_id'],meta['source_annotation_id'])]
    gtrows.append(dict(**meta,positive_anchors=pos_count,target_weight=float(scores),positive_anchor_indices=torch.where(picked)[0].tolist(),positive_anchor_strides=strides.flatten()[picked].int().cpu().tolist(),positive_target_scores=target_scores[0,picked,code].detach().cpu().tolist(),positive_anchor_centers=(anchors*strides)[picked].cpu().tolist(),positive_in_ignore=int((picked & criterion.bce.mask[0]).sum()),saved_batch4_positives=prior_gt['positives'],same_as_saved_batch4=pos_count==prior_gt['positives']))
   with torch.backends.cudnn.flags(enabled=torch.backends.cudnn.enabled,benchmark=torch.backends.cudnn.benchmark,benchmark_limit=torch.backends.cudnn.benchmark_limit,deterministic=torch.backends.cudnn.deterministic,allow_tf32=False):
    native_vec=torch.zeros(sum(p.numel() for p in params),dtype=torch.float32)
    native_vec,_=gradient(native,params,native_vec);counters['autograd_calls']+=1
    sums={t:torch.zeros_like(native_vec) for t in TERMS};all_vec=torch.zeros_like(native_vec);class_records={}
    for code in CODES:
     class_vec=torch.zeros_like(native_vec);rec={}
     for term in TERMS:
      vec,computed=gradient(parts[code][term],params,native_vec);counters['autograd_calls']+=int(computed)
      class_vec+=vec;sums[term]+=vec
      rec[term]=dict(loss=float(parts[code][term].detach()),gradient_norm=norm64(vec),dot_native=dot64(vec,native_vec),cosine_native=dot64(vec,native_vec)/(norm64(vec)*norm64(native_vec)) if norm64(vec) and norm64(native_vec) else None,vector_sha256=tensor_hash(vec))
      del vec
     class_direct,computed=gradient(pc[code]['total'],params,native_vec);counters['autograd_calls']+=int(computed)
     class_check=check(class_vec,class_direct)
     rec['four_terms_vs_class_total_vector']=class_check;rec['class_total_gradient_norm']=norm64(class_direct);rec['four_terms_loss_sum']=sum(rec[t]['loss'] for t in TERMS)
     all_vec+=class_vec;class_records[NAMES[code]]=rec
     del class_direct,class_vec
    vectors={'all_28_terms_vs_native':check(all_vec,native_vec)}
    for j,k in enumerate(['box','cls','dfl']):
     nv,computed=gradient(native_parts[j],params,native_vec);counters['autograd_calls']+=int(computed)
     sv=sums['cls_pos']+sums['cls_neg'] if k=='cls' else sums[k]
     vectors[k+'_terms_vs_native_component']=check(sv,nv)
     del nv,sv
   saved=oldgrad[f];record=dict(batch=bi,file_names=[f],image_id=metadata[f][0]['image_id'] if metadata[f] else next(r['image_id'] for r in original['manifest']['images'] if r['file_name']==f),GT=gtrows,native_loss=float(native.detach()),native_parts={k:float(native_parts[j].detach()) for j,k in enumerate(['box','cls','dfl'])},scalar_parts_abs_diff=scalar_diff,native_vs_previous_batch1_loss_abs_diff=abs(float(native.detach())-saved['native_loss']),classes=class_records,vector_checks=vectors,native_vector_sha256=tensor_hash(native_vec),sum_vector_sha256=tensor_hash(all_vec),assignment_calls=1,counters_after=dict(counters))
   write_new(directory/f'batch_{bi:03d}.json',record);batches.append(record)
   assert all(p.grad is None for p in params) and all(p.grad is None for p in loss_fn.detector.parameters())
   if (bi+1)%8==0:print(model_name,'component gradients',bi+1,'/64','elapsed',round(time.perf_counter()-start,1),'s','maxerr',max(v['relative_l2_error'] for v in vectors.values()),flush=True)
   del restored,native,raw,pred_distri,pred_scores,assigned,bce,pc,parts,native_parts,scalar_parts,sums,all_vec,native_vec,target_scores,target_boxes,fg,gtidx,labels,gtlabels,gtboxes,images,record
   del cp,cn
   capture.clear()
   if device=='cuda':
    torch.cuda.empty_cache()
    print('memory',model_name,bi,'allocatedMiB',round(torch.cuda.memory_allocated()/1048576,1),'reservedMiB',round(torch.cuda.memory_reserved()/1048576,1),flush=True)
  invariants=dict(model_tensor_before=before_state,model_tensor_after=state_tensor_sha256(model),checkpoint_before=before_hash,checkpoint_after=sha256_file(checkpoint),detector_before=detector_initial,detector_after=state_tensor_sha256(loss_fn.detector),optimizer_created=False,optimizer_steps=0)
  assert invariants['model_tensor_before']==invariants['model_tensor_after'] and invariants['checkpoint_before']==invariants['checkpoint_after'] and invariants['detector_before']==invariants['detector_after']
  model_result=dict(checkpoint=prior['checkpoint'],checkpoint_sha256=before_hash,batch_size=1,batches=batches,invariants=invariants,elapsed_seconds=time.perf_counter()-start)
  write_new(result_path,model_result);report['models'][model_name]=dict(checkpoint=prior['checkpoint'],checkpoint_sha256=before_hash,result_file=result_path.name,batches=64,all_vector_checks_pass=True,max_relative_l2=max(v['relative_l2_error'] for b in batches for v in b['vector_checks'].values()),min_cosine=min(v['cosine'] for b in batches for v in b['vector_checks'].values()),max_absolute_error=max(v['max_abs_diff'] for b in batches for v in b['vector_checks'].values()),all_anchor_counts_equal_saved_batch4=all(g['same_as_saved_batch4'] for b in batches for g in b['GT']),native_loss_previous_maxdiff=max(b['native_vs_previous_batch1_loss_abs_diff'] for b in batches),invariants=invariants)
  del model,params,batches;torch.cuda.empty_cache() if device=='cuda' else None
 dh.remove();ah.remove();loss_fn.criterion.get_assigned_targets_and_loss=native_get
 report.update(status='completed',counters=counters,detector_file_sha256=sha256_file(ROOT/'yolov8n.pt'))
 assert report['detector_file_sha256']==original['detector_sha256'] and counters['native_assigner_calls']==counters['restoration_forward']==counters['detector_forward']==192
 write_new(OUT/'component_gradient_summary.json',report)
 print('ALL component gradients completed',counters,flush=True)

if __name__=='__main__':
 OUT.mkdir(exist_ok=False)
 run()