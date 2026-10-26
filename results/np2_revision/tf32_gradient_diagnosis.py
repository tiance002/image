"""One fixed original train image; precision diagnosis only, no optimizer."""
from pathlib import Path
import json,sys,gc
ROOT=Path.cwd();sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import torch
from scripts.np2r_supervision_probe import (build_sample,batch_forward,decomposed_total,grad_vector,compare_vectors,state_tensor_sha256,sha256_file,SUPERVISED)
from saferestore.data.detection_gt import ACDCDetectionDataset,collate_detection_batch
from saferestore.losses.task_detection import FrozenYoloTaskLoss
from saferestore.models.nafnet import build_nafnet
OUT=ROOT/'results/np2_revision/supervision_tf32_diagnosis.json'
assert not OUT.exists()
torch.set_num_threads(4)
annotation=ROOT/'data/raw/ACDC_detection/gt_detection/instancesonly_train_gt_detection.json'
rows,_=build_sample(annotation)
dataset=ACDCDetectionDataset(annotation,ROOT/'data/raw/ACDC_images/rgb_anon',image_size=640,seed=42)
order={r['file_name']:i for i,r in enumerate(dataset.records)}
batch=collate_detection_batch([dataset[order[rows[0]['file_name']]]]);images=batch['img'].cuda()
checkpoint=ROOT/'runs/real_v4_allweather/latest.pt'
state=torch.load(checkpoint,map_location='cpu',weights_only=False)
model=build_nafnet().cuda().float().eval();model.load_state_dict(state['ema']);del state
loss_fn=FrozenYoloTaskLoss(ROOT/'yolov8n.pt',device='cuda')
parameters=list(model.parameters())
model_before=state_tensor_sha256(model);detector_before=state_tensor_sha256(loss_fn.detector)
original_flags={'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic}
report={'scope':'one first original fixed64 train image; no updates','image':rows[0], 'original_flags':original_flags,'cases':[]}
for label,forward_tf32,backward_tf32 in [('tf32_forward_and_backward',True,True),('tf32_forward_only',True,False),('tf32_disabled',False,False)]:
 torch.backends.cudnn.allow_tf32=forward_tf32;torch.backends.cuda.matmul.allow_tf32=False
 model.zero_grad(set_to_none=True)
 out=batch_forward(loss_fn,model(images),batch,keep_graph=True)
 a=out['assigned'];signature={str(c):int((a['fg_mask'] & (a['target_labels'].long()==c)).sum()) for c in SUPERVISED}
 torch.backends.cudnn.allow_tf32=backward_tf32;torch.backends.cuda.matmul.allow_tf32=False
 native=out['native_scalar'];manual=decomposed_total(out['per_class'])
 native.backward(retain_graph=True);native_vector=grad_vector(parameters)
 model.zero_grad(set_to_none=True);manual.backward(retain_graph=True);manual_vector=grad_vector(parameters)
 summed=torch.zeros_like(native_vector)
 for code in SUPERVISED:
  model.zero_grad(set_to_none=True)
  term=out['per_class'][code]['total']
  if term.requires_grad:
   term.backward(retain_graph=True);current=grad_vector(parameters)
  else:current=torch.zeros_like(native_vector)
  summed+=current
 result={'name':label,'forward_cudnn_tf32':forward_tf32,'backward_global_cudnn_tf32':backward_tf32,'native_loss':float(native),'manual_loss':float(manual),'loss_abs_diff':abs(float(native)-float(manual)),'positive_anchors_by_class':signature,'native_vs_manual_total_gradient':compare_vectors(manual_vector,native_vector),'native_vs_sum_class_gradients':compare_vectors(summed,native_vector),'same_manual_graph_sumclass_vs_total_gradient':compare_vectors(summed,manual_vector)}
 report['cases'].append(result)
 OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n');assert json.loads(OUT.read_text(encoding='utf-8'))['cases'][-1]['name']==label
 print(json.dumps(result),flush=True)
 model.zero_grad(set_to_none=True)
 del out,a,native,manual,native_vector,manual_vector,summed,current,term
 gc.collect();torch.cuda.empty_cache()
for key,value in original_flags.items():
 if key=='cudnn_allow_tf32':torch.backends.cudnn.allow_tf32=value
 if key=='matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32=value
report['model_unchanged']=state_tensor_sha256(model)==model_before
report['detector_including_bn_unchanged']=state_tensor_sha256(loss_fn.detector)==detector_before
assert report['model_unchanged'] and report['detector_including_bn_unchanged']
report['no_optimizer_created']=True;report['optimizer_steps']=0
OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n');assert json.loads(OUT.read_text())['model_unchanged']
