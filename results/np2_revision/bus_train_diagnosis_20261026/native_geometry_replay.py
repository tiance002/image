"""Recheck candidate geometry with pinned native method; no model or assignment forward."""
from pathlib import Path
import hashlib,json
import torch
from ultralytics.utils.tal import TaskAlignedAssigner
BASE=Path(__file__).resolve().parent

def run():
 d=json.loads((BASE/'native_candidate_geometry_audit.json').read_text(encoding='utf-8'))
 assert hashlib.sha256(Path(d['tal_source_path']).read_bytes()).hexdigest()==d['tal_sha256']
 centers=[]
 for st,n in [(8,80),(16,40),(32,20)]:
  yy,xx=torch.meshgrid(torch.arange(n),torch.arange(n),indexing='ij');centers.append(torch.stack((xx+.5,yy+.5),-1).reshape(-1,2)*st)
 centers=torch.cat(centers).float();assigner=TaskAlignedAssigner(topk=10,num_classes=80,alpha=.5,beta=6,stride=[8,16,32]);count=0
 for model in ['v4','C0','C1']:
  actual=json.loads((BASE/f'fp64_component_gradients/component_gradients_{model}.json').read_text(encoding='utf-8'))
  records={(r['image_id'],r['source_gt_id']):r for r in d['per_gt'] if r['model']==model}
  for b in actual['batches']:
   if not b['GT']:continue
   boxes=torch.tensor([[records[(g['image_id'],g['source_annotation_id'])]['original_box'] for g in b['GT']]],dtype=torch.float32)
   mask=assigner.select_candidates_in_gts(centers,boxes,torch.ones((1,len(b['GT']),1),dtype=torch.bool))
   for i,g in enumerate(b['GT']):
    expected=records[(g['image_id'],g['source_annotation_id'])]['native_expanded_candidate_indices']
    assert torch.where(mask[0,i])[0].tolist()==expected
    assert all(bool(mask[0,i,k]) for k in g['positive_anchor_indices']);count+=len(g['positive_anchor_indices'])
 assert count==7415;print(json.dumps({'status':'PASS_NATIVE_GEOMETRY','positive_slots':count,'model_forwards':0,'assigner_forward_calls':0,'geometry_method_only':True}))
if __name__=='__main__':run()