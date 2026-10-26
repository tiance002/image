"""Source for the executed mask5 + FP64-metric4 checks. No models are created.
The recorded run is in mask_contract_checks.json/fp64_metric_checks.json.
Running this file repeats only these small tensor tests and prints a summary.
"""
from pathlib import Path
import importlib.util,json,sys
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from saferestore.losses.task_detection import IgnoreRegionBCE,ignore_anchor_mask

def run():
 torch.set_num_threads(4);checks=[]
 logits=torch.tensor([[[.8,-.5,.7],[.2,.9,-.4]]],requires_grad=True);targets=torch.tensor([[[.3,0.,1e-12],[0.,.8,0.]]])
 criterion=IgnoreRegionBCE();criterion.mask=torch.tensor([[True,False]])
 base=torch.nn.functional.binary_cross_entropy_with_logits(logits,targets,reduction='none');actual=criterion(logits,targets)
 assert torch.equal(actual[targets>0],base[targets>0]);checks.append('positive_soft_targets_preserved')
 assert actual[0,0,1]==0 and torch.equal(actual[0,1],base[0,1]);checks.append('ignored_negative_zero_unmasked_retained')
 grad=torch.autograd.grad(actual.sum(),logits)[0];assert grad[0,0,1]==0 and torch.all(grad[targets>0]!=0);checks.append('mask_gradient_contract')
 anchors=torch.tensor([[4.,4.],[12.,4.],[20.,4.]]);pred=torch.tensor([[[2.,2.,6.,6.],[8.,2.,16.,6.],[19.,2.,23.,6.]]]);regions=torch.tensor([[3.,3.,5.,5.],[15.,3.,17.,5.]])
 assert torch.equal(ignore_anchor_mask(anchors,pred,regions,torch.tensor([0,0]),1),torch.tensor([[True,True,False]]));checks.append('pixel_space_ignore_geometry')
 x=torch.zeros((1,2,80),requires_grad=True);t=torch.zeros_like(x);t[0,0,5]=.5;w=torch.zeros((1,1,80));w[..., [0,1,2,3,5,6,7]]=1
 g=torch.autograd.grad((torch.nn.functional.binary_cross_entropy_with_logits(x,t,reduction='none')*w).sum(),x)[0]
 assert torch.count_nonzero(g[..., [i for i in range(80) if i not in [0,1,2,3,5,6,7]]])==0;checks.append('73_unsupervised_classes_zero')
 p=BASE/'component_gradient_probe_fp64_metrics.py';spec=importlib.util.spec_from_file_location('recorded_fp64_metrics',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 a=torch.linspace(-1,1,17061219);b=a+torch.sin(torch.arange(a.numel()).float())*1e-6
 for name,x,y in [('long_close',a,b),('long_equal',a,a),('long_opposite',a,-a),('both_zero',torch.zeros(8),torch.zeros(8))]:
  got=m.compare_vectors(x,y);xx=x.numpy().astype(np.float64);yy=y.numpy().astype(np.float64);nx=np.linalg.norm(xx);ny=np.linalg.norm(yy);ref=float(np.dot(xx,yy)/(nx*ny)) if nx and ny else 1.
  assert abs(got['cosine']-ref)<1e-10 and -1<=got['cosine']<=1;checks.append(name)
 print(json.dumps({'status':'PASS','checks':checks,'model_forwards':0,'optimizer_steps':0}))
if __name__=='__main__':run()