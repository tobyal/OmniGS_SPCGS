"""Numerical integration checks for optional native semantic validity masks."""
import sys,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from spcgs.common import ROOT,read,write,native
from spcgs.official_teacher import targets

def check(run,gpu=0):
    out=Path(run)/'03_semantic/mask_validation';out.mkdir(parents=True,exist_ok=True)
    cache=read(Path(run)/'03_semantic/cache/manifest.json');text=np.fromfile(cache['text'],dtype='f4').reshape(7,512)
    basis=np.fromfile(cache['decoder'],dtype='f4').reshape(512,32)
    labels=np.array([[0,1,255,2],[3,255,5,6]],dtype='u1');F,S,W=targets(labels,text);features=np.random.default_rng(8).normal(0,.2,(32,2,4)).astype('f4');features.tofile(out/'features.f32')
    base={'gpu':gpu,'seed':42,'action':'semantic_probe','height':2,'width':4,'semantic_height':2,'semantic_width':4,'classes':cache['classes'],
      'text':cache['text'],'decoder':cache['decoder'],'probe_features':str(out/'features.f32')}
    results={};gradients={}
    for case in ['masked','invalid_changed','valid_changed','zero_mask','legacy','ones_mask']:
        folder=out/case;folder.mkdir(exist_ok=True);f,s,w=F.copy(),S.copy(),W.copy()
        if case=='invalid_changed':f[:,W==0]=.7;s[:,W==0]=np.arange(7)[:,None]*3
        if case=='valid_changed':s[:,0,0]=np.roll(s[:,0,0],1)
        if case=='zero_mask':w[:]=0
        if case=='ones_mask':w[:]=1
        f.tofile(folder/'F.f16');s.tofile(folder/'S.f16');w.tofile(folder/'W.u8')
        view={'feature':str(folder/'F.f16'),'feature_height':2,'feature_width':4,'relevancy':str(folder/'S.f16'),'relevancy_height':2,'relevancy_width':4,'relevancy_classes':7,
          'semantic_target':'softmax_relevancy','semantic_mask':str(folder/'W.u8'),'mask_height':2,'mask_width':4}
        if case=='legacy':view.pop('semantic_mask')
        request={**base,'output':str(folder),'views':[view]};native(request,folder)
        results[case]=read(folder/'probe.json');gradients[case]=np.fromfile(folder/'gradient.f32',dtype='f4').reshape(32,2,4)
    assert results['zero_mask']['loss']==0 and results['zero_mask']['gradient_norm']==0
    assert results['masked']==results['invalid_changed']
    assert np.array_equal(gradients['masked'],gradients['invalid_changed'])
    assert np.all(gradients['masked'][:,W==0]==0)
    assert abs(results['masked']['ce']-results['valid_changed']['ce'])>1e-4
    assert results['legacy']==results['ones_mask']
    feature=torch.tensor(features.reshape(32,-1),requires_grad=True);pred=torch.tensor(basis)@feature
    norm=lambda a:torch.nn.functional.normalize(a,dim=0,eps=1e-6)
    target=torch.tensor(F.astype('f4').reshape(512,-1));scores=torch.tensor(S.astype('f4').reshape(7,-1));weight=torch.tensor(W.reshape(-1))
    cosine=((1-(norm(pred)*norm(target)).sum(0))*weight).sum()/weight.sum()
    logits=torch.tensor(text)@norm(pred)*10
    ce=-((torch.softmax(scores,0)*torch.log_softmax(logits,0)).sum(0)*weight).sum()/weight.sum();(cosine+ce).backward()
    assert abs(cosine.item()-results['masked']['cosine'])<1e-5
    assert abs(ce.item()-results['masked']['ce'])<1e-5
    assert np.max(np.abs(feature.grad.numpy().reshape(32,2,4)-gradients['masked']))<1e-5
    write(out/'results.json',{'status':'passed','cases':results,'python_cosine':cosine.item(),'python_ce':ce.item(),'checks':['invalid pixels have zero loss gradient','changing invalid F/S has no effect','changing valid S changes CE','all-zero W produces zero semantic loss and gradient','absent W matches all-ones W','native loss and gradient match independent PyTorch expression']})
    print('MASK CHECKS PASSED',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='sparse27_official_sgi_20261024/opacity_lr_fixed');parser.add_argument('--gpu',type=int,default=0);args=parser.parse_args()
    check(ROOT/'runs/industrial_spcgs'/args.run,args.gpu)
