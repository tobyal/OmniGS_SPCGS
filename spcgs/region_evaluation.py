"""Compare C0/C1/C2 on the SAME SAM2 masks and SAME pseudo camera poses.

All measurements are consistency proxies, never ground-truth semantic accuracy.
"""
from pathlib import Path
import argparse,csv
import numpy as np
from .common import *
from .pipeline import base_request

def probs(x):
    x=x.astype('f4');e=np.exp(x-x.max(0,keepdims=True));return e/e.sum(0,keepdims=True)
def cosine(a,b):return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8))
def evaluate_regions(run,max_events=4):
    run=Path(run).resolve();cfg=read(run/'config.json');out=run/'06_comparison/fixed_region_consistency';out.mkdir(parents=True,exist_ok=True)
    events=[p.parent for p in sorted((run/'05_spc/c2/isp').glob('*/outputs.json'),key=lambda p:int(p.parent.name)) if any(read(p)['valid'])][-max_events:]
    if not events:
        write(out/'outputs.json',{'status':'skipped','reason':'No valid SAM2 cross-view correspondence; no consistency statistic invented.'});return
    results=[]
    for model in ['c0','c1','c2']:
        archive=run/'05_spc'/model/'final/model.pt'
        if not archive.exists():continue
        for event in events:
            inputs=read(event/'request.json');cams=inputs['cameras'];views=[{**cam,'name':f'frame_{i}'} for i,cam in enumerate(cams)];dest=out/model/event.name
            req={**base_request(cfg),'action':'render','archive':str(archive),'views':views,'output':str(dest/'renders')};native(req,dest)
            H,W=cfg['semantic_height'],cfg['semantic_width'];masks=np.fromfile(event/'masks.u8',dtype='u1').reshape(3,H,W)>0;features=[];ps=[];ids=[]
            for i in range(3):
                features.append(np.fromfile(dest/'renders'/f'frame_{i}_features.f16',dtype='f2').astype('f4').reshape(32,H,W))
                ps.append(probs(np.fromfile(dest/'renders'/f'frame_{i}_logits.f16',dtype='f2').reshape(len(cfg['classes']),H,W)))
                ids.append(np.fromfile(dest/'renders'/f'frame_{i}_ids.i32',dtype='i4').reshape(H,W))
            valid=read(event/'outputs.json')['valid'];ma=masks[0]&(ids[0]>=0)
            for i in [1,2]:
                mb=masks[i]&(ids[i]>=0)
                if not valid[i-1] or ma.sum()<9 or mb.sum()<9:continue
                fa=features[0][:,ma].mean(1);fb=features[i][:,mb].mean(1);pa=ps[0][:,ma].mean(1);pb=ps[i][:,mb].mean(1)
                results.append({'model':model,'event':event.name,'pair':i,'feature_cosine':cosine(fa,fb),'region_class_distribution_KL':float((pa*(np.log(pa+1e-8)-np.log(pb+1e-8))).sum()),'dominant_class_agrees':int(pa.argmax()==pb.argmax()),'real_pixels':int(ma.sum()),'pseudo_pixels':int(mb.sum())})
    write(out/'metrics.json',{'measurements':results,'note':'Same C2 snapshot masks and poses for all models; model-derived consistency proxy, not semantic correctness.'})
    if results:
        with (out/'summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    write(out/'inputs.json',{'events':[str(x) for x in events],'models':['c0','c1','c2']});write(out/'config.json',cfg);write(out/'outputs.json',{'status':'completed','samples':len(results)})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--max-events',type=int,default=4);a=p.parse_args();evaluate_regions(a.run,a.max_events)
