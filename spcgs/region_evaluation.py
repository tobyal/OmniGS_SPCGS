"""Compare C0/C1/C2 on the SAME SAM2 masks and SAME camera poses.

All measurements are consistency proxies, never ground-truth semantic accuracy.
"""
from pathlib import Path
import argparse
import numpy as np
from .common import *
from .pipeline import base_request

def probs(x):
    x=x.astype('f4');e=np.exp(x-x.max(0,keepdims=True));return e/e.sum(0,keepdims=True)
def cosine(a,b):return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8))
def evaluate_regions(run,max_events=4,events_manifest=None,gpu=None):
    run=Path(run).resolve();cfg=read(run/'config.json');out=run/'06_comparison/fixed_region_consistency';out.mkdir(parents=True,exist_ok=True)
    if gpu is not None:cfg['gpu']=gpu
    if events_manifest:
        events=[Path(x['event']) for x in read(events_manifest)['events']]
        source_note='Fixed captured-RGB SAM2 masks from an independent evaluation sequence; never optimized. No semantic ground truth.'
    else:
        events=[p.parent for p in sorted((run/'05_spc/c2/isp').glob('*/outputs.json'),key=lambda p:int(p.parent.name)) if any(read(p)['valid'])][-max_events:]
        source_note='Same C2 training snapshot masks and poses for all models; model-derived consistency proxy, not semantic correctness.'
    if not events:
        write(out/'outputs.json',{'status':'skipped','reason':'No valid SAM2 cross-view correspondence; no consistency statistic invented.'});return
    results=[]
    import cv2
    def eroded(mask):
        x=np.pad(np.pad(mask.astype('u1'),((0,0),(2,2)),mode='wrap'),((2,2),(0,0)),mode='constant')
        return cv2.erode(x,np.ones((5,5),'u1'),borderType=cv2.BORDER_CONSTANT,borderValue=0)[2:-2,2:-2]>0
    for model in ['c0','c1','c2']:
        archive=run/'05_spc'/model/'final/model.pt'
        if not archive.exists():continue
        semantic=np.memmap(archive.parent/'semantic.f32',dtype='f4',mode='r').reshape(-1,32)
        for event in events:
            inputs=read(event/'request.json');cams=inputs['cameras'];views=[{**cam,'name':f'frame_{i}'} for i,cam in enumerate(cams)];dest=out/model/event.name
            cached=run/'05_spc'/model/'renders_all'
            if events_manifest and all((cached/(cam['name']+'_features.f16')).exists() for cam in cams):
                render_dir=cached;names=[cam['name'] for cam in cams]
            else:
                render_dir=dest/'renders';names=[f'frame_{i}' for i in range(3)]
                if not all((render_dir/(n+'_features.f16')).exists() for n in names):
                    req={**base_request(cfg),'action':'render','archive':str(archive),'views':views,'output':str(render_dir)};native(req,dest)
            H,W=cfg['semantic_height'],cfg['semantic_width'];masks=np.fromfile(event/'masks.u8',dtype='u1').reshape(3,H,W)>0;features=[];ps=[];ids=[]
            for i in range(3):
                features.append(np.fromfile(render_dir/(names[i]+'_features.f16'),dtype='f2').astype('f4').reshape(32,H,W))
                ps.append(probs(np.fromfile(render_dir/(names[i]+'_logits.f16'),dtype='f2').reshape(len(cfg['classes']),H,W)))
                ids.append(np.fromfile(render_dir/(names[i]+'_ids.i32'),dtype='i4').reshape(H,W))
            valid=read(event/'outputs.json')['valid'];ma=masks[0]&(ids[0]>=0)
            for i in [1,2]:
                mb=masks[i]&(ids[i]>=0)
                row={'model':model,'event':event.name,'pair':i,'sam_basic_valid':valid[i-1],
                    'real_pixels':int(ma.sum()),'pseudo_pixels':int(mb.sum()),
                    'real_coverage':float(ma.sum()/max(masks[0].sum(),1)),'target_coverage':float(mb.sum()/max(masks[i].sum(),1)),
                    'usable':bool(valid[i-1] and ma.sum()>=9 and mb.sum()>=9)}
                if not row['usable']:results.append(row);continue
                fa=features[0][:,ma].mean(1);fb=features[i][:,mb].mean(1);pa=ps[0][:,ma].mean(1);pb=ps[i][:,mb].mean(1)
                ca=int(np.bincount(ps[0].argmax(0)[ma],minlength=len(cfg['classes'])).argmax());cb=int(np.bincount(ps[i].argmax(0)[mb],minlength=len(cfg['classes'])).argmax())
                row.update(feature_cosine=cosine(fa,fb),region_class_distribution_KL=float((pa*(np.log(pa+1e-8)-np.log(pb+1e-8))).sum()),
                    dominant_class_agrees=int(ca==cb),real_class=cfg['classes'][ca],target_class=cfg['classes'][cb],
                    real_entropy=float(-(pa*np.log(pa+1e-8)).sum()),target_entropy=float(-(pb*np.log(pb+1e-8)).sum()))
                ea=eroded(masks[0])&(ids[0]>=0);eb=eroded(masks[i])&(ids[i]>=0)
                row.update(eroded_real_pixels=int(ea.sum()),eroded_target_pixels=int(eb.sum()))
                if ea.any() and eb.any():
                    ga=probs(semantic[ids[0][ea]].T).T;gb=probs(semantic[ids[i][eb]].T).T;q=ga.mean(0).clip(1e-8)
                    row['gaussian_semantic_KL']=float((q*(np.log(q)-np.log(gb.clip(1e-8)))).sum(1).mean())
                results.append(row)
    common=set.intersection(*[{(r['event'],r['pair']) for r in results if r['model']==model and r['usable']} for model in ['c0','c1','c2']])
    common3d=set.intersection(*[{(r['event'],r['pair']) for r in results if r['model']==model and r['usable'] and 'gaussian_semantic_KL' in r} for model in ['c0','c1','c2']])
    summaries={}
    for model in ['c0','c1','c2']:
        rows=[r for r in results if r['model']==model and (r['event'],r['pair']) in common]
        summaries[model]={'common_pairs':len(rows),'common_3d_pairs':len(common3d)}
        for key in ['feature_cosine','region_class_distribution_KL','dominant_class_agrees','real_coverage','target_coverage','real_entropy','target_entropy']:
            values=[r[key] for r in rows if key in r];summaries[model][key]=float(np.mean(values)) if values else None
        values=[r['gaussian_semantic_KL'] for r in rows if (r['event'],r['pair']) in common3d]
        summaries[model]['gaussian_semantic_KL']=float(np.mean(values)) if values else None
    write(out/'metrics.json',{'measurements':results,'common_pair_summary':summaries,'common_pairs':sorted(common),'common_3d_pairs':sorted(common3d),'note':source_note,
        'interpretation':'Consistency proxies may improve even when a shared class is wrong. Report coverage, class distribution and visual review alongside them.'})
    write(out/'inputs.json',{'events':[str(x) for x in events],'models':['c0','c1','c2'],'source_note':source_note});write(out/'config.json',cfg);write(out/'outputs.json',{'status':'completed','samples':len(results)})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--max-events',type=int,default=4);p.add_argument('--events-manifest');p.add_argument('--gpu',type=int);a=p.parse_args();evaluate_regions(a.run,a.max_events,a.events_manifest,a.gpu)
