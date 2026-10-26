"""Overlap new-run CPU feature/match caching with ongoing Wan360 generation."""
import sys,time,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,cv2
from spcgs.common import ROOT,read,write,ref,signature
from spcgs.ipsgr_protocol import DEFAULT_RUN,check_inputs
from spcgs.m1_wan360 import extract_candidates,geometry_settings
from spcgs.m1_geometry import load_features,match_features

def main():
    run=DEFAULT_RUN;cfg=read(run/'experiment_config.json');m=check_inputs(run);train=m['train'];by={v['name']:v for v in train};geo=geometry_settings(cfg);features={v['name']:load_features(geo['features'],v['name']) for v in train};centers=np.array([v['center'] for v in train]);cache=Path(geo['match_cache']);cache.mkdir(exist_ok=True);worker=ref(ROOT/'spcgs/m1_geometry.py');jobs=read(run/'01_geometry/Wan360/generation_jobs.json')['jobs'];done=set();root=run/'01_geometry/prefetch';root.mkdir(exist_ok=True);cv2.setNumThreads(4);batches=0
    if (root/'status.json').exists():done=set(read(root/'status.json')['videos'])
    while len(done)<80:
        pending=[]
        for job in jobs:
            if job['name'] in done:continue
            p=Path(job['output'])/'generation.json'
            if p.exists():
                v=read(p);assert v['signature']==job['signature'];v['frames']=[f for f in v['frames'] if f['frame_index'] in {0,*cfg['wan360']['candidate_frames']}];pending.append(v)
        if not pending:time.sleep(10);continue
        pending=pending[:8];batches+=1;folder=root/f'batch_{time.time_ns()}';folder.mkdir();candidates=extract_candidates(folder,cfg,pending)
        for c in candidates:
            source=by[c['source_training_view']];near=[train[i] for i in np.argsort(np.linalg.norm(centers-source['center'],axis=1))[:geo['neighbors']]]
            for real in near:
                rf=features[real['name']];key=signature({'real':rf['fingerprint'],'generated':c['features']['fingerprint'],'ratio':geo['ratio'],'mutual':geo['mutual'],'seed':geo['seed'],'opencv':cv2.__version__,'implementation':worker});path=cache/(key+'.npz')
                if not path.exists():
                    cv2.setRNGSeed(geo['seed']);pairs,ratios=match_features(rf,c['features'],geo['ratio'],geo['mutual']);tmp=path.with_suffix(f'.{os.getpid()}.tmp.npz');np.savez_compressed(tmp,pairs=pairs,ratio=ratios);tmp.replace(path)
        done.update(v['name'] for v in pending);write(root/'status.json',{'videos':sorted(done),'completed':len(done),'total':80,'updated':time.time(),'status':'completed' if len(done)==80 else 'running','private_new_run_caches_only':True});print('PREFETCH',len(done),flush=True)

if __name__=='__main__':main()
