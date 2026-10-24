"""Run and inspect M4 independently of any optimizer."""
import argparse,shutil
from pathlib import Path
import numpy as np
from PIL import Image
from .common import *
from .pipeline import base_request

def run_isp(model,output,rows,cfg,view=None,seed=0,gpu=0):
    v=next(x for x in rows if x['name']==view) if view else rows[0];rng=np.random.default_rng(seed);out=Path(output).resolve();cfg={**cfg,'gpu':gpu,'seed':seed}
    distances=np.linalg.norm(np.array([r['center'] for r in rows])-v['center'],axis=1);near=[k for k in np.argsort(distances) if distances[k]>1e-5][:2]
    assert len(near)==2,'Need three distinct training centers'
    pseudo=[]
    for i,k in enumerate(near):
        t=rng.uniform(.1,.3);center=(1-t)*np.array(v['center'])+t*np.array(rows[k]['center'])+rng.normal(0,cfg.get('pose_noise_fraction',.02)*distances[near[0]],3)
        pseudo.append({**v,'name':f'pseudo_{i}','center':center.tolist()})
    with Stage(out,{'model':ref(model),'real':v,'pseudo':pseudo,'allowed_training_ids':[r['name'] for r in rows]},cfg) as stage:
        req={**base_request(cfg),'action':'render','archive':str(Path(model).resolve()),'output':str(out/'renders'),'views':[v]+pseudo};native(req,out)
        Image.open(v['image']).resize((cfg['width'],cfg['height']),Image.Resampling.BOX).save(out/'real.png')
        sam={**cfg,'root':str(ROOT),'height':cfg['semantic_height'],'width':cfg['semantic_width'],'output':str(out/'correspondence'),'images':[str(out/'real.png')]+[str(out/'renders'/(x['name']+'_rgb.png')) for x in pseudo],'cameras':[v]+pseudo}
        Path(sam['output']).mkdir(exist_ok=True);write(out/'sam_request.json',sam);env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
        run_command([ROOT/'.venv/bin/python',ROOT/'tools/sam2_worker.py',out/'sam_request.json'],out/'sam.log',env)
        result=read(Path(sam['output'])/'outputs.json');stage.done({'renders':req['output'],'correspondence':sam['output']},{'valid_pairs':sum(result['valid']),'attempted_pairs':2,'validity_is_not_accuracy':True})
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True);p.add_argument('--view');p.add_argument('--seed',type=int,default=0);p.add_argument('--gpu',type=int,default=0);p.add_argument('--config',required=True);a=p.parse_args()
    cfg={**default_config(),**read(a.config)}
    assert cfg.get('train_names'),'Independent ISP requires explicit training image IDs in --config'
    available={v['name']:v for v in cameras('train')}
    rows=[available[n] for n in cfg['train_names']]
    assert len({v['name'] for v in rows})==len(rows)
    run_isp(a.model,a.output,rows,cfg,a.view,a.seed,a.gpu)
if __name__=='__main__':main()
