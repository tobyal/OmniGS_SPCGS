"""Run and inspect M4 independently of any optimizer."""
import argparse,shutil
from pathlib import Path
import numpy as np
from PIL import Image
from .common import *
from .pipeline import base_request

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True);p.add_argument('--view');p.add_argument('--seed',type=int,default=0);p.add_argument('--gpu',type=int,default=0);a=p.parse_args()
    rows=cameras('train');v=next(x for x in rows if x['name']==a.view) if a.view else rows[0];rng=np.random.default_rng(a.seed);out=Path(a.output).resolve();cfg=default_config();cfg['gpu']=a.gpu;cfg['seed']=a.seed
    distances=np.linalg.norm(np.array([r['center'] for r in rows])-v['center'],axis=1);near=np.argsort(distances)[1:3]
    pseudo=[]
    for i,k in enumerate(near):
        t=rng.uniform(.1,.3);center=(1-t)*np.array(v['center'])+t*np.array(rows[k]['center'])+rng.normal(0,.02*distances[near[0]],3)
        pseudo.append({**v,'name':f'pseudo_{i}','center':center.tolist()})
    with Stage(out,{'model':ref(a.model),'real':v,'pseudo':pseudo},cfg) as stage:
        req={**base_request(cfg),'action':'render','archive':str(Path(a.model).resolve()),'output':str(out/'renders'),'views':[v]+pseudo};native(req,out)
        Image.open(v['image']).resize((cfg['width'],cfg['height']),Image.Resampling.BOX).save(out/'real.png')
        sam={**cfg,'root':str(ROOT),'height':cfg['semantic_height'],'width':cfg['semantic_width'],'output':str(out/'correspondence'),'images':[str(out/'real.png')]+[str(out/'renders'/(x['name']+'_rgb.png')) for x in pseudo],'cameras':[v]+pseudo}
        Path(sam['output']).mkdir(exist_ok=True);write(out/'sam_request.json',sam);env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(a.gpu)
        run_command([ROOT/'.venv/bin/python',ROOT/'tools/sam2_worker.py',out/'sam_request.json'],out/'sam.log',env)
        result=read(Path(sam['output'])/'outputs.json');stage.done({'renders':req['output'],'correspondence':sam['output']},{'valid_pairs':sum(result['valid']),'attempted_pairs':2,'validity_is_not_accuracy':True})
if __name__=='__main__':main()
