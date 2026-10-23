"""M2 entry: inspectable teacher extraction and existing native training adapter."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spcgs.common import ROOT,read,write,cameras,default_config,STAGES

def config(path):
    cfg=default_config('full');cfg.update(read(ROOT/path))
    cfg['classes']=[c['name_en'] for c in read(ROOT/cfg['class_config'])['classes']]
    return cfg

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['features','legacy']);p.add_argument('--config',default='configs/m2_industrial_multiscale.json');p.add_argument('--indices',type=int,nargs='+');p.add_argument('--output');p.add_argument('--generated',action='store_true');a=p.parse_args()
    cfg=config(a.config);views=cameras('train')
    if a.generated:
        if a.indices:raise ValueError('Training indices do not apply to generated views')
        views=read(ROOT/cfg['m1_source'])['registered_generated_views']
        if not views:raise ValueError('M1 has no registered generated views')
    if a.indices:views=[views[i-1] for i in a.indices]
    if a.action=='legacy':
        cfg['feature_backend']='local_crops';cfg['classes']=default_config()['classes']
    folder=ROOT/(a.output or ('runs/industrial_semantics_20261023/01_clip/'+a.action+('_generated' if a.generated else '_pilot' if a.indices else '_all')))
    from spcgs.features import extract
    extract(views,folder,cfg)
if __name__=='__main__':main()
