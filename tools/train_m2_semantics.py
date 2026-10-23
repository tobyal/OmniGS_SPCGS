"""Train only M2/S0 or SGI with an existing F/S/T manifest and completed M1."""
import argparse,sys,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spcgs.common import ROOT,STAGES,read,write,ref,cameras,native
from spcgs.pipeline import train_stage,base_request
from tools.m2_semantics import config

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--run',required=True)
    p.add_argument('--config',default='configs/m2_industrial_multiscale.json');p.add_argument('--kind',choices=['s0','sgi'],default='sgi')
    p.add_argument('--iterations',type=int);p.add_argument('--rgb-width',type=int);p.add_argument('--resume');p.add_argument('--no-test',action='store_true');p.add_argument('--generated-cache');a=p.parse_args()
    cfg=config(a.config);cache=read(ROOT/a.cache);run=ROOT/a.run
    if cache['classes']!=cfg['classes']:raise ValueError('Cache and configured text row order differ')
    if a.iterations:cfg['semantic_iterations' if a.kind=='s0' else 'sgi_iterations']=a.iterations
    if a.rgb_width:cfg.update(width=a.rgb_width,height=a.rgb_width//2)
    cfg.update(train_views=len(cache['views']),eval_views=0 if a.no_test else 16)
    run.mkdir(parents=True,exist_ok=True)
    if (run/'config.json').exists() and read(run/'config.json')!=cfg:raise ValueError('Use a new run directory for a changed configuration')
    if (run/STAGES['sgi' if a.kind=='sgi' else 'semantic']/('final/model.pt' if a.kind=='sgi' else 's0/final/model.pt')).exists() and not a.resume:raise ValueError('Completed run already exists')
    write(run/'config.json',cfg);write(run/'data_manifest.json',{'train':cache['views'],'test':[] if a.no_test else cameras('test'),'cache':ref(ROOT/a.cache)})
    cache={**cache,'source_manifest':ref(ROOT/a.cache)}
    if a.generated_cache:
        generated=read(ROOT/a.generated_cache)
        if generated['classes']!=cache['classes']:raise ValueError('Generated and real cache vocabularies differ')
        cache.update(generated_views=generated['views'],generated_source_manifest=ref(ROOT/a.generated_cache))
    local=run/STAGES['semantic']/'cache';local.mkdir(parents=True,exist_ok=True)
    for key in ['text','decoder']:
        shutil.copy2(cache[key],local/(key+'.f32'));cache[key]=str(local/(key+'.f32'))
    write(local/'manifest.json',cache)
    m1=read(ROOT/cfg['m1_source']);assert m1['status']=='completed' and Path(m1['points']).exists()
    write(run/STAGES['m1']/'outputs.json',{**m1,'reused_from':ref(ROOT/cfg['m1_source']),'note':'Reuse completed M1; no view generation or SfM execution in this run.'})
    stage=train_stage(run,cfg,a.kind,cache['views'],[] if a.no_test else cameras('test'),cache,a.resume)
    wanted={cameras('train')[i-1]['name'] for i in [1,23,47,52]};views=[v for v in cache['views'] if v['name'] in wanted]
    for state in ['initial','final']:
        native({**base_request(cfg),'action':'render','archive':str(stage/state/'model.pt'),'output':str(stage/'validation'/state),'views':views},stage/'validation'/(state+'_job'))
    from tools.compare_m2_student import compare
    compare(stage,cache,cfg,views)
    print(stage,flush=True)
if __name__=='__main__':main()
