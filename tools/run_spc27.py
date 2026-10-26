"""Second stage only: immutable sparse SGI + official teacher, no M1/M2 work."""
import sys,argparse,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spcgs.common import ROOT,read,write,ref,sha,signature,native
from spcgs.pipeline import base_request,prepare,train_stage

SOURCE=ROOT/'runs/industrial_spcgs/sparse27_official_sgi_20261024/opacity_lr_fixed'
RUN=ROOT/'runs/industrial_spcgs/spc27_hieral_20261024'

def setup(run):
    run=Path(run).resolve();run.mkdir(parents=True,exist_ok=True)
    archive=SOURCE/'02_sgi_gaussians/final/model.pt';audit=read(SOURCE/'completion_audit.json')
    assert audit['recommended_for_spc'] and sha(archive)==audit['models']['sgi']['archive']['sha256']
    protocol=read(SOURCE/'00_protocol/manifest.json');teacher=read(SOURCE/'03_semantic/cache/manifest.json')
    assert len(protocol['train'])==27 and len(teacher['classes'])==7
    assert {v['name'] for v in teacher['views']}=={v['name'] for v in protocol['train']}
    cfg={**read(SOURCE/'sgi_config.json'),'sgi_archive':str(archive),
         'sam_config':'configs/sam2/sam2_hiera_l.yaml','sam_checkpoint':str(ROOT/'weights/sam2_hiera_large.pt'),
         'require_sam2_hiera_large':True,'sam_reuse_model':True,'spc_iterations':10000,
         'isp_every':100,'spc_ramp':4000,'contrast_samples':512,'generated_semantic_loss':False}
    for name,target in [('00_protocol',SOURCE/'00_protocol'),('01_sgi_sfm',SOURCE/'01_sgi_sfm'),('02_sgi_gaussians',SOURCE/'02_sgi_gaussians')]:
        link=run/name
        if not link.exists():link.symlink_to(target,target_is_directory=True)
        assert link.resolve()==target.resolve()
    (run/'03_semantic').mkdir(exist_ok=True);link=run/'03_semantic/cache'
    if not link.exists():link.symlink_to(SOURCE/'03_semantic/cache',target_is_directory=True)
    assert link.resolve()==(SOURCE/'03_semantic/cache').resolve()
    if not (run/'upstream.json').exists():write(run/'upstream.json',{'sgi':ref(archive),'teacher':ref(SOURCE/'03_semantic/cache/manifest.json'),'protocol':ref(SOURCE/'00_protocol/manifest.json'),'scope':'Second stage only; original SGI and teacher immutable'})
    train,test=prepare(run,cfg)
    return run,cfg,protocol,teacher,train,test

def initialize(run,cfg,cache,gpu):
    folder=run/'03_stage_initialization'
    if (folder/'final/model.pt').exists():return folder
    request={**base_request(cfg),'gpu':gpu,'action':'train','archive':cfg['sgi_archive'],'stage_reset':True,
        'output':str(folder),'iterations':0,'views':cache['views'],'text':cache['text'],'decoder':cache['decoder'],
        'generated_views':[],'semantic_supervision':True,'spc2d':False,'spc3d':False,'ogr':False}
    request['signature']=signature(request);native(request,folder)
    return folder

def render_model(run,cfg,protocol,cache,model,gpu,initial=False):
    from spcgs.sparse27_evaluate import evaluate_renders,representatives
    folder=run/'05_spc'/model if model!='initial' else run/'03_stage_initialization'
    views=representatives(protocol) if initial or model=='initial' else [*protocol['train'],*[{**v,'split':'unseen_train29'} for v in protocol['unseen_train29']],*protocol['validation'],*protocol['test']]
    target=folder/('renders_initial' if initial or model=='initial' else 'renders_all')
    if not all((target/(v['name']+'_rgb.f32')).exists() for v in views):
        native({**base_request(cfg),'gpu':gpu,'action':'render','archive':str(folder/('initial' if initial else 'final')/'model.pt'),'views':views,'output':str(target)},folder/('render_initial_job' if initial else 'render_all_job'))
    print(evaluate_renders(target,views,cache)['groups'],flush=True)

def main():
    if '--legacy27' not in sys.argv:
        from spcgs.ipsgr_compat import main as current_main
        return current_main('cross_view')
    sys.argv.remove('--legacy27')
    p=argparse.ArgumentParser();p.add_argument('--run',default=str(RUN));p.add_argument('--stage',choices=['prepare','initialize','isp','short','c0','c1','c2','render'],required=True)
    p.add_argument('--gpu',type=int,default=0);p.add_argument('--model',default='c2',choices=['initial','c0','c1','c2']);p.add_argument('--view-index',type=int,default=1);p.add_argument('--seed',type=int,default=42);p.add_argument('--initial',action='store_true');a=p.parse_args()
    run,cfg,protocol,cache,train,test=setup(a.run);cfg={**cfg,'gpu':a.gpu}
    if a.stage=='prepare':print('PREPARED',run)
    elif a.stage=='initialize':initialize(run,cfg,cache,a.gpu);render_model(run,cfg,protocol,cache,'initial',a.gpu)
    elif a.stage=='isp':
        from spcgs.isp import run_isp
        cam=next(v for v in protocol['train'] if v['original_train_index']==a.view_index)
        initialize(run,cfg,cache,a.gpu)
        run_isp(run/'03_stage_initialization/final/model.pt',run/'04_isp/preflight'/f'view{a.view_index:02d}_seed{a.seed}',train,cfg,cam['name'],a.seed,a.gpu)
    elif a.stage=='short':
        short=run/'short';short.mkdir(exist_ok=True)
        for name in ['01_sgi_sfm','03_semantic']:
            if not (short/name).exists():(short/name).symlink_to(run/name,target_is_directory=True)
        local={**cfg,'spc_iterations':200,'save_every':100,'isp_every':20,'densify_until':0}
        train_stage(short,local,'c2',train,test[:1],cache)
    elif a.stage in ['c0','c1','c2']:train_stage(run,cfg,a.stage,train,test,cache)
    elif a.stage=='render':render_model(run,cfg,protocol,cache,a.model,a.gpu,a.initial)

if __name__=='__main__':main()
