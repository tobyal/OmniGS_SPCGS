import argparse,copy,json,os,sys,shutil
from pathlib import Path
import numpy as np
from .common import *

def choose(rows,n):return [rows[i] for i in np.linspace(0,len(rows)-1,min(n,len(rows)),dtype=int)]
def base_request(cfg):return {**cfg,'root':str(ROOT),'python':str(ROOT/'.venv/bin/python'),'sam_worker':str(ROOT/'tools/sam2_worker.py'),'geometry_worker':str(ROOT/'tools/geometry_worker.py')}
def prepare(run,cfg):
    run.mkdir(parents=True,exist_ok=True)
    if (run/'config.json').exists():
        assert read(run/'config.json')==cfg,'Existing run has different configuration. Use a new run ID.'
    else:write(run/'config.json',cfg)
    train=choose(cameras('train'),cfg['train_views']);test=choose(cameras('test'),cfg['eval_views'])
    assert not set(v['name'] for v in train)&set(v['name'] for v in test)
    write(run/'data_manifest.json',{'train':train,'test':test,'validation':cameras('val'),'protocol':ref(ROOT/'data/protocol.json'),'quick_subset_uses_original_training_only_point_cloud':True})
    return train,test

def evaluate(folder,views,cfg,model=None):
    import cv2
    from skimage.metrics import structural_similarity
    from .visuals import semantic_preview
    folder=Path(folder);rows=[]
    for v in views:
        p=folder/(v['name']+'_rgb.f32')
        if not p.exists():continue
        pred=np.fromfile(p,dtype='f4').reshape(3,cfg['height'],cfg['width']).transpose(1,2,0)
        gt=cv2.cvtColor(cv2.imread(v['image']),cv2.COLOR_BGR2RGB);gt=cv2.resize(gt,(cfg['width'],cfg['height']),interpolation=cv2.INTER_AREA)/255.
        mse=float(np.mean((pred-gt)**2));ssim=float(structural_similarity(gt,pred,data_range=1,channel_axis=2,gaussian_weights=True,sigma=1.5,use_sample_covariance=False))
        row={'view':v['name'],'split':v['split'],'psnr':float(-10*np.log10(max(mse,1e-12))),'ssim':ssim,'lpips':None}
        logits=folder/(v['name']+'_logits.f16')
        if logits.exists():
            scores=np.fromfile(logits,dtype='f2').astype('f4').reshape(len(cfg['classes']),cfg['semantic_height'],cfg['semantic_width']);probs=np.exp(scores-scores.max(0));probs/=probs.sum(0)
            row['semantic_confidence_proxy']=float(probs.max(0).mean());row['semantic_entropy_proxy']=float(-(probs*np.log(probs+1e-8)).sum(0).mean())
            ids=np.fromfile(folder/(v['name']+'_ids.i32'),dtype='i4').reshape(cfg['semantic_height'],cfg['semantic_width']);row['valid_contributor_fraction']=float((ids>=0).mean())
            labels=scores.argmax(0).astype('u1');labels[ids<0]=255
            from PIL import Image
            Image.fromarray(labels).save(folder/(v['name']+'_labels.png'))
            semantic_preview(folder/(v['name']+'_rgb.png'),labels,cfg['classes'],folder/(v['name']+'_semantic.png'))
        rows.append(row)
    testing=[r for r in rows if r['split']=='test'];training=[r for r in rows if r['split']=='train']
    metrics={'views':rows,'psnr':float(np.mean([r['psnr'] for r in testing or rows])),'ssim':float(np.mean([r['ssim'] for r in testing or rows])),
      'test_psnr':float(np.mean([r['psnr'] for r in testing])) if testing else None,'test_ssim':float(np.mean([r['ssim'] for r in testing])) if testing else None,
      'train_psnr':float(np.mean([r['psnr'] for r in training])) if training else None,'aggregation':'Arithmetic mean over held-out test views only; train reference reported separately',
      'lpips':None,'lpips_reason':'Not computed by the lightweight evaluation entry; PSNR and SSIM use the saved float RGB.',
      'miou':None,'macc':None,'semantic_gt_reason':'No independent industrial class ground truth supplied; CLIP pseudolabels are not ground truth.',
      'resolution':[cfg['width'],cfg['height']],'model':model}
    write(folder/'metrics.json',metrics);return metrics

def baseline(run,cfg,train,test):
    folder=run/STAGES['baseline'];src=ROOT/'baseline/checkpoint.ply'
    with Stage(folder,{'model':ref(src),'provenance':read(ROOT/'baseline/provenance.json'),'protocol':ref(ROOT/'data/protocol.json')},cfg) as stage:
        request={**base_request(cfg),'action':'render','ply':str(src),'output':str(folder/'renders'),'views':train[:1]+test}
        native(request,folder);metrics=evaluate(folder/'renders',train[:1]+test,cfg,str(src));xyz,rgb,_=ply(src)
        assert len(xyz)==647929, f'Unexpected baseline count {len(xyz)}'
        stage.done({'model':str(src),'renders':str(folder/'renders'),'gaussians':len(xyz),'baseline_sha256':sha(src)},metrics)

def m1(run,cfg,train):
    from .visuals import projection_preview
    folder=run/STAGES['m1'];src=DATA/'openMVG/scene.ply'
    failure=None
    with Stage(folder,{'point_cloud':ref(src),'all_training_views':cameras('train'),'used_training_views':train},cfg) as stage:
        xyz,rgb,v=ply(src);np.concatenate([xyz,rgb],1).astype('f4').tofile(folder/'original_points.f32');shutil.copy2(src,folder/'original.ply')
        write(folder/'metrics.json',{'original_count':len(xyz),'enhanced_count':None,'m1_mode':cfg['m1_mode']})
        for cam in train[:2]:projection_preview(xyz,rgb,cam,folder/'preview'/(cam['name']+'_original.png'))
        if cfg['m1_mode']=='skip':
            result={'points':str(folder/'original_points.f32'),'point_count':len(xyz),'source':'original SfM','enhanced_ply':None,'enhanced_points':None}
            stage.done(result,{'original_count':len(xyz),'enhanced_count':None},status='skipped',reason='m1_mode=skip: view generation is explicitly disabled. Original training-only SfM is passed to SGI.')
        else:
            try:
                if cfg['m1_mode']=='wan360':
                    from .m1_wan360 import augment
                elif cfg['m1_mode'] in ['import','online']:
                    from .sfm import augment
                else:
                    raise ValueError('Unknown m1_mode: '+cfg['m1_mode'])
                result,metrics=augment(folder,cfg,train,xyz,rgb);stage.done(result,metrics)
            except Exception as error:
                # Preserve partial statistics and artifacts instead of letting the
                # generic Stage error handler replace them with an empty dict.
                import traceback
                failure=error
                (folder/'error.log').write_text(traceback.format_exc())
                metrics=read(folder/'metrics.json') if (folder/'metrics.json').exists() else {'original_count':len(xyz)}
                if cfg['m1_mode']=='wan360':
                    metrics['completed_video_artifacts']=sum(read(p).get('status')=='completed' for p in (folder/'generation').glob('*/generation.json'))
                stage.done({'points':None,'point_count':0,'enhanced_ply':None,
                            'original_ply':str(folder/'original.ply'),'partial_directory':str(folder)},
                           metrics,status='failed',reason=str(error))
                if cfg['m1_mode']=='wan360':
                    from .m1_wan360 import report as m1_report
                    m1_report(folder,metrics,str(error))
    if failure is not None:
        raise failure
    return read(folder/'outputs.json')

def train_stage(run,cfg,kind,train,test,cache,resume=None):
    from .features import init_semantics
    stage_dir={'s0':run/STAGES['semantic']/'s0','sgi':run/STAGES['sgi'],'c0':run/STAGES['spc']/'c0','c1':run/STAGES['spc']/'c1','c2':run/STAGES['spc']/'c2'}[kind]
    pointinfo=read(run/STAGES['m1']/'outputs.json');request={**base_request(cfg),'action':'train','output':str(stage_dir),'views':cache['views'],'text':cache['text'],'decoder':cache['decoder'],
        'ogr':kind=='sgi','spc2d':kind in ['c1','c2'],'spc3d':kind=='c2','generated_views':cache.get('generated_views',[])}
    center=np.array([v['center'] for v in cameras('train')]);request['extent']=float(np.linalg.norm(center-center.mean(0),axis=1).max()*1.1)
    request['iterations']=cfg[{'s0':'semantic_iterations','sgi':'sgi_iterations'}.get(kind,'spc_iterations')]
    inputs={'cache':ref(Path(cache['text']).parent/'manifest.json'),'initialization':kind,'upstream_m1':pointinfo}
    with Stage(stage_dir,inputs,cfg) as stage:
        if kind=='s0':
            request['ply']=str(ROOT/'baseline/checkpoint.ply');xyz,_,_=ply(request['ply']);request['densify_until']=0
            request['initial_semantic']=init_semantics(xyz,cache,stage_dir/'initial_semantic.f32')
        elif kind=='sgi':
            assert pointinfo.get('status') in ['completed','skipped'] and pointinfo.get('points'),'M1 must provide a valid completed/enabled input before SGI'
            request.update(points=pointinfo['points'],point_count=pointinfo['point_count']);xyz=np.fromfile(request['points'],dtype='f4').reshape(-1,6)[:,:3]
            request['initial_semantic']=init_semantics(xyz,cache,stage_dir/'initial_semantic.f32')
        else:
            request['archive']=str(run/STAGES['sgi']/'final/model.pt');assert Path(request['archive']).exists(),'Run SGI first'
            request['stage_reset']=True
        request['signature']=signature({k:v for k,v in request.items() if k not in ['output','signature','archive']})
        if resume:request.update(archive=str(Path(resume).resolve()),resume=True,stage_reset=False)
        native(request,stage_dir)
        model=stage_dir/'final/model.pt';assert model.exists()
        render_request={**base_request(cfg),'action':'render','archive':str(model),'output':str(stage_dir/'renders'),'views':train[:1]+test}
        native(render_request,stage_dir/'render_job');metrics=evaluate(stage_dir/'renders',train[:1]+test,cfg,str(model))
        loss=[json.loads(x) for x in (stage_dir/'losses.jsonl').read_text().splitlines()]
        metrics.update(first_loss=loss[0],last_loss=loss[-1],valid_isp_pairs=sum(x['isp_valid_pairs'] for x in loss),ogr_removed=sum(x.get('ogr_removed',0) for x in loss))
        xyz,rgb,attr=ply(stage_dir/'final/point_cloud.ply')
        metrics.update(gaussians=len(xyz),scale_quantiles=np.quantile(np.exp(np.stack([attr[f'scale_{i}'] for i in range(3)],1)),[.1,.5,.9]).tolist(),opacity_quantiles=np.quantile(1/(1+np.exp(-attr['opacity'])),[.1,.5,.9]).tolist())
        from .visuals import projection_preview
        for cam in train[:1]+test[:1]:projection_preview(xyz,rgb,cam,stage_dir/'preview'/(cam['name']+'_gaussians.png'))
        ineffective=kind in ['c1','c2'] and metrics['valid_isp_pairs']==0
        stage.done({'model':str(model),'ply':str(stage_dir/'final/point_cloud.ply'),'semantic':str(stage_dir/'final/semantic.f32'),'renders':str(stage_dir/'renders'),'iterations':request['iterations']},metrics,
            status='failed' if ineffective else 'completed',reason='Training ran, but SAM2 produced no valid pair: SPC inter-view/3D optimization was not exercised.' if ineffective else None)
    return stage_dir

def collect_isp(run,cfg):
    target=run/STAGES['isp'];pairs=[]
    for p in (run/STAGES['spc']).glob('*/isp/*/outputs.json'):pairs.append({'snapshot':str(p.parent),**read(p)})
    with Stage(target,{'snapshots':[x['snapshot'] for x in pairs]},cfg) as stage:
        stage.done({'snapshots':pairs},{'events':len(pairs),'valid_pairs':sum(sum(p['valid']) for p in pairs),'attempted_pairs':2*len(pairs),'metric_note':'Sanity-gate pass rate, not matching accuracy.'},status='completed' if pairs else 'skipped',reason=None if pairs else 'No SAM2 event executed')

def report(run,cfg):
    from .comparison import make_report
    collect_isp(run,cfg);make_report(run,cfg)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default='quick_20261023');p.add_argument('--mode',choices=['quick','full'],default='quick');p.add_argument('--config');p.add_argument('--stage',choices=['all','baseline','m1','features','s0','sgi','c0','c1','c2','compare'],default='all');p.add_argument('--resume');a=p.parse_args()
    cfg=default_config(a.mode)
    if a.config:cfg.update(read(a.config))
    if cfg.get('class_config'):cfg['classes']=[c['name_en'] for c in read(ROOT/cfg['class_config'])['classes']]
    run=ROOT/'runs/industrial_spcgs'/a.run;train,test=prepare(run,cfg)
    stages=['baseline','m1','features','s0','sgi','c0','c1','c2','compare'] if a.stage=='all' else [a.stage]
    for name in stages:
        print('STAGE',name,flush=True)
        try:
            if name=='baseline':baseline(run,cfg,train,test)
            elif name=='m1':m1(run,cfg,train)
            elif name=='features':
                from .features import extract
                cache=extract(train,run/STAGES['semantic']/'cache',cfg)
                m1p=run/STAGES['m1']/'outputs.json'
                if m1p.exists() and read(m1p).get('registered_generated_views'):
                    generated=extract(read(m1p)['registered_generated_views'],run/STAGES['semantic']/'generated_cache',cfg)
                    cache['generated_views']=generated['views'];write(run/STAGES['semantic']/'cache/manifest.json',cache)
            elif name=='compare':report(run,cfg)
            else:train_stage(run,cfg,name,train,test,read(run/STAGES['semantic']/'cache/manifest.json'),a.resume)
        except Exception:
            report(run,cfg);raise
    print('RUN',run,flush=True)
if __name__=='__main__':main()
