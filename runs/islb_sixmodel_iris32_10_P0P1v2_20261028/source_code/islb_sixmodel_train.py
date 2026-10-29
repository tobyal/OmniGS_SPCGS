"""Six independent native OmniGS fits, strict resume and GPU task scheduling."""
from pathlib import Path
import argparse,os,sys,time,json,subprocess,traceback,collections,shutil
import numpy as np
from .common import ROOT,read,write,ref,signature,native
from .islb_sixmodel_data import RUN,MODELS

def request(name,gpu,folder,steps=10000,source=None):
    cfg=read(RUN/'00_protocol/common_config.json');m=read(RUN/'00_protocol/dataset_manifest.json');teacher=read(RUN/'02_teacher/manifest.json')
    semantic=name!='B-RGB';layout=name in ['L0','L1','L2'];stage2=name=='ISLB-Only';cloud='P0' if name in ['B-RGB','B-SEM','L0'] else 'P1'
    centers=np.array([v['center'] for v in m['train']]);folder.mkdir(parents=True,exist_ok=True)
    req={**cfg,'action':'train','gpu':gpu,'output':str(folder),'iterations':steps,'save_every':1000 if steps==10000 else 10,
         'extent':float(np.linalg.norm(centers-centers.mean(0),axis=1).max()*1.1),'views':teacher['views'] if semantic else m['train'],
         'semantic_supervision':semantic,'ogr':layout,'local_weight':1. if layout else 0.,'training_stage':'ISLB' if layout else name,
         'stage_reset':stage2,'anchor_weight':.05 if name=='L2' else 0.,'anchor_views':[],'generated_views':[],
         'text':teacher['text'],'decoder':teacher['decoder'],'root':str(ROOT),'sam_worker':str(ROOT/'tools/sam2_worker.py')}
    inputs={'protocol':ref(RUN/'00_protocol/dataset_manifest.json'),'teacher':ref(RUN/'02_teacher/manifest.json') if semantic else None,
        'native_binary':ref(cfg['native_binary']),'native_build':ref(RUN/'native/build_audit.json'),'model':name,'data_leakage':'No test supervision; no old Gaussian archives'}
    if name=='L2':
        anchor=read(RUN/'03_anchors/manifest.json');req.update(anchor_views=anchor['views'],anchor_class_scores=anchor['class_scores']);inputs['anchors']=ref(RUN/'03_anchors/manifest.json')
    if stage2:
        source=source or RUN/'05_models/L2/final/model.pt';assert source.is_file()
        if steps==10000:
            assert read(source.parent/'model.json')['step']==10000 and read(RUN/'05_models/L2/outputs.json')['status']=='completed'
            assert read(RUN/'05_models/L2/checkpoint_validation.json')['status']=='passed'
        req['archive']=str(source);inputs['layout_checkpoint']=ref(source)
    else:
        pc=RUN/'01_initialization'/(cloud+'_v2.f32');sem=RUN/'01_initialization'/(cloud+'_semantic.f32')
        if not semantic:
            sem=folder/'unused_semantic.f32';np.zeros((2781,32),'f4').tofile(sem)
            np.zeros((8,512),'f4').tofile(folder/'unused_text.f32');np.zeros((512,32),'f4').tofile(folder/'unused_decoder.f32');req.update(text=str(folder/'unused_text.f32'),decoder=str(folder/'unused_decoder.f32'))
        req.update(points=str(pc),point_count=pc.stat().st_size//24,initial_semantic=str(sem));inputs.update(point_cloud=ref(pc),initial_semantic=ref(sem))
    signed={k:v for k,v in req.items() if k not in ['output','gpu','stop_after_step']};req['signature']=signature({'request':signed,'inputs':inputs})
    return req,inputs

def checkpoint(folder,semantic=True,expected_step=10000):
    import torch
    torch.set_num_threads(4);state=torch.jit.load(str(folder/'final/model.pt'),map_location='cpu');meta=read(folder/'final/model.json')
    assert int(state.step)==meta['step']==expected_step and len(state.xyz)==meta['gaussians']
    assert state.xyz.shape[1]==3 and state.semantic.shape==(len(state.xyz),32)
    assert state.text.shape==(8,512) and state.classifier.shape==(8,8) and state.decoder.shape==(512,32)
    assert all(torch.isfinite(getattr(state,k)).all() for k in ['xyz','dc','rest','opacity','scale','rotation','semantic','decoder','bias','classifier','cbias','text'])
    if semantic:assert state.semantic.abs().max()>0
    result={'status':'passed','step':int(state.step),'gaussians':len(state.xyz),'finite_all_parameters':True,'classes':8,'semantic_dimension':32,'checkpoint':ref(folder/'final/model.pt')}
    write(folder/'checkpoint_validation.json',result);return result

def reset_check(source,folder):
    import torch
    from scipy.spatial import cKDTree
    torch.set_num_threads(4);s=torch.jit.load(str(source),map_location='cpu');a=torch.jit.load(str(folder/'initial/model.pt'),map_location='cpu')
    keys=['xyz','dc','semantic','decoder','bias','classifier','cbias','projection','text'];errors={k:float((getattr(a,k)-getattr(s,k)).abs().max()) for k in keys};assert all(v==0 for v in errors.values())
    assert a.rest.abs().max()==0 and int(a.sh)==int(a.step)==0 and torch.allclose(a.opacity.sigmoid(),torch.full_like(a.opacity,.1),atol=1e-7)
    rot=torch.zeros_like(a.rotation);rot[:,0]=1;assert torch.equal(rot,a.rotation)
    assert all(getattr(a,k).abs().max()==0 for k in ['radii','gradient','denom','exist','added','pruned'])
    d=cKDTree(a.xyz.numpy()).query(a.xyz.numpy(),k=4)[0][:,1:];expected=np.log(np.sqrt(np.maximum((d*d).mean(1),1e-7)))
    err=float(np.max(np.abs(a.scale.numpy()-expected[:,None])));assert err<.003,err
    audit=read(folder/'stage_reset_audit.json');assert audit['optimizer_state_entries']==audit['head_optimizer_state_entries']==0
    result={'status':'passed','source':ref(source),'inherited_max_error':errors,'reset_scale_log_error':err,'optimizer_states_empty':True,'reset_rotation_opacity_SH_and_densification':True}
    write(folder/'initialization_validation.json',result);return result

def validate_log(folder,name,steps):
    rows=[json.loads(s) for s in (folder/'losses.jsonl').read_text().splitlines()];assert [r['step'] for r in rows]==list(range(1,steps+1))
    names={v['name'] for v in read(RUN/'00_protocol/dataset_manifest.json')['train']};assert {r['view'] for r in rows}<=names
    assert all(np.isfinite(r['total']) and r['inter']==r['intra']==r['spc3d']==r['generated_semantic']==0 for r in rows)
    if name=='B-RGB':assert all(r['clip']==r['local3d']==r['semantic_grad_norm']==0 for r in rows)
    else:assert max(r['semantic_grad_norm'] for r in rows)>0 and max(r['decoder_grad_norm'] for r in rows)>0
    if name in ['B-RGB','B-SEM','ISLB-Only']:assert all(r['local3d']==0 and 'ogr_removed' not in r for r in rows)
    if name=='L2':
        diag=[r for r in rows if 'anchor_semantic_grad_norm' in r]
        assert diag and all(r['anchor_semantic_grad_norm']>0 and r['anchor_geometry_gradient_sum']==0 and r['anchor_ce_only'] and r['anchor_semantic_only'] for r in diag)
        assert all(abs(r['anchor_weighted_loss']-.05*r['anchor_ce'])<1e-5 for r in rows)
    else:assert all(r['anchor_weighted_loss']==0 for r in rows)
    local=1 if name in ['L0','L1','L2'] else 0
    assert max(abs(r['total']-r['rgb']-r['semantic_loss']-local*r['local3d']-r['anchor_weighted_loss']) for r in rows)<2e-5
    audit={'status':'passed','iterations':steps,'train_only':True,'ICSC_events':0,'generated_RGB_or_dense_semantic_loss':False,
         'gaussians':rows[-1]['gaussians'],'anchor_gradient_events':sum('anchor_semantic_grad_norm' in r for r in rows),'OGR_events':[{'step':r['step'],'removed':r['ogr_removed']} for r in rows if 'ogr_removed' in r]}
    write(folder/'training_audit.json',audit);return rows

def fit(name,gpu):
    assert read(RUN/'04_preflight/results.json')['status']=='passed'
    folder=RUN/'05_models'/name;req,inputs=request(name,gpu,folder);key=req['signature'];output=folder/'outputs.json'
    if (folder/'inputs.json').exists():assert read(folder/'inputs.json')['signature']==key,'Model config changed: refusing resume'
    write(folder/'inputs.json',{**inputs,'signature':key});write(folder/'fixed_request.json',{**req,'gpu_assignment_excluded_from_signature':True})
    if output.exists() and read(output)['status']=='completed':checkpoint(folder,name!='B-RGB');return
    previous=[]
    for p in sorted(folder.glob('checkpoint_*/model.pt'),key=lambda p:int(p.parent.name.split('_')[-1]),reverse=True):
        if (p.parent/'model.json').exists():previous.append(p);break
    if previous:
        saved=read(previous[0].parent/'model.json')['step'];assert int(previous[0].parent.name.split('_')[-1])==saved
        req.update(archive=str(previous[0]),resume=True,stage_reset=False)
        log=folder/'losses.jsonl'
        if log.exists():
            shutil.copy2(log,folder/f'losses_before_resume_{time.time_ns()}.jsonl');valid=[]
            for line in log.read_text().splitlines():
                try:r=json.loads(line)
                except json.JSONDecodeError:continue
                if r['step']<=saved:valid.append(line)
            log.write_text('\n'.join(valid)+'\n')
    attempts=folder/'attempts';attempts.mkdir(exist_ok=True);attempt=attempts/f'{time.time_ns()}';started=time.time()
    write(output,{'status':'running','signature':key,'GPU':gpu,'started':started,'resumed_from':str(previous[0]) if previous else None,'attempt':str(attempt)})
    try:
        native(req,attempt)
        val=checkpoint(folder,name!='B-RGB');rows=validate_log(folder,name,10000)
        if name=='ISLB-Only':reset_check(RUN/'05_models/L2/final/model.pt',folder)
        if name!='B-RGB':
            import torch
            state=torch.jit.load(str(folder/'final/model.pt'),map_location='cpu');expected=np.fromfile(read(RUN/'02_teacher/manifest.json')['text'],'f4').reshape(8,512)
            assert np.array_equal(state.text.numpy(),expected),'Frozen CLIP prototypes changed'
        write(output,{'status':'completed','signature':key,'model':ref(folder/'final/model.pt'),'iterations':10000,'gaussians':val['gaussians'],'GPU':gpu,'seconds_this_invocation':time.time()-started,'finished':time.time(),'resumed':bool(previous)})
    except Exception as e:
        failure=folder/f'failure_{time.time_ns()}.log';failure.write_text(traceback.format_exc());write(output,{'status':'failed','signature':key,'error':str(e),'failure_log':str(failure),'GPU':gpu});raise
    print('TRAIN_COMPLETE',name,val['gaussians'],flush=True)

def preflight(gpu):
    out=RUN/'04_preflight';assert not (out/'results.json').exists()
    m=read(RUN/'00_protocol/dataset_manifest.json');assert len(m['train'])==32 and len(m['test'])==10
    # All five fresh model configurations exercise their actual objective;
    # preflight artifacts never initialize a formal fit.
    for name in ['B-RGB','B-SEM','L0','L1','L2']:
        folder=out/name;req,_=request(name,gpu,folder,steps=20)
        req.update(densify_from=10,densify_every=10,densify_until=20,ogr_every=20,opacity_reset_every=0)
        native(req,folder/'execution');checkpoint(folder,name!='B-RGB',20);validate_log(folder,name,20)
    teacher=read(RUN/'02_teacher/manifest.json');anchors=read(RUN/'03_anchors/manifest.json')
    # One actual anchor-only Adam update: semantics change, geometry/RGB do not.
    req,_=request('L2',gpu,out/'anchor_update',steps=20)
    req.update(action='anchor_update_probe',anchor_views=sorted(anchors['views'],key=lambda r:r['anchor_count'],reverse=True)[:1])
    native(req,out/'anchor_update/execution');av=read(out/'anchor_update/anchor_update_validation.json')
    assert av['geometry_RGB_max_change']==0 and av['semantic_max_change']>0 and av['decoder_max_change']>0 and av['text_unchanged']
    # Validate reset mechanics on this run's diagnostic layout, then again on
    # the real L2 checkpoint before accepting +ISLB training.
    src=out/'L2/final/model.pt';req,_=request('ISLB-Only',gpu,out/'stage_reset',steps=20,source=src);req['iterations']=0
    native(req,out/'stage_reset/execution');reset_check(src,out/'stage_reset')
    # Same total-step schedule/signature, optimizer/RNG checkpoint restored.
    folder=out/'resume';req,_=request('B-SEM',gpu,folder,steps=20);req.update(densify_until=0,opacity_reset_every=0,stop_after_step=10)
    native(req,folder/'first');assert not (folder/'final/model.pt').exists()
    req.pop('stop_after_step');req.update(archive=str(folder/'checkpoint_10/model.pt'),resume=True,stage_reset=False)
    native(req,folder/'resumed');checkpoint(folder,True,20);validate_log(folder,'B-SEM',20)
    # Dense objective and sparse CE use the same eight-class CLIP head.
    import torch
    l1=torch.jit.load(str(out/'L1/initial/model.pt'),map_location='cpu');l2=torch.jit.load(str(out/'L2/initial/model.pt'),map_location='cpu')
    equal={k:torch.equal(getattr(l1,k),getattr(l2,k)) for k in ['xyz','dc','rest','scale','rotation','opacity','semantic','decoder','bias','classifier','cbias','text']};assert all(equal.values())
    b=torch.jit.load(str(out/'B-SEM/initial/model.pt'),map_location='cpu');l0=torch.jit.load(str(out/'L0/initial/model.pt'),map_location='cpu');assert torch.equal(b.xyz,l0.xyz) and torch.equal(b.semantic,l0.semantic)
    result={'status':'passed','five_objectives_and_densification':'passed','OGR_layouts':'passed','anchor_actual_optimizer_update':av,
       'L1_L2_initial_parameters_equal':equal,'BSEM_L0_same_P0_initialization':True,'stage_reset':read(out/'stage_reset/initialization_validation.json'),'same_config_resume_to_20':'passed',
       'formal_models_reuse_preflight_checkpoints':False,'teacher_signature':teacher['signature']}
    write(out/'results.json',result);print('PREFLIGHT_PASSED',flush=True)

def gpu_inventory():
    s=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    return [{'id':int(a),'total_MB':int(b),'used_MB':int(c),'free_MB':int(b)-int(c),'utilization':int(d)} for a,b,c,d in (line.split(', ') for line in s.strip().splitlines())]

def schedule():
    assert read(RUN/'04_preflight/results.json')['status']=='passed'
    directory=RUN/'scheduler';lock=directory/'scheduler.lock'
    import fcntl
    lf=lock.open('w');fcntl.flock(lf,fcntl.LOCK_EX|fcntl.LOCK_NB);lf.write(str(os.getpid()));lf.flush()
    active={};attempts=collections.Counter();order=['L2','L1','B-SEM','B-RGB','L0','ISLB-Only'];done=set()
    while len(done)<6:
        for name,job in list(active.items()):
            if job['process'].poll() is not None:
                code=job['process'].returncode;job['log'].close();del active[name]
                if code==0:done.add(name)
                else:
                    write(directory/f'{name}_attempt{attempts[name]}_exit.json',{'returncode':code,'GPU':job['gpu'],'time':time.time()})
                    if attempts[name]>=2:raise RuntimeError('Worker failed twice; preserved logs: '+name)
        for name in order:
            ev=RUN/'06_evaluation'/name/'metrics.json'
            if ev.exists() and name not in active:done.add(name)
        inventory=gpu_inventory();used={j['gpu'] for j in active.values()};available=sorted([g for g in inventory if g['id'] not in used and g['free_MB']>=16000],key=lambda g:(-g['free_MB'],g['utilization']))
        for name in order:
            if not available:break
            if name in done or name in active:continue
            if name=='ISLB-Only':
                p=RUN/'05_models/L2/outputs.json'
                if not p.exists() or read(p).get('status')!='completed':continue
            gpu=available.pop(0)['id'];attempts[name]+=1;log=(directory/f'{name}_attempt{attempts[name]}.log').open('a')
            args=[str(ROOT/'.venv/bin/python'),'-u','-m','spcgs.islb_sixmodel_train','worker','--model',name,'--gpu',str(gpu)]
            env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'2'};process=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
            active[name]={'process':process,'gpu':gpu,'log':log};write(directory/f'{name}_attempt{attempts[name]}_assignment.json',{'model':name,'GPU':gpu,'pid':process.pid,'command':args,'inventory':inventory,'started':time.time()})
        progress={}
        for name in MODELS:
            p=RUN/'05_models'/name/'outputs.json';row=read(p) if p.exists() else {'status':'pending'}
            if name in active:row.update(worker_pid=active[name]['process'].pid,GPU=active[name]['gpu'])
            row['evaluation_completed']=name in done;progress[name]=row
        write(RUN/'status.json',{'status':'completed' if len(done)==6 else 'running','models':progress,'GPU_inventory':inventory,'updated':time.time()})
        if len(done)<6:time.sleep(15)
    print('ALL_SIX_TRAINED_AND_EVALUATED',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','train','worker','schedule']);p.add_argument('--model',choices=MODELS);p.add_argument('--gpu',type=int,default=2);a=p.parse_args()
    if a.action=='preflight':preflight(a.gpu)
    elif a.action=='schedule':schedule()
    else:
        fit(a.model,a.gpu)
        if a.action=='worker':
            from .islb_sixmodel_evaluate import evaluate
            evaluate(a.model,a.gpu)
