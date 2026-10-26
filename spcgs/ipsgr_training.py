"""Independent models using only the frozen repaired20/6 training protocol."""
import json, time, traceback
from pathlib import Path
import numpy as np
from .common import ROOT, read, write, ref, signature, native
from .ipsgr_protocol import check_inputs, MODEL_NAMES
from .pipeline import base_request


def teacher(run,cfg):
    from .official_teacher import extract
    m=check_inputs(run);cfg={**cfg,'teacher_directory':str(run/'02_semantic_teacher/cache'),'protocol_manifest':str(run/'00_dataset/dataset_manifest.json')}
    start=time.time();out=extract(run,cfg)
    assert {v['name'] for v in out['views']}=={v['name'] for v in m['train']}
    assert not ({v['name'] for v in out['views']}&{v['name'] for v in m['test']})
    write(run/'02_semantic_teacher/outputs.json',{'status':'completed','teacher':ref(run/'02_semantic_teacher/cache/manifest.json'),'seconds':time.time()-start,'train_only':True})
    return out


def model_folder(run,name):
    return run/'03_layout_bootstrap' if name=='ISLB' else run/('04_cross_view' if name.startswith('smoke-') else '05_models')/name


def reset_check(run,folder):
    import torch
    from scipy.spatial import cKDTree
    source=torch.jit.load(str(run/'03_layout_bootstrap/final/model.pt'),map_location='cpu');model=torch.jit.load(str(folder/'initial/model.pt'),map_location='cpu');N=len(source.xyz)
    assert model.xyz.shape==(N,3) and model.semantic.shape==(N,32)
    keys=['xyz','dc','semantic','decoder','bias','classifier','cbias','projection','text']
    errors={k:float((getattr(model,k)-getattr(source,k)).abs().max()) for k in keys};assert all(x==0 for x in errors.values())
    assert model.rest.abs().max()==0 and int(model.sh)==int(model.step)==0
    assert torch.allclose(model.opacity.sigmoid(),torch.full_like(model.opacity,.1),atol=1e-7)
    rotation=torch.zeros_like(model.rotation);rotation[:,0]=1;assert torch.equal(rotation,model.rotation)
    assert all(getattr(model,k).abs().max()==0 for k in ['radii','gradient','denom','exist','added','pruned'])
    # OmniGS distCUDA2 uses the mean squared distance of three nearest neighbors.
    d=cKDTree(model.xyz.numpy()).query(model.xyz.numpy(),k=4)[0][:,1:];expected=np.log(np.sqrt(np.maximum((d*d).mean(1),1e-7)))
    scale_error=float(np.max(np.abs(model.scale.numpy()-expected[:,None])));assert scale_error<.003,scale_error
    audit=read(folder/'stage_reset_audit.json');assert audit['optimizer_state_entries']==audit['head_optimizer_state_entries']==0
    comparisons=[]
    for name in ['L-BOOT','CV-2D','IPSGR-Full']:
        p=run/'05_models'/name/'initial/model.pt'
        if not p.exists() or p==folder/'initial/model.pt':continue
        other=torch.jit.load(str(p),map_location='cpu');same={k:torch.equal(getattr(model,k),getattr(other,k)) for k in keys+['scale','rotation','opacity','rest','cpu_rng','cuda_rng','camera_rng']};assert all(same.values());comparisons.append({'model':name,'equal':same})
    out={'status':'passed','source':ref(run/'03_layout_bootstrap/final/model.pt'),'gaussians':N,'inherited_max_error':errors,'scale_log_max_error_vs_three_neighbor_init':scale_error,'optimizer_states_empty':True,'independent_initializations':comparisons};write(folder/'initialization_validation.json',out);return out


def train(run,cfg,name):
    from .official_teacher import initialize
    m=check_inputs(run);assert name in MODEL_NAMES+['ISLB','smoke-IPSGR-Full']
    folder=model_folder(run,name);folder.mkdir(parents=True,exist_ok=True);smoke=name.startswith('smoke-');kind=name.removeprefix('smoke-');second=kind in ['L-BOOT','CV-2D','IPSGR-Full'];semantic=kind!='B-RGB'
    centers=np.array([v['center'] for v in m['train']]);cfg={**cfg,'classes':[c['name'] for c in m['classes']['classes']]}
    request={**base_request(cfg),'action':'train','output':str(folder),'generated_views':[],'iterations':100 if smoke else 10000,'save_every':100 if smoke else 1000,'extent':float(np.linalg.norm(centers-centers.mean(0),axis=1).max()*1.1),'semantic_supervision':semantic,'ogr':kind=='ISLB','spc2d':kind in ['CV-2D','IPSGR-Full'],'spc3d':kind=='IPSGR-Full','stage_reset':second}
    inputs={'dataset':ref(run/'00_dataset/dataset_manifest.json'),'native':ref(ROOT/'code/bin/spcgs_native'),'implementation':ref(__file__),'semantic_supervision':semantic,'test_images_used':False}
    if semantic:
        cache=read(run/'02_semantic_teacher/cache/manifest.json');assert [v['name'] for v in cache['views']]==[v['name'] for v in m['train']]
        request.update(views=cache['views'],text=cache['text'],decoder=cache['decoder']);inputs['teacher']=ref(run/'02_semantic_teacher/cache/manifest.json')
    else:
        request['views']=m['train'];np.zeros((7,512),dtype='f4').tofile(folder/'unused_text.f32');np.zeros((512,32),dtype='f4').tofile(folder/'unused_decoder.f32');request.update(text=str(folder/'unused_text.f32'),decoder=str(folder/'unused_decoder.f32'))
    if second:
        source=run/'03_layout_bootstrap/final/model.pt';assert read(source.parent/'model.json')['step']==10000
        request['archive']=str(source);inputs['initialization_archive']=ref(source)
    else:
        pc=read(run/'01_geometry'/('P0' if kind in ['B-RGB','B-SEM'] else 'P1')/'outputs.json');assert pc['status']=='completed'
        xyz=np.fromfile(pc['points'],dtype='f4').reshape(-1,6)[:,:3];initial=folder/'initial_semantic.f32'
        if semantic:initialize(xyz,cache,initial)
        else:np.zeros((len(xyz),32),dtype='f4').tofile(initial)
        request.update(points=pc['points'],point_count=len(xyz),initial_semantic=str(initial));inputs['point_cloud']=ref(pc['points']);inputs['initial_semantic']=ref(initial)
    # Device placement is not a model hyperparameter; all numerical inputs are signed.
    key=signature({'request':{k:v for k,v in request.items() if k not in ['output','gpu']},'inputs':inputs});request['signature']=key
    if (folder/'inputs.json').exists():assert read(folder/'inputs.json')['signature']==key,'Input/config changed: use another model directory'
    write(folder/'inputs.json',{**inputs,'signature':key});write(folder/'fixed_request.json',request)
    if (folder/'outputs.json').exists() and read(folder/'outputs.json')['status']=='completed':return folder
    previous=sorted(folder.glob('checkpoint_*/model.pt'),key=lambda p:int(p.parent.name.split('_')[-1]))
    if previous:
        request.update(archive=str(previous[-1]),resume=True,stage_reset=False)
        # A killed process may have logged unsaved steps: preserve then trim the log.
        log=folder/'losses.jsonl';saved=int(previous[-1].parent.name.split('_')[-1])
        if log.exists():
            raw=log.read_text();(folder/f'losses_before_resume_{time.time_ns()}.jsonl').write_text(raw);valid=[]
            for line in raw.splitlines():
                try:r=json.loads(line)
                except json.JSONDecodeError:continue
                if r['step']<=saved:valid.append(line)
            log.write_text('\n'.join(valid)+'\n')
    started=time.time();write(folder/'outputs.json',{'status':'running','started':started,'signature':key,'iterations':request['iterations'],'resumed':bool(previous)})
    try:
        native(request,folder)
        if second:reset_check(run,folder)
        losses=[json.loads(x) for x in (folder/'losses.jsonl').read_text().splitlines()];assert len(losses)==request['iterations'] and losses[-1]['step']==request['iterations']
        assert set(x['view'] for x in losses)<=set(v['name'] for v in m['train'])
        if semantic:assert max(x['semantic_grad_norm'] for x in losses)>0 and max(x['decoder_grad_norm'] for x in losses)>0
        else:assert all(x['clip']==x['local3d']==x['semantic_grad_norm']==0 for x in losses)
        events=[x for x in losses if x.get('isp_event')];grad={k:sum(x.get(k,0)>0 for x in events) for k in ['inter_semantic_grad_norm','intra_semantic_grad_norm','spc3d_semantic_grad_norm']}
        if kind=='IPSGR-Full':assert all(v>0 for v in grad.values()),grad
        elif kind=='CV-2D':assert grad['inter_semantic_grad_norm']>0 and grad['intra_semantic_grad_norm']>0
        for other in MODEL_NAMES:
            p=run/'05_models'/other/'losses.jsonl'
            if not p.exists():continue
            seq=[json.loads(x)['view'] for x in p.read_text().splitlines()];n=min(len(seq),len(losses));assert seq[:n]==[x['view'] for x in losses[:n]],'Training image sequence differs'
        write(folder/'training_audit.json',{'train_only':True,'same_camera_sequence':True,'isp_events':len(events),'expected_isp_events':request['iterations']//100 if request['spc2d'] else 0,'nonzero_gradient_events':grad,'isp_valid_pairs':sum(x['isp_valid_pairs'] for x in events),'semantic_gradient_max':max(x['semantic_grad_norm'] for x in losses),'RGB_only_semantic_state_unused':not semantic})
        write(folder/'outputs.json',{'status':'completed','signature':key,'iterations':request['iterations'],'model':ref(folder/'final/model.pt'),'gaussians':read(folder/'final/model.json')['gaussians'],'seconds_this_invocation':time.time()-started,'started':started,'finished':time.time(),'resumed':bool(previous)})
    except Exception as e:
        (folder/f'failure_{time.time_ns()}.log').write_text(traceback.format_exc());write(folder/'outputs.json',{'status':'failed','signature':key,'error':str(e),'seconds':time.time()-started});raise
    if second and not smoke:
        dest=run/'04_cross_view'/kind
        if not dest.exists():dest.symlink_to(folder,target_is_directory=True)
    print('TRAIN COMPLETE',name,flush=True);return folder
