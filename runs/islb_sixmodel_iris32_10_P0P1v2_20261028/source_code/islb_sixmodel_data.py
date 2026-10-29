"""Frozen IRIS32/10 + geometry-v2 inputs for the six-model ISLB experiment."""
from pathlib import Path
import os,shutil,time,collections
import numpy as np
from PIL import Image
from .common import ROOT,read,write,ref,signature,sha,project
from .official_teacher import targets

RUN=ROOT/'runs/islb_sixmodel_iris32_10_P0P1v2_20261028'
GEOMETRY=ROOT/'runs/iris32_P0P1_20261027_v2'
MODELS=['B-RGB','B-SEM','L0','L1','L2','ISLB-Only']
EXPECTED_TRAIN=['1-01','1-02','1-03','1-04','1-13','1-15','1-16','1-18','1-21','1-22','1-24','1-25','1-28','1-29','1-36','1-37','1-40','1-44','1-47','1-58','1-59','1-62','1-63','1-64','1-65','1-66','1-67','1-70','1-71','1-74','1-77','1-78']
EXPECTED_TEST=['1-17','1-19','1-26','1-27','1-34','1-35','1-60','1-61','1-68','1-69']

def setup():
    assert not RUN.exists(),'A new independent run is required; use existing resume commands if prepared'
    for f in ['00_protocol','01_initialization','02_teacher','03_anchors','04_preflight','05_models','06_evaluation','07_figures','08_tables','09_report','source_code','native','scheduler']:(RUN/f).mkdir(parents=True,exist_ok=True)
    m=read(GEOMETRY/'00_dataset/dataset_manifest.json')
    assert [v['camera_id'] for v in m['train']]==EXPECTED_TRAIN
    assert [v['camera_id'] for v in m['test']]==EXPECTED_TEST and not m['validation']
    classes=read(ROOT/'configs/iris42_compact8_classes.json');assert m['classes']==classes
    assert [c['id'] for c in classes['classes']]==list(range(8))
    m.update(experiment='islb_sixmodel_iris32_10_P0P1v2_20261028',authorized_training_request='2026-10-28',
       semantic_supervision='Existing official8 consolidated labels, only provenance1/2 and real RGB observations. No SAM labels; synthetic floor and residual neighbor fill excluded from primary supervision/evaluation.',
       semantic_primary_resolution=[512,256],semantic_training_resolution=[256,128],semantic_qualitative_resolution=[1024,512],
       evaluation='10 fixed test ERP images; RGB full2048x1024; semantic pooled confusion with uncovered as error; full-consolidated-label secondary results reported separately.',
       reference='https://arxiv.org/html/2503.12535',teacher_method='Frozen CLIP ViT-B/16 text prototypes derived from current eight-class prompts; supervised prototype cosine+CE, not image-CLIP distillation')
    references=[ref(GEOMETRY/'00_dataset/dataset_manifest.json'),ref(ROOT/'configs/iris42_compact8_classes.json')]
    rows=[]
    for split in ['train','test']:
        for v in m[split]:
            assert sha(v['image'])==v['rgb_sha256'] and sha(v['label'])==v['label_sha256']
            lab=np.array(Image.open(v['label']));prov=np.array(Image.open(v['semantic_provenance']['path']));rgb=np.array(Image.open(v['rgb_observation_mask']['path']))>0
            assert lab.shape==prov.shape==rgb.shape==(1024,2048) and set(np.unique(lab))<=set(range(8))
            mask=np.isin(prov,[1,2])&rgb;dest=RUN/'00_protocol/valid_masks'/(v['name']+'.png');dest.parent.mkdir(exist_ok=True)
            Image.fromarray(mask.astype('u1')*255).save(dest);v['supervision_mask']=str(dest)
            rows.append({'camera_id':v['camera_id'],'split':split,'valid_fraction':float(mask.mean()),'full_class_pixels':np.bincount(lab.ravel(),minlength=8).tolist(),'valid_class_pixels':np.bincount(lab[mask],minlength=8).tolist(),'RGB_state':v['rgb_state']})
            references.extend(ref(p) for p in [v['image'],v['label'],v['semantic_provenance']['path'],v['rgb_observation_mask']['path']])
    m['labels']=[{'id':v['name'],'label':v['label'],'label_sha256':v['label_sha256'],'valid_mask':v['supervision_mask']} for v in m['train']]
    m['dataset_signature']=signature(m);write(RUN/'00_protocol/dataset_manifest.json',m);write(RUN/'00_protocol/label_statistics.json',rows)
    for key,source,count in [('P0',GEOMETRY/'01_geometry/P0_v2/points.f32',2781),('P1',GEOMETRY/'01_geometry/P1_v2/enhanced_points.f32',7265)]:
        dest=RUN/'01_initialization'/(key+'_v2.f32');shutil.copy2(source,dest)
        a=np.fromfile(dest,'f4').reshape(-1,6);assert len(a)==count and np.isfinite(a).all()
        write(RUN/'01_initialization'/(key+'_provenance.json'),{'source':ref(source),'copy':ref(dest),'count':count,'version':'iris32_P0P1_20261027_v2','modified':False});references.append(ref(source))
    assert np.array_equal(np.fromfile(RUN/'01_initialization/P0_v2.f32','f4'),np.fromfile(RUN/'01_initialization/P1_v2.f32','f4')[:2781*6])
    for p in [ROOT/'code/examples/spcgs_native.cpp',ROOT/'code/src/gaussian_renderer.cpp',ROOT/'code/src/gaussian_model.cpp',ROOT/'spcgs/official_teacher.py',ROOT/'spcgs/islb_anchor.py',ROOT/'spcgs/ipsgr_training.py',ROOT/'spcgs/common.py',ROOT/'spcgs/ipsgr_evaluate.py']:
        dest=RUN/'source_code/before'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);references.append(ref(p))
    for p in (ROOT/'code/lib').glob('*.so'):references.append(ref(p))
    references.append(ref(ROOT/'weights/clip-vit-base-patch16/pytorch_model.bin'))
    write(RUN/'00_protocol/readonly_inputs.json',references)
    cfg=read(GEOMETRY/'experiment_config.json')
    for key in ['m1_geometry','wan360','sam_checkpoint','sam_config','train_names']:cfg.pop(key,None)
    cfg.update(iterations=10000,save_every=1000,seed=42,width=2048,height=1024,semantic_width=256,semantic_height=128,
       classes=[c['name'] for c in classes['classes']],train_views=32,eval_views=10,semantic_head='clip_prototype',
       generated_views=[],generated_semantic_loss=False,spc2d=False,spc3d=False,sam_reuse_model=False,require_sam2_hiera_large=False,
       anchor_weight=.05,anchor_ce_only=True,anchor_semantic_only=True,anchor_gradient_every=100,
       local_samples=128,local_weight=1.,ogr_every=3000,ogr_radius=1.,ogr_min_neighbors=5,
       densify_from=500,densify_until=9000,densify_every=100,densify_threshold=.0002,max_gaussians=1000000,opacity_reset_every=3000,
       native_binary=str(RUN/'native/spcgs_islb_sixmodel'),geometry_worker=str(ROOT/'tools/geometry_worker.py'),python=str(ROOT/'.venv/bin/python'))
    write(RUN/'00_protocol/common_config.json',cfg)
    write(RUN/'status.json',{'status':'preparing','models':{name:{'status':'pending'} for name in MODELS},'created':time.time()})
    print('PREPARED',str(RUN),flush=True)

def teacher(gpu):
    import torch
    from transformers import CLIPModel,CLIPTokenizer
    m=read(RUN/'00_protocol/dataset_manifest.json');out=RUN/'02_teacher';classes=m['classes']['classes'];K=len(classes)
    if (out/'manifest.json').exists():return read(out/'manifest.json')
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu);torch.set_num_threads(4)
    modeldir=ROOT/'weights/clip-vit-base-patch16';model=CLIPModel.from_pretrained(modeldir,local_files_only=True).eval().cuda();model.requires_grad_(False)
    tokenizer=CLIPTokenizer.from_pretrained(modeldir,local_files_only=True);vectors=[];prompts={c['name']:c['prompts'] for c in classes}
    with torch.inference_mode():
        for c in classes:
            tokens=tokenizer(c['prompts'],padding=True,return_tensors='pt');v=model.get_text_features(**{k:x.cuda() for k,x in tokens.items()})
            vectors.append(torch.nn.functional.normalize(torch.nn.functional.normalize(v,dim=-1).mean(0),dim=0).cpu().numpy())
    text=np.array(vectors,dtype='f4');basis=np.linalg.qr(np.concatenate([text.T,np.eye(512,dtype='f4')],1))[0][:,:32].astype('f4')
    assert text.shape==(8,512) and np.max(abs(text@basis@basis.T-text))<1e-5
    text.tofile(out/'text.f32');basis.tofile(out/'decoder.f32');write(out/'prompts.json',prompts)
    views=[];counts=np.zeros(K,dtype='i8');H,W=128,256
    for v in m['train']:
        labs=np.array(Image.open(v['label']).resize((W,H),Image.Resampling.NEAREST));valid=np.array(Image.open(v['supervision_mask']).resize((W,H),Image.Resampling.NEAREST))>0
        labs[~valid]=255;F,S,mask=targets(labs,text,.05);base=out/v['name']
        F.tofile(str(base)+'_F.f16');S.tofile(str(base)+'_S.f16');mask.tofile(str(base)+'_W.u8');Image.fromarray(labs).save(str(base)+'_labels.png')
        counts+=np.bincount(labs[valid],minlength=K)
        assert np.all(F[:,~valid]==0) and np.array_equal(S.argmax(0)[valid],labs[valid])
        views.append({**v,'feature':str(base)+'_F.f16','feature_height':H,'feature_width':W,'relevancy':str(base)+'_S.f16','relevancy_height':H,'relevancy_width':W,
           'relevancy_classes':K,'semantic_target':'softmax_relevancy','semantic_mask':str(base)+'_W.u8','mask_height':H,'mask_width':W,'labels':str(base)+'_labels.png','valid_fraction':float(valid.mean())})
    outputs={p.name:ref(p) for p in out.iterdir() if p.is_file()}
    manifest={'schema':'official8_clip_prototype_v2','views':views,'generated_views':[],'text':str(out/'text.f32'),'decoder':str(out/'decoder.f32'),'classes':[c['name'] for c in classes],
       'smoothing':.05,'frozen_text':True,'class_counts':counts.tolist(),'protocol':ref(RUN/'00_protocol/dataset_manifest.json'),'CLIP_weights':ref(modeldir/'pytorch_model.bin'),
       'targets':'CLIP text prototype F at valid true-label pixels; smoothed CE distribution S; not dense image-CLIP features','outputs':outputs,'test_data_used':False}
    manifest['signature']=signature(manifest);write(out/'manifest.json',manifest)
    for cloud in ['P0','P1']:
        xyz=np.fromfile(RUN/'01_initialization'/(cloud+'_v2.f32'),'f4').reshape(-1,6)[:,:3];initialize(xyz,manifest,RUN/'01_initialization'/(cloud+'_semantic.f32'))
    print('TEACHER_READY',counts.tolist(),flush=True);return manifest

def initialize(xyz,cache,dest):
    from scipy.spatial import cKDTree
    views=cache['views'];basis=np.fromfile(cache['decoder'],'f4').reshape(512,32);text=np.fromfile(cache['text'],'f4').reshape(len(cache['classes']),512);low=text@basis
    nearest=cKDTree([v['center'] for v in views]).query(xyz,k=len(views))[1];out=np.zeros((len(xyz),32),'f4');pending=np.ones(len(xyz),bool)
    for rank in range(len(views)):
        for i,v in enumerate(views):
            ids=np.flatnonzero(pending&(nearest[:,rank]==i))
            if not len(ids):continue
            H,W=v['feature_height'],v['feature_width'];u,y,_=project(xyz[ids],v,H,W);labs=np.array(Image.open(v['labels']))[np.clip(np.rint(y).astype(int),0,H-1),np.rint(u).astype(int)%W]
            valid=labs!=255;out[ids[valid]]=low[labs[valid]];pending[ids[valid]]=False
        if not pending.any():break
    out.tofile(dest);write(str(dest)+'.json',{'points':len(xyz),'initialized':int((~pending).sum()),'zero_unobserved':int(pending.sum()),'policy':'Existing nearest-valid-label projection heuristic; not a visibility claim. Same immutable initialization for all models on this cloud.'})

def anchors():
    from .islb_anchor import depth_buffer,observation_check,semantic_coordinates,sample_label
    from .m1_geometry import load_anchors,project_erp,quantiles
    m=read(RUN/'00_protocol/dataset_manifest.json');teacher=read(RUN/'02_teacher/manifest.json');dest=RUN/'03_anchors'
    if (dest/'manifest.json').exists():return read(dest/'manifest.json')
    cfg={**read(ROOT/'configs/islb_anchor_v1.json'),'classes':8,'semantic_width':256,'semantic_height':128,'semantic_only':True,'ce_only':True}
    real={v['name']:v for v in m['train']};labs={}
    for n,v in real.items():
        lab=np.array(Image.open(v['label']));lab[np.array(Image.open(v['supervision_mask']))==0]=255;labs[n]=lab
    p0=GEOMETRY/'01_geometry/P0_v2';p1=GEOMETRY/'01_geometry/P1_v2'
    original=np.fromfile(p0/'points.f32','f4').reshape(-1,6);enhanced=np.fromfile(p1/'enhanced_points.f32','f4').reshape(-1,6)
    landmark=load_anchors(p0/'structure.json',p0/'features',m['train'],original[:,:3],dest/'source_anchor_audit.json')
    assert len(landmark['landmarks'])==2781
    selected=read(p1/'selected_views.json');generated={v['name']:v for v in selected};assert len(selected)==109
    depths={n:depth_buffer(enhanced[:,:3],v,cfg['depth_grid_width']) for n,v in {**real,**generated}.items()}
    proposed=[];rejected=collections.Counter();rejection_rows=[]
    for v in selected:
        uv=np.loadtxt(p1/'features'/(v['name']+'.feat'),dtype='f4',ndmin=2)[:,:2]
        for c in v['correspondences']:
            if not c['inlier']:rejected['registration_outlier']+=1;continue
            a=landmark['landmarks'][c['anchor_id']]
            proposed.append({'track':'P0:'+str(a['source_landmark_id']),'source':'P0_registration','xyz':a['xyz'],'real_observations':a['observations'],
                'generated_view':v['name'],'generated_feature_id':c['generated_feature_id'],'generated_uv':uv[c['generated_feature_id']].tolist()})
    for i,a in enumerate(read(p1/'observations.json')):proposed.append({**a,'track':'P1:'+str(i),'source':'P1_added'})
    consensus_cache={};accepted=[]
    for a in proposed:
        if a['track'] not in consensus_cache:
            votes=[];errors=[];reason=None;validobs=[]
            # Ignore invalid labels, but reject any disagreement among valid,
            # geometrically sound real observations. Require >=2 distinct views.
            for ob in a['real_observations']:
                n=ob['view'];assert n in real
                lab=sample_label(labs[n],ob['uv'])
                if lab==255:continue
                why,e=observation_check(a['xyz'],ob['uv'],real[n],2048,1024,depths[n],cfg)
                if why:reason='real_'+why;break
                if sample_label(labs[n],project_erp(np.array(a['xyz'])[None],real[n],2048,1024)[0])!=lab:reason='real_boundary';break
                votes.append(lab);errors.append(e);validobs.append(ob)
            if not reason and len({o['view'] for o in validobs})<2:reason='fewer_than_two_valid_real_labels'
            if not reason and len(set(votes))!=1:reason='real_label_conflict'
            consensus_cache[a['track']]=(votes[0] if votes else None,reason,errors,validobs)
        lab,why,errors,validobs=consensus_cache[a['track']];v=generated[a['generated_view']];err=None
        if not why:
            why,err=observation_check(a['xyz'],a['generated_uv'],v,v['width'],v['height'],depths[v['name']],cfg)
            if why:why='generated_'+why
        if why:
            rejected[why]+=1;rejection_rows.append({'track':a['track'],'view':v['name'],'reason':why});continue
        accepted.append({**a,'class_id':lab,'real_support':len(validobs),'valid_real_observations':validobs,'real_errors_px_2k':errors,'generated_error_px_2k':err,
            'semantic_xy':semantic_coordinates(a['generated_uv'],v['width'],v['height'],256,128).tolist()})
    cells=collections.defaultdict(list)
    for a in accepted:
        x,y=np.floor(np.array(a['semantic_xy'])+.5).astype(int);cells[(a['generated_view'],int(x)%256,int(np.clip(y,0,127)))].append(a)
    final=[]
    for rows in cells.values():
        if len({a['class_id'] for a in rows})!=1:rejected['cell_conflict']+=len(rows);continue
        rows.sort(key=lambda a:(a['generated_error_px_2k'],-a['real_support'],a['track']));final.append(rows[0]);rejected['cell_duplicate']+=len(rows)-1
    grouped=collections.defaultdict(list)
    for a in final:grouped[a['generated_view']].append(a)
    views=[]
    for n,rows in sorted(grouped.items()):
        base=dest/n;np.array([a['semantic_xy'] for a in rows],'f4').tofile(str(base)+'_xy.f32');np.array([a['class_id'] for a in rows],'i8').tofile(str(base)+'_labels.i64')
        views.append({k:generated[n][k] for k in ['name','center','rotation','direction','source_training_view']}|{'anchor_count':len(rows),'anchor_xy':str(base)+'_xy.f32','anchor_labels':str(base)+'_labels.i64'})
    assert final and all(a['real_support']>=2 for a in final)
    text=np.fromfile(teacher['text'],'f4').reshape(8,512);targets(np.arange(8,dtype='u1')[None],text,.05)[1].reshape(8,8).tofile(dest/'class_scores.f16')
    summary={'observations':len(final),'unique_tracks':len({a['track'] for a in final}),'generated_views':len(views),'class_counts':{n:sum(a['class_id']==k for a in final) for k,n in enumerate(teacher['classes'])},
       'sources':dict(collections.Counter(a['source'] for a in final)),'real_support_counts':dict(collections.Counter(a['real_support'] for a in final)),
       'proposals':len(proposed),'rejected':dict(rejected),'generated_error_2k_px':quantiles([a['generated_error_px_2k'] for a in final])}
    write(dest/'anchors.json',final);write(dest/'rejections.json',rejection_rows)
    result={'views':views,'summary':summary,'config':cfg,'class_scores':str(dest/'class_scores.f16'),'smoothing':.05,'geometry_source':str(GEOMETRY),'teacher':ref(RUN/'02_teacher/manifest.json'),
      'policy':'At least2 valid real-view labels, unanimous among valid reliable observations; no test labels, generated dense CLIP or generated RGB supervision. Sparse-depth consistency is a filter, not complete visibility truth.'}
    write(dest/'manifest.json',result);print('ANCHORS_READY',summary,flush=True);return result

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['setup','teacher','anchors']);ap.add_argument('--gpu',type=int,default=2);a=ap.parse_args()
    if a.action=='setup':setup()
    elif a.action=='teacher':teacher(a.gpu)
    else:anchors()
