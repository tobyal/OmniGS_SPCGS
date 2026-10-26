"""Frozen repaired20/6 protocol for the independent IPSGR experiment."""
from pathlib import Path
import copy, csv, json, shutil, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .common import ROOT, read, write, ref, sha, signature, default_config

DEFAULT_RUN=ROOT/'runs/industrial_ipsgr_20train_6test_repaired7_20261025'
SOURCE=ROOT.parent/'datasets/iris_official20_6_no_tripod_2k_20261025'
DIRECTORIES=['00_dataset','01_geometry','02_semantic_teacher','03_layout_bootstrap','04_cross_view','05_models','06_evaluation','07_industrial_analysis','08_paper_figures','09_paper_tables','10_report']
MODEL_NAMES=['B-RGB','B-SEM','G-AUG','L-BOOT','CV-2D','IPSGR-Full']

def protocol(run):
    return read(Path(run)/'00_dataset/dataset_manifest.json')

def check_inputs(run, split='train'):
    manifest=protocol(run)
    for v in manifest[split]:
        assert sha(v['image'])==v['rgb_sha256'] and sha(v['label'])==v['label_sha256'],v['name']
    return manifest

def audit(run):
    run=Path(run).resolve()
    target=run/'00_dataset/dataset_manifest.json'
    if target.exists():
        check_inputs(run,'train');check_inputs(run,'test');return protocol(run)
    for name in DIRECTORIES:(run/name).mkdir(parents=True,exist_ok=True)
    data=read(SOURCE/'dataset_manifest.json');sem=read(SOURCE/'semantics_iris26_compact7_v4/manifest.json');classes=read(SOURCE/'semantics_iris26_compact7_v4/classes.json')
    assert classes['version']=='iris26_compact7_v4' and [c['id'] for c in classes['classes']]==list(range(7))
    selection=read(data['source_split']['path']);selected={r['image_id']:r for r in selection['selected_cameras']}
    source_rgb={r['image_id']:r for r in data['records']};poses={};scenes={}
    for split in ['train','test']:
        scene=read(SOURCE/'openMVG'/f'data_openmvg_{split}.json');scenes[split]=scene;extr={e['key']:e['value'] for e in scene['extrinsics']}
        for node in scene['views']:
            v=node['value']['ptr_wrapper']['data'];poses[Path(v['filename']).stem]=(extr[v['id_pose']],split,node)
    rows=[];offsets=[];pixel_rows=[];align=[]
    palette=np.array([c['color'] for c in classes['classes']],dtype='u1')
    for r in sem['records']:
        name=r['image_id'];split=r['split'];cid=r['camera_id'];pose,psplit,node=poses[name];assert psplit==split
        rgb_info=source_rgb[name];assert sha(r['rgb']['path'])==r['rgb']['sha256']==rgb_info['clean_rgb']['sha256']
        assert sha(r['labels']['path'])==r['labels']['sha256']
        rgb=np.asarray(Image.open(r['rgb']['path']).convert('RGB'));labels=np.asarray(Image.open(r['labels']['path']))
        original=np.asarray(Image.open(rgb_info['source_rgb']['path']).convert('RGB'));edit=np.asarray(Image.open(rgb_info['edit_mask']['path']))>0
        assert rgb.shape==(1024,2048,3) and labels.shape==(1024,2048) and labels.dtype==np.uint8
        assert set(np.unique(labels))<=set(range(7)) and np.all(labels[edit]==2)
        assert np.array_equal(rgb[~edit],original[~edit]),'ERP outside repair changed'
        R=np.array(pose['rotation']);assert np.max(np.abs(R@R.T-np.eye(3)))<1e-6 and abs(np.linalg.det(R)-1)<1e-6
        raw=selected[name];raw_center=np.array([raw['x_m'],raw['y_m'],raw['z_m']]);offsets.append(raw_center-np.array(pose['center']))
        dest=run/'00_dataset/rgb'/split/(name+'.png');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(r['rgb']['path'],dest)
        lab=run/'00_dataset/labels'/split/(name+'.png');lab.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(r['labels']['path'],lab)
        obs=run/'00_dataset/rgb_observation_masks'/(name+'.png');obs.parent.mkdir(exist_ok=True);shutil.copy2(r['rgb_observation_mask']['path'],obs)
        prov=run/'00_dataset/semantic_provenance'/(name+'.png');prov.parent.mkdir(exist_ok=True);shutil.copy2(r['provenance']['path'],prov)
        row={'name':name,'camera_id':cid,'split':split,'image':str(dest),'label':str(lab),'rotation':pose['rotation'],'center':pose['center'],'absolute_center_m':raw_center.tolist(),'width':2048,'height':1024,'view_id':node['key'],'original_train_index':int(cid.split('-')[1]),'rgb_sha256':sha(dest),'label_sha256':sha(lab),'rgb_source':r['rgb'],'label_source':r['labels'],'rgb_observation_mask':ref(obs),'semantic_provenance':ref(prov),'rmx':ref(raw['rmx_source']),'official_annotations_available':True,'labels_are_repaired_derivatives':True}
        rows.append(row)
        for k in range(7):pixel_rows.append({'camera_id':cid,'split':split,'class_id':k,'class_name':classes['classes'][k]['name'],'pixels':int((labels==k).sum()),'erp_fraction':float((labels==k).mean())})
        overlay=(.55*rgb+.45*palette[labels]).astype('u1');panel=Image.new('RGB',(1024,1024));panel.paste(Image.fromarray(rgb).resize((1024,512)),(0,0));panel.paste(Image.fromarray(overlay).resize((1024,512)),(0,512));folder=run/'00_dataset/alignment';folder.mkdir(exist_ok=True);panel.save(folder/(cid+'.jpg'),quality=94)
        align.append({'camera_id':cid,'dimensions_aligned':True,'outside_repair_RGB_identical':True,'repair_pixels':int(edit.sum()),'repair_semantics_building':True,'semantic_ignore_pixels':0,'ERP_orientation_unchanged':True})
    assert np.max(np.ptp(np.array(offsets),axis=0))<1e-6
    ordered={r['camera_id']:r for r in rows};train=[ordered[c] for c in selection['train_ids']];test=[ordered[c] for c in selection['test_ids']]
    assert len(train)==20 and len(test)==6 and not ({v['name'] for v in train}&{v['name'] for v in test})
    label_rows=[{'id':v['name'],'label':v['label'],'label_sha256':v['label_sha256'],'valid_fraction':1.,'label_kind':'repaired7_v4'} for v in train]
    manifest={'schema':'ipsgr_repaired20_6_v1','frozen_at':time.time(),'train':train,'test':test,'validation':[],'unseen_train29':[],'labels':label_rows,'classes':classes,'train_count':20,'test_count':6,'validation_count':0,'resolution':[2048,1024],'semantic_resolution':[256,128],'coordinate_system':'fixed OpenMVG world, camera axes +X right,+Y down,+Z ERP center; absolute RMX center minus shared origin','absolute_to_native_origin':offsets[0].tolist(),'source_split':data['source_split'],'source_rgb_manifest':ref(SOURCE/'dataset_manifest.json'),'source_semantic_manifest':ref(SOURCE/'semantics_iris26_compact7_v4/manifest.json'),'repair_policy':'Use fixed repaired RGB and all seven-class labels for supervision and primary full-frame evaluation. RGB synthesis provenance is kept separately. No reannotation or repair.','split_history':'Test image feature overlap was previously consulted for choosing the user-approved interpolation split; fixed before current experiments. Not a blind randomly sampled benchmark.','evaluation_protocol':{'rgb':'full repaired ERP2048x1024; arithmetic mean across6 held-out test views; final10k checkpoint only','semantic':'GT nearest-neighbor256x128;7x8 confusion includes uncovered prediction column; uncovered valid GT is always an error; pooled mIoU and per-image metrics reported separately','semantic_highres_check':[512,256],'test_labels_never_used_in_teacher':True,'class_query_separate_from_classifier':True},'generated_semantic_loss':False,'generated_semantic_note':'Wan360 RGB frames are geometry inputs only; no generated labels are fabricated. Train supervision uses exactly20 repaired real labels.'}
    manifest['dataset_signature']=signature({'train':train,'test':test,'classes':classes,'evaluation_protocol':manifest['evaluation_protocol']})
    write(target,manifest);write(run/'00_dataset/classes.json',classes);write(run/'00_dataset/alignment_audit.json',align);write(run/'00_dataset/pixel_statistics.json',pixel_rows)
    # Compatibility aliases point only to this new run, never to old experiments.
    compat=run/'00_protocol';compat.mkdir(exist_ok=True)
    if not (compat/'manifest.json').exists():(compat/'manifest.json').symlink_to(target)
    for split,views in [('train',train),('test',test)]:
        scene=copy.deepcopy(scenes[split]);scene.update(root_path=str(run/'00_dataset/rgb'/split),structure=[],control_points=[])
        write(run/'00_dataset'/f'cameras_{split}.json',scene)
    config=default_config('full');config.update(seed=42,gpu=0,train_views=20,eval_views=6,train_names=[v['name'] for v in train],classes=[c['name'] for c in classes['classes']],cache_width=256,cache_height=128,feature_backend='official_prototypes',generated_semantic_loss=False,sam_config='configs/sam2/sam2_hiera_l.yaml',sam_checkpoint=str(ROOT/'weights/sam2_hiera_large.pt'),require_sam2_hiera_large=True,sam_reuse_model=True,semantic_supervision=True,m1_mode='wan360',selection_by_novel_points=True)
    config['wan360']={'views':20,'view_names':[v['name'] for v in train],'directions':['forward','backward','left','right'],'distance':.8,'width':1024,'height':512,'num_frames':49,'steps':30,'candidate_frames':[1,2,3,4],'cache':str(run/'01_geometry/wan360_cache'),'reuse':True}
    config['m1_geometry']={'features':str(run/'01_geometry/P0/features'),'structure':str(run/'01_geometry/P0/structure.json'),'feature_cache':str(run/'01_geometry/candidate_feature_cache'),'match_cache':str(run/'01_geometry/candidate_match_cache'),'scene_template':str(run/'00_dataset/cameras_train.json')}
    write(run/'experiment_config.json',config)
    write(run/'experiment_manifest.json',{'method':'IPSGR','stages':{'ISLB':'Industrial Scene Layout Bootstrapping','ICSC':'Industrial Cross-View Semantic Consolidation'},'dataset':ref(target),'config':ref(run/'experiment_config.json'),'models':{name:{'status':'pending','iterations':10000,'independent':True} for name in MODEL_NAMES},'layout_iterations':10000,'all_old_image_dependent_caches_forbidden':True,'planned_wan360_jobs':80,'prior_checkpoints_allowed':False,'reference':'https://arxiv.org/html/2503.12535','status':'running','created':time.time()})
    write(run/'08_paper_figures/registry.json',{f'F{i:02}':{'status':'pending' if i!=10 else 'optional','raw_data':[],'script':None} for i in range(1,12)})
    write(run/'09_paper_tables/registry.json',{f'T{i:02}':{'status':'pending','raw_data':[],'script':None} for i in range(1,7)})
    figures(run,manifest,pixel_rows)
    print('DATASET_FROZEN',len(train),len(test),manifest['dataset_signature'],flush=True)
    return manifest

def figures(run,m,rows):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    train=np.array([v['center'] for v in m['train']]);test=np.array([v['center'] for v in m['test']]);nearest=np.linalg.norm(test[:,None]-train[None],axis=-1)
    relations=[]
    fig=plt.figure(figsize=(17,6));a=fig.add_subplot(131);b=fig.add_subplot(132,projection='3d');c=fig.add_subplot(133)
    for split,color,marker in [('train','#226ca5','o'),('test','#e06b26','^')]:
        views=m[split];xyz=np.array([v['center'] for v in views]);a.scatter(xyz[:,0],xyz[:,1],c=color,marker=marker,label=f'{split}: {len(views)}');b.scatter(*xyz.T,c=color,marker=marker,label=split)
        ordered=sorted(m['train']+m['test'],key=lambda v:int(v['camera_id'].split('-')[1]));indices={v['name']:i for i,v in enumerate(ordered)};c.scatter([indices[v['name']] for v in views],[v['absolute_center_m'][2] for v in views],c=color,marker=marker,label=split)
        for v in views:
            offset=(5,6) if v['absolute_center_m'][2]>152.1 else (5,-12)
            a.annotate(v['camera_id'],v['center'][:2],fontsize=7,xytext=offset,textcoords='offset points',color=color,arrowprops=dict(arrowstyle='-',lw=.5,color=color))
    for i,v in enumerate(m['test']):
        ids=np.argsort(nearest[i])[:3];relations.append({'test':v['camera_id'],'neighbors':[{'train':m['train'][j]['camera_id'],'distance_m':float(nearest[i,j])} for j in ids]})
        for j in ids:a.plot([test[i,0],train[j,0]],[test[i,1],train[j,1]],c='#999',alpha=.3,lw=.7)
    a.set(xlabel='X (m)',ylabel='Y (m)',title='Fixed20/6 / XY and nearest supports');a.set_aspect('equal');a.margins(.12);a.legend();b.set(xlabel='X',ylabel='Y',zlabel='Z',title='Camera centers / XYZ');c.set_xticks(range(len(ordered)),[v['camera_id'] for v in ordered],rotation=90,fontsize=7);c.set(xlabel='Camera ID',ylabel='Absolute RMX Z (m)',title='Two camera-height levels');c.legend();fig.tight_layout();fig.savefig(run/'08_paper_figures/F04_cameras.png',dpi=220);fig.savefig(run/'08_paper_figures/F04_cameras.pdf');plt.close(fig)
    write(run/'00_dataset/test_train_neighbors.json',relations)
    table=[]
    for cl in m['classes']['classes']:
        r={'id':cl['id'],'name':cl['name']}
        for split,n in [('train',20),('test',6)]:
            rr=[x for x in rows if x['class_id']==cl['id'] and x['split']==split];r[split+'_pixels']=sum(x['pixels'] for x in rr);r[split+'_fraction']=r[split+'_pixels']/(n*2048*1024);r[split+'_views']=sum(x['pixels']>0 for x in rr)
        table.append(r)
    write(run/'09_paper_tables/T01.json',{'protocol':ref(run/'00_dataset/dataset_manifest.json'),'classes':table,'ignore_fraction':0.})
    with (run/'09_paper_tables/T01.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    fig,ax=plt.subplots(figsize=(12,5));x=np.arange(7)
    for off,split,col in [(-.18,'train','#226ca5'),(.18,'test','#e06b26')]:ax.bar(x+off,[r[split+'_fraction']*100 for r in table],width=.35,label=split,color=col)
    ax.set_xticks(x,[c['name'] for c in m['classes']['classes']],rotation=15);ax.set(ylabel='ERP pixels (%)',title='Fixed seven-class label coverage');ax.legend();fig.tight_layout();fig.savefig(run/'08_paper_figures/F04_classes.png',dpi=220);plt.close(fig)
    for folder,key in [('08_paper_figures','F04'),('09_paper_tables','T01')]:
        registry=read(run/folder/'registry.json');registry[key]={'status':'completed','raw_data':[str(run/'00_dataset/dataset_manifest.json'),str(run/'00_dataset/pixel_statistics.json')],'script':str(ROOT/'spcgs/ipsgr_protocol.py')}
        if key=='F04':registry[key]['figures']=[str(run/folder/n) for n in ['F04_cameras.png','F04_cameras.pdf','F04_classes.png']]
        write(run/folder/'registry.json',registry)
