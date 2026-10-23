"""Registered generated-view adapter using the existing OpenMVG SIFT pipeline.

The input manifest must contain independently registered poses in the RMX world
frame. Rotations without a new center are rejected. No model hallucination or
estimated image similarity is used as a substitute for geometric registration.
"""
from pathlib import Path
import copy,os
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from .common import *
from .visuals import projection_preview

BIN=ROOT.parent/'deps/openmvg-build/Linux-x86_64-Release'

def augment(folder,cfg,train,original_xyz,original_rgb):
    folder=Path(folder)
    if cfg['m1_mode']=='online':
        cmd=cfg.get('generation_command');assert isinstance(cmd,list) and cmd,'generation_command must be an argv list'
        # Provider is explicitly configured; it must write the declared manifest.
        run_command(cmd,folder/'generation.log')
    manifest=read(cfg['generated_manifest']);generated=manifest['views'];assert generated,'No generated views'
    assert manifest.get('coordinate_system')=='rmx_openmvg_world','Generated pose frame must match the frozen RMX/OpenMVG frame'
    base=read(DATA/'openMVG/data_openmvg_train.json');scene=copy.deepcopy(base);scene['views']=[];scene['extrinsics']=[];scene['structure']=[];scene['control_points']=[]
    images=folder/'images';images.mkdir(exist_ok=True);matches=folder/'matches';matches.mkdir(exist_ok=True)
    scene['root_path']=str(images);allviews=[];template=base['views'][0];base_intr=base['intrinsics'][0]
    scene['intrinsics']=[]
    for i,v in enumerate(train+generated):
        v=copy.deepcopy(v);im=Image.open(v['image']);w,h=im.size
        if i>=len(train):
            assert v.get('registered',False) and v.get('registration_evidence'),'Registered pose and its evidence are required'
            assert v.get('source_training_view') in {x['name'] for x in train},'Generated input must originate from this run training split'
            assert v.get('projection','equirectangular')=='equirectangular','This OpenMVG adapter currently accepts registered ERP images. Perspective generator output must first be registered/converted by the provider; it is never relabeled as ERP.'
            assert abs(w/h-2)<.01,'ERP aspect ratio must be 2:1'
            assert np.min(np.linalg.norm(np.array([x['center'] for x in train])-v['center'],axis=1))>1e-4,'Pure rotation is not a new observation center'
        R=np.array(v['rotation']);assert R.shape==(3,3) and np.allclose(R@R.T,np.eye(3),atol=1e-4) and np.linalg.det(R)>.999,'Invalid rotation'
        name=f'{i:04d}_{Path(v["image"]).stem}.png';im.convert('RGB').save(images/name)
        node=copy.deepcopy(template);node['key']=i;node['value']['ptr_wrapper']['id']=2147484000+i;d=node['value']['ptr_wrapper']['data']
        d.update(filename=name,width=w,height=h,id_view=i,id_pose=i,id_intrinsic=i)
        intr=copy.deepcopy(base_intr);intr['key']=i;intr['value']['ptr_wrapper']['id']=2147485000+i;intr['value']['ptr_wrapper']['data']['value0'].update(width=w,height=h)
        scene['views'].append(node);scene['intrinsics'].append(intr);scene['extrinsics'].append({'key':i,'value':{'rotation':v['rotation'],'center':v['center']}});allviews.append({**v,'image':str(images/name),'width':w,'height':h})
    write(folder/'registered_views.json',scene);write(folder/'generated_input_manifest.json',manifest)
    pairs=[]
    for j in range(len(train),len(allviews)):
        ds=np.linalg.norm(np.array([x['center'] for x in train])-allviews[j]['center'],axis=1)
        pairs.extend((int(i),j) for i in np.argsort(ds)[:min(4,len(train))])
    (folder/'pairs.txt').write_text(''.join(f'{a} {b}\n' for a,b in pairs))
    env=os.environ.copy();env['OMP_NUM_THREADS']='4'
    run_command([BIN/'openMVG_main_ComputeFeatures','-i',folder/'registered_views.json','-o',matches,'-m','SIFT_ANATOMY','-p','NORMAL','-n','4'],folder/'sfm.log',env)
    run_command([BIN/'openMVG_main_ComputeMatches','-i',folder/'registered_views.json','-o',matches/'matches.putative.bin','-p',folder/'pairs.txt','-r','.75','-n','BRUTEFORCEL2'],folder/'sfm.log',env)
    # Known-pose guided matching + robust triangulation; no bundle adjustment of RMX poses.
    run_command([BIN/'openMVG_main_ComputeStructureFromKnownPoses','-i',folder/'registered_views.json','-m',matches,'-f',matches/'matches.putative.bin','-o',folder/'triangulated.json','-r','2'],folder/'sfm.log',env)
    result=read(folder/'triangulated.json');accepted=[];colors=[];quality=[];rejected={}
    for landmark in result['structure']:
        point=landmark['value'];xyz=np.array(point['X']);obs=point['observations'];ids=[x['key'] for x in obs]
        reason=None
        if not any(i<len(train) for i in ids) or not any(i>=len(train) for i in ids):reason='no_real_generated_track'
        rays=[];residuals=[];col=[]
        for o in obs:
            v=allviews[o['key']];u,y,rad=project(xyz[None],v,v['height'],v['width']);uv=np.array(o['value']['x']);dx=abs(u[0]-uv[0]);dx=min(dx,v['width']-dx);residuals.append(float(np.hypot(dx,y[0]-uv[1])))
            ray=xyz-np.array(v['center']);rays.append(ray/np.linalg.norm(ray))
            if o['key']<len(train):
                arr=np.array(Image.open(v['image']).convert('RGB'));col.append(arr[int(round(uv[1]))%arr.shape[0],int(round(uv[0]))%arr.shape[1]]/255.)
        angle=float(np.rad2deg(np.arccos(np.clip(np.min(np.array(rays)@np.array(rays).T),-1,1)))) if len(rays)>1 else 0
        if not np.isfinite(xyz).all() or max(residuals,default=999)>2.5:reason='reprojection'
        if angle<1:reason='low_parallax'
        if reason:rejected[reason]=rejected.get(reason,0)+1;continue
        accepted.append(xyz);colors.append(np.mean(col,0));quality.append({'landmark':landmark['key'],'observations':obs,'max_reprojection_px':max(residuals),'parallax_deg':angle})
    assert accepted,'No geometrically valid generated observations survived; no enhanced point cloud is claimed'
    add=np.array(accepted,dtype='f4');rgb=np.array(colors,dtype='f4');dist,_=cKDTree(original_xyz).query(add);keep=dist>1e-4;add=add[keep];rgb=rgb[keep]
    xyz=np.concatenate([original_xyz,add]);rgb=np.concatenate([original_rgb,rgb]);pcd_write(folder/'enhanced.ply',xyz,rgb);np.concatenate([xyz,rgb],1).astype('f4').tofile(folder/'enhanced_points.f32')
    write(folder/'observations.json',quality)
    for cam in train[:2]:projection_preview(xyz,rgb,cam,folder/'preview'/(cam['name']+'_enhanced.png'))
    metrics={'original_count':len(original_xyz),'enhanced_count':len(xyz),'added':len(add),'pairs':len(pairs),'raw_triangulated':len(result['structure']),'rejected':rejected,'new_point_neighbor_distance_quantiles':np.quantile(dist,[.1,.5,.9]).tolist(),'backend':'OpenMVG SIFT_ANATOMY with default RootSIFT, fixed-pose guided matching and robust triangulation'}
    return {'points':str(folder/'enhanced_points.f32'),'point_count':len(xyz),'source':'enhanced SfM','enhanced_ply':str(folder/'enhanced.ply'),'enhanced_points':str(folder/'enhanced_points.f32'),'registered_generated_views':generated},metrics
