"""Fresh train-only fixed-pose spherical SfM and the validated Wan360 augmentation."""
import copy, itertools, os, time
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from .common import ROOT, read, write, ref, signature, run_command, pcd_write
from .ipsgr_protocol import check_inputs
from .m1_geometry import load_features, match_features, epipolar_mask, triangulate_track, quantiles
from .m1_wan360 import geometry_settings, augment, point_previews


def build_p0(run,cfg):
    m=check_inputs(run);train=m['train'];folder=run/'01_geometry/P0';folder.mkdir(parents=True,exist_ok=True)
    key=signature({'dataset':m['dataset_signature'],'geometry':geometry_settings(cfg),'implementation':ref(__file__)})
    if (folder/'outputs.json').exists():
        out=read(folder/'outputs.json');assert out['signature']==key
        if out['status']=='completed':return out
    started=time.time();geo=geometry_settings(cfg);write(folder/'outputs.json',{'status':'running','signature':key,'started':started})
    feat=folder/'features';feat.mkdir(exist_ok=True);pairsdir=folder/'matches';pairsdir.mkdir(exist_ok=True)
    scene=read(run/'00_dataset/cameras_train.json');write(feat/'views.json',scene)
    # A fresh algorithm configuration; no previous .feat, .desc or tracks are read.
    describer={'image_describer':{'polymorphic_id':2147483649,'polymorphic_name':'SIFT_Image_describer','ptr_wrapper':{'valid':1,'data':{'params':{'first_octave':-1,'num_octaves':6,'num_scales':3,'edge_threshold':10.,'peak_threshold':0.009999999776482582,'root_sift':True},'bOrientation':True}}},'regions_type':{'polymorphic_id':2147483650,'polymorphic_name':'SIFT_Regions','ptr_wrapper':{'valid':1,'data':{'value0':[],'value1':[]}}}}
    write(feat/'image_describer.json',describer)
    env=os.environ.copy();env['OMP_NUM_THREADS']='4'
    run_command([Path(geo['openmvg_bin'])/'openMVG_main_ComputeFeatures','-i',feat/'views.json','-o',feat,'-m','SIFT','-p','NORMAL','-n','4'],folder/'features.log',env)
    features=[load_features(feat,v['name']) for v in train];cv2.setNumThreads(4)
    offsets=np.cumsum([0]+[len(f['uv']) for f in features]);parent=np.arange(offsets[-1],dtype='i4');size=np.ones(len(parent),dtype='i4');used=set()
    def root(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=int(parent[x])
        return x
    def union(a,b):
        used.update((a,b));a=root(a);b=root(b)
        if a==b:return
        if size[a]<size[b]:a,b=b,a
        parent[b]=a;size[a]+=size[b]
    stats=[]
    for i,j in itertools.combinations(range(len(train)),2):
        path=pairsdir/(train[i]['name']+'__'+train[j]['name']+'.npz')
        if path.exists():
            with np.load(path) as z:pairs,ratio,inlier,error=z['pairs'],z['ratio'],z['inlier'],z['error']
        else:
            cv2.setRNGSeed(cfg['seed']);pairs,ratio=match_features(features[i],features[j],geo['ratio'],geo['mutual'])
            inlier,error=epipolar_mask(train[i],train[j],features[i]['uv'][pairs[:,0]],features[j]['uv'][pairs[:,1]],geo)
            np.savez_compressed(path,pairs=pairs,ratio=ratio,inlier=inlier,error=error)
        for a,b in pairs[inlier]:union(int(offsets[i]+a),int(offsets[j]+b))
        stats.append({'a':train[i]['name'],'b':train[j]['name'],'matches':len(pairs),'inliers':int(inlier.sum()),'epipolar_error':quantiles(error[inlier]),'evidence':str(path)})
        write(folder/'matching_statistics.json',stats);print('P0 PAIR',i,j,len(pairs),int(inlier.sum()),flush=True)
    tracks={}
    for x in sorted(used):
        i=int(np.searchsorted(offsets,x,side='right')-1);tracks.setdefault(root(x),[]).append((i,int(x-offsets[i])))
    write(folder/'tracks.json',list(tracks.values()));images=[np.array(Image.open(v['image']).convert('RGB')) for v in train]
    attempts=[];points=[];colors=[];accepted=[];rejected=Counter()
    for tid,links in tracks.items():
        obs=[{'view':train[i]['name'],'view_id':train[i]['view_id'],'feature_id':fid,'uv':features[i]['uv'][fid].tolist()} for i,fid in links]
        record={'track_id':int(tid),'observations':obs,'reason':None};attempts.append(record)
        if len(set(i for i,_ in links))!=len(links):record['reason']='conflicting observations within one camera';rejected[record['reason']]+=1;continue
        pixels=[features[i]['uv'][fid] for i,fid in links]
        point,info=triangulate_track([train[i] for i,_ in links],pixels,[(2048,1024)]*len(links),geo)
        info['real_views']=len(links)
        if info['reason']=='low real-generated parallax':info['reason']='low real-real parallax'
        record.update(info);record['xyz']=point.tolist() if point is not None and np.isfinite(point).all() else None
        if info['reason']:rejected[info['reason']]+=1;continue
        if points and np.min(np.linalg.norm(np.array(points)-point,axis=1))<geo['duplicate_radius']:
            record['reason']='duplicate point';rejected[record['reason']]+=1;continue
        rgb=[images[i][int(np.clip(round(uv[1]),0,1023)),int(round(uv[0]))%2048]/255. for (i,_),uv in zip(links,pixels)]
        points.append(point);colors.append(np.median(rgb,axis=0));accepted.append(record)
    xyz=np.array(points,dtype='f4').reshape(-1,3);rgb=np.array(colors,dtype='f4').reshape(-1,3)
    assert len(xyz)>=100,'Too few freshly triangulated points'
    neighbors=cKDTree(xyz).query_ball_point(xyz,geo['outlier_radius'],return_length=True)-1;keep=neighbors>=geo['outlier_min_neighbors']
    for r,n,k in zip(accepted,neighbors,keep):
        r['neighbors']=int(n)
        if not k:r['reason']='isolated point';rejected['isolated point']+=1
    xyz=xyz[keep];rgb=rgb[keep];accepted=[r for r,k in zip(accepted,keep) if k]
    assert sum(len(r['observations'])>=3 for r in accepted)>=100,'Insufficient train-only 3-view anchors'
    structure=[]
    for k,r in enumerate(accepted):structure.append({'key':k,'value':{'X':r['xyz'],'observations':[{'key':o['view_id'],'value':{'id_feat':o['feature_id'],'x':o['uv']}} for o in r['observations']]}})
    scene['structure']=structure;write(folder/'structure.json',scene);write(folder/'triangulation.json',attempts);write(folder/'observations.json',accepted)
    pcd_write(folder/'points.ply',xyz,rgb);np.concatenate([xyz,rgb],1).astype('f4').tofile(folder/'points.f32')
    (folder/'preview').mkdir(exist_ok=True);point_previews(folder/'preview',xyz,np.empty((0,3)),train)
    metrics={'points':len(xyz),'anchors_three_or_more_views':sum(len(r['observations'])>=3 for r in accepted),'features':{v['name']:len(f['uv']) for v,f in zip(train,features)},'matched_pairs':len(stats),'descriptor_matches':sum(r['matches'] for r in stats),'geometric_matches':sum(r['inliers'] for r in stats),'tracks':len(tracks),'rejected':dict(rejected),'reprojection_spherical_px_2k':quantiles([e for r in accepted for e in r['error_spherical_px_2k']]),'max_track_error_px':quantiles([r['max_error_px'] for r in accepted]),'parallax_deg':quantiles([r['parallax_deg'] for r in accepted]),'seconds':time.time()-started,'all_observations_train_only':True,'features_extracted_from_repaired_RGB':True}
    write(folder/'metrics.json',metrics);out={'status':'completed','signature':key,'point_count':len(xyz),'points':str(folder/'points.f32'),'ply':str(folder/'points.ply'),'metrics':ref(folder/'metrics.json'),'seconds':time.time()-started};write(folder/'outputs.json',out);print('P0 COMPLETE',metrics,flush=True);return out


def build_p1(run,cfg):
    m=check_inputs(run);folder=run/'01_geometry/P1';folder.mkdir(exist_ok=True);(folder/'preview').mkdir(exist_ok=True)
    p0=read(run/'01_geometry/P0/outputs.json');assert p0['status']=='completed'
    data=np.fromfile(p0['points'],dtype='f4').reshape(-1,6);xyz,rgb=data[:,:3],data[:,3:];pcd_write(folder/'original.ply',xyz,rgb)
    videos=read(run/'01_geometry/Wan360/generated_videos.json');assert len(videos)==80
    for v in videos:
        assert Path(v['video']).resolve().is_relative_to(run.resolve());v['cache_reused']=True
    tracks=read(run/'01_geometry/P0/observations.json');existing={(o['view'],o['feature_id']) for t in tracks for o in t['observations']}
    start=time.time();out,metrics=augment(folder,cfg,m['train'],xyz,rgb,cached_videos=videos,existing_features=existing)
    selected={v['name']:v for v in read(folder/'selected_views.json')};obs=read(folder/'observations.json');metrics['by_direction']=dict(Counter(selected[o['generated_view']]['direction'] for o in obs));metrics['by_source']=dict(Counter(selected[o['generated_view']]['source_training_view'] for o in obs));metrics['fresh_videos_this_experiment']=80;metrics['old_videos_reused']=0;metrics['seconds']=time.time()-start
    write(folder/'metrics.json',metrics);out.update(status='completed',metrics=ref(folder/'metrics.json'));write(folder/'outputs.json',out);print('P1 COMPLETE',metrics,flush=True);return out
