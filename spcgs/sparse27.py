"""Sparse official-label protocol and retriangulation from selected observations."""
from collections import Counter
from itertools import combinations
from pathlib import Path
import copy
import numpy as np
from PIL import Image
from scipy.cluster.hierarchy import linkage,fcluster
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from .common import ROOT,read,write,ref,sha,cameras,pcd_write
from .segmentation import Store,DEFAULT_RUN
from .m1_geometry import load_features,triangulate_track,quantiles
from .m1_wan360 import geometry_settings,point_previews

def prepare_protocol(run,review_root=DEFAULT_RUN):
    run=Path(run).resolve();run.mkdir(parents=True,exist_ok=True);out=run/'00_protocol'
    if (out/'manifest.json').exists():return read(out/'manifest.json')
    store=Store(review_root);all_train={c['name']:c for c in cameras('train')};train=[];unseen=[];labels=[]
    for name,item in store.items.items():
        coverage=read(store.folder(name)/'coverage.json');cam=all_train[name]
        if not all(f['has_record'] for f in coverage):unseen.append(cam);continue
        assert len(coverage)==6 and sha(cam['image'])==item['rgb_sha256']
        current,source,state=store.arrays(name);official=np.array(Image.open(store.folder(name)/'official_mapped.png'))
        approved=state['review_state']=='approved' and state['approved_revision']==state['revision']
        # Only official pixels and explicitly approved human edits are supervised.
        # Inspected SAM artifact removals (source 4) stay ignore. Unapproved SAM
        # additions are never promoted to official ground truth.
        reliable=(source==1)&(current==official)&(official!=255)
        if approved:reliable|=(source==3)&(current!=255)
        used=np.where(reliable,current,255).astype('u1');folder=out/'labels';folder.mkdir(parents=True,exist_ok=True)
        Image.fromarray(used).save(folder/(name+'.png'));Image.fromarray(source).save(folder/(name+'_source.png'))
        train.append({**cam,'original_train_index':item['index']})
        labels.append({'id':name,'label':str(folder/(name+'.png')),'label_sha256':sha(folder/(name+'.png')),
          'official_mapped':ref(store.folder(name)/'official_mapped.png'),'review_revision':state['revision'],
          'review_state':state['review_state'],'human_approved':approved,'current_label':ref(store.folder(name)/'revisions'/state['revision']/'labels.png'),
          'source':ref(folder/(name+'_source.png')),'valid_fraction':float(reliable.mean()),'excluded_sam_pixels':int(((source==2)&(current!=255)).sum())})
    expected=[1,2,10,12,13,14,15,18,19,20,21,25,26,27,30,34,42,43,44,45,46,47,48,49,50,51,56]
    assert [v['original_train_index'] for v in train]==expected,'Official manifest differs from requested list'
    assert len(train)==27 and len(unseen)==29
    xyz=np.array([v['center'] for v in train]);raw_groups=fcluster(linkage(xyz,method='ward'),4,criterion='maxclust')
    order=sorted(set(raw_groups),key=lambda g:xyz[raw_groups==g,1].mean(),reverse=True)
    groups={int(g):j+1 for j,g in enumerate(order)};distance=np.linalg.norm(xyz[:,None]-xyz[None,:],axis=-1)
    spatial=[]
    for i,v in enumerate(train):
        spatial.append({'id':v['name'],'index':v['original_train_index'],'center':v['center'],'region':groups[int(raw_groups[i])],
          'height_layer':'upper' if xyz[i,2]>=0 else 'lower','nearest_training_views':[{'id':train[j]['name'],'distance_3d':float(distance[i,j])} for j in np.argsort(distance[i])[1:5]]})
    protocol={'schema':'iris_sparse27_official_v1','selection':'actual review manifest: all six official cubemap records; expected indices checked after ID selection',
      'train':train,'unseen_train29':unseen,'validation':cameras('val'),'test':cameras('test'),'labels':labels,
      'classes':store.config,'spatial':spatial,'grouping':'Ward clustering on original 3D centers, 4 regions; Z>=0 separates the observed height layers',
      'label_policy':'source=1 matching official mapping; approved source=3 allowed; all source=2 and source=4 ignored',
      'generated_semantic_loss':False,'pose_policy':'Provided RMX/OpenMVG poses fixed; poses are not re-estimated with sparse images. Historical track associations are proposals only; coordinates are recomputed from selected observations.'}
    write(out/'manifest.json',protocol);write(out/'classes.json',store.config)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(12,5));ax=fig.add_subplot(121,projection='3d');side=fig.add_subplot(122)
    for g in sorted(set(groups.values())):
        ids=np.array([r['region']==g for r in spatial]);ax.scatter(*xyz[ids].T,label=f'Region {g}')
        side.scatter(xyz[ids,0],xyz[ids,1],label=f'Region {g}')
    for v in train:side.annotate(str(v['original_train_index']),v['center'][:2],xytext=(3,3),textcoords='offset points',fontsize=7)
    ax.set(xlabel='X',ylabel='Y',zlabel='Z',title='27 selected camera centers / 3D');side.set(xlabel='X',ylabel='Y',title='Plan view / original training indices');side.set_aspect('equal');ax.legend();fig.tight_layout();fig.savefig(out/'camera_distribution.png',dpi=180);plt.close(fig)
    print('PROTOCOL',len(train),'training views',flush=True);return protocol

def build_p0(run,cfg):
    run=Path(run).resolve();protocol=prepare_protocol(run);out=run/'01_p0';out.mkdir(exist_ok=True)
    if (out/'outputs.json').exists():return read(out/'outputs.json')
    geo=geometry_settings(cfg);scene=read(geo['structure']);train=protocol['train'];by_name={v['name']:v for v in train}
    names={v['key']:Path(v['value']['ptr_wrapper']['data']['filename']).stem for v in scene['views']}
    features={n:load_features(geo['features'],n) for n in by_name};images={n:np.asarray(Image.open(c['image']).convert('RGB')) for n,c in by_name.items()}
    accepted=[];attempts=[];rejected=Counter();points=[];colors=[]
    for record in scene['structure']:
        selected=[o for o in record['value']['observations'] if names[o['key']] in by_name]
        if len(selected)<2:rejected['fewer than two selected observations']+=1;continue
        cams=[];pixels=[];obs=[]
        for o in selected:
            name=names[o['key']];fid=o['value']['id_feat'];uv=np.asarray(o['value']['x'])
            if fid>=len(features[name]['uv']) or np.linalg.norm(features[name]['uv'][fid]-uv)>.02:raise ValueError('Cached feature IDs mismatch')
            cams.append(by_name[name]);pixels.append(uv);obs.append({'view':name,'feature_id':fid,'uv':uv.tolist()})
        point,info=triangulate_track(cams,pixels,[(2048,1024)]*len(cams),geo)
        info.update(source_track=record['key'],observations=obs,real_views=len(cams),xyz=point.tolist() if point is not None and np.isfinite(point).all() else None)
        attempts.append(info)
        if info['reason']:rejected[info['reason']]+=1;continue
        if points and np.min(np.linalg.norm(np.array(points)-point,axis=1))<geo['duplicate_radius']:
            info['reason']='duplicate point';rejected[info['reason']]+=1;continue
        rgb=[images[o['view']][int(np.clip(round(o['uv'][1]),0,1023)),int(round(o['uv'][0]))%2048]/255 for o in obs]
        points.append(point);colors.append(np.median(rgb,axis=0));accepted.append((record,selected,info))
    xyz=np.asarray(points,dtype='f4');rgb=np.asarray(colors,dtype='f4');assert len(xyz)>100
    neighbors=cKDTree(xyz).query_ball_point(xyz,geo['outlier_radius'],return_length=True)-1;keep=neighbors>=geo['outlier_min_neighbors']
    for (_,_,info),k,n in zip(accepted,keep,neighbors):
        info['neighbors']=int(n)
        if not k:info['reason']='isolated point'
    rejected['isolated point']=int((~keep).sum());xyz=xyz[keep];rgb=rgb[keep];accepted=[a for a,k in zip(accepted,keep) if k]
    sparse=copy.deepcopy(scene);sparse['views']=[v for v in sparse['views'] if names[v['key']] in by_name]
    pose_ids={v['value']['ptr_wrapper']['data']['id_pose'] for v in sparse['views']};sparse['extrinsics']=[e for e in sparse['extrinsics'] if e['key'] in pose_ids];sparse['structure']=[]
    for (record,obs,info),point in zip(accepted,xyz):sparse['structure'].append({'key':record['key'],'value':{'X':point.tolist(),'observations':obs}})
    write(out/'structure_sparse27.json',sparse);write(out/'observations.json',[a[2] for a in accepted]);write(out/'triangulation.json',attempts)
    pcd_write(out/'points.ply',xyz,rgb);np.concatenate([xyz,rgb],1).astype('f4').tofile(out/'points.f32')
    nmap={n:i for i,n in enumerate(by_name)};matrix=np.zeros((len(train),len(train)),int)
    for _,_,info in accepted:
        for a,b in combinations(info['observations'],2):i,j=nmap[a['view']],nmap[b['view']];matrix[i,j]+=1;matrix[j,i]+=1
    ncomp,comp=connected_components(csr_matrix(matrix>0),directed=False)
    edges=[{'a':train[i]['name'],'b':train[j]['name'],'shared_retriangulated_tracks':int(matrix[i,j])} for i in range(27) for j in range(i) if matrix[i,j]]
    write(out/'connectivity.json',{'views':list(by_name),'shared_track_matrix':matrix.tolist(),'edges':edges,'components':int(ncomp),'component_ids':comp.tolist()})
    metrics={'points':len(xyz),'source_tracks':len(scene['structure']),'attempted_tracks':len(attempts),'rejected':dict(rejected),
      'observations_per_point':quantiles([len(a[2]['observations']) for a in accepted]),'max_reprojection_px_2k':quantiles([a[2]['max_error_px'] for a in accepted]),
      'parallax_deg':quantiles([a[2]['parallax_deg'] for a in accepted]),'camera_graph_components':int(ncomp),'source':ref(geo['structure']),
      'coordinate_policy':'Historical X discarded; every retained point solved anew with selected real camera rays, robust refinement, existing reprojection/parallax/density tests',
      'excluded_observation_views':sorted(set(names.values())-set(by_name))}
    write(out/'metrics.json',metrics);(out/'preview').mkdir(exist_ok=True);point_previews(out/'preview',xyz,np.empty((0,3)),train)
    result={'points':str(out/'points.f32'),'point_count':len(xyz),'ply':str(out/'points.ply'),'structure':str(out/'structure_sparse27.json'),'status':'completed'}
    write(out/'outputs.json',result);print('P0',metrics,flush=True);return result

def build_m1(run,cfg):
    from .m1_wan360 import augment
    from collections import defaultdict,deque
    import shutil
    run=Path(run).resolve();protocol=prepare_protocol(run);p0=build_p0(run,cfg)
    out=run/'01_sgi_sfm';out.mkdir(exist_ok=True);(out/'preview').mkdir(exist_ok=True)
    if (out/'outputs.json').exists():return read(out/'outputs.json')
    historical=ROOT/'runs/industrial_spcgs/m1_wan360_56_batch_20261023/01_sgi_sfm'
    all_videos=read(historical/'generated_videos.json');sources={v['name'] for v in protocol['train']}
    spatial={v['id']:v for v in protocol['spatial']};buckets=defaultdict(deque)
    # Round-robin source allocation across 3D regions and the two height layers.
    for row in protocol['spatial']:buckets[(row['region'],row['height_layer'])].append(row['id'])
    ordered=[]
    while any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key]:ordered.append(buckets[key].popleft())
    lookup={(v['source_training_view'],v['direction']):v for v in all_videos};videos=[]
    for name in ordered:
        for direction in ['forward','backward','left','right']:
            v=copy.deepcopy(lookup[(name,direction)]);v['frames']=[f for f in v['frames'] if f['frame_index'] in range(5)]
            assert [f['frame_index'] for f in v['frames']]==list(range(5))
            assert all(Path(f['image']).is_file() for f in v['frames']) and Path(v['video']).is_file()
            v.update(cache_reused=True,region=spatial[name]['region'],height_layer=spatial[name]['height_layer']);videos.append(v)
    assert len(videos)==108 and {v['source_training_view'] for v in videos}==sources
    write(out/'reuse_provenance.json',{'historical_run':str(historical),'videos_generated':0,'video_count':108,'frames':[1,2,3,4],
      'source_order':ordered,'selection':'One registered frame per source/direction; maximize novel tracks passing unchanged geometry tests, then registration score',
      'generated_semantic_loss':False,'real_support_views':sorted(sources)})
    cfg=copy.deepcopy(cfg);cfg['m1_geometry']={**cfg.get('m1_geometry',{}),'structure':p0['structure']}
    cfg['wan360']={**cfg.get('wan360',{}),'candidate_frames':[1,2,3,4]};cfg['selection_by_novel_points']=True;cfg['generated_semantic_loss']=False
    array=np.fromfile(p0['points'],dtype='f4').reshape(-1,6);shutil.copy2(p0['ply'],out/'original.ply')
    existing={(o['view'],o['feature_id']) for t in read(run/'01_p0/observations.json') for o in t['observations']}
    write(out/'config.json',cfg)
    result,metrics=augment(out,cfg,protocol['train'],array[:,:3],array[:,3:],cached_videos=videos,existing_features=existing)
    selected=read(out/'selected_views.json');by_gen={v['name']:v for v in selected};contribution=Counter(r['generated_view'] for r in read(out/'observations.json'))
    detail=[]
    for v in selected:
        item=spatial[v['source_training_view']]
        detail.append({k:v[k] for k in ['name','source_training_view','direction','frame_index','inliers','novel_support_points']})
        detail[-1].update(added_points=contribution[v['name']],region=item['region'],height_layer=item['height_layer'])
    write(out/'contributions.json',detail)
    metrics['contribution_by_direction']={d:sum(r['added_points'] for r in detail if r['direction']==d) for d in ['forward','backward','left','right']}
    write(out/'metrics.json',metrics);result['status']='completed';write(out/'outputs.json',result)
    print('M1 COMPLETE',metrics,flush=True);return result

def train(run,cfg,kind):
    from .common import native,signature
    from .pipeline import base_request
    from .features import init_semantics
    run=Path(run).resolve();protocol=prepare_protocol(run);cache=read(run/'03_semantic/cache/manifest.json')
    cfg={**cfg,'classes':cache['classes'],'cache_height':128,'cache_width':256,'semantic_height':128,'semantic_width':256}
    cfg.update(train_names=[v['name'] for v in protocol['train']],feature_backend='official_prototypes',official_run=str(run),generated_semantic_loss=False)
    if kind=='sgi':write(run/'sgi_config.json',cfg)
    assert {v['name'] for v in cache['views']}=={v['name'] for v in protocol['train']} and len(cache['classes'])==7
    folder=run/{'short':'02_sgi_short','sgi':'02_sgi_gaussians','rgb':'02_rgb_p0_reference'}[kind];folder.mkdir(exist_ok=True)
    if (folder/'final/model.pt').exists():return folder
    pointinfo=read(run/('01_p0' if kind=='rgb' else '01_sgi_sfm')/'outputs.json')
    xyz=np.fromfile(pointinfo['points'],dtype='f4').reshape(-1,6)[:,:3]
    initial=init_semantics(xyz,cache,folder/'initial_semantic.f32');center=np.asarray([v['center'] for v in protocol['train']])
    request={**base_request(cfg),'action':'train','output':str(folder),'views':cache['views'],'text':cache['text'],'decoder':cache['decoder'],
      'points':pointinfo['points'],'point_count':len(xyz),'initial_semantic':initial,'generated_views':[],
      'extent':float(np.linalg.norm(center-center.mean(0),axis=1).max()*1.1),'semantic_supervision':kind!='rgb','ogr':kind!='rgb',
      'spc2d':False,'spc3d':False,'iterations':400 if kind=='short' else 10000,'save_every':400 if kind=='short' else 1000}
    request['signature']=signature({k:v for k,v in request.items() if k!='output'})
    write(folder/'inputs.json',{'protocol':ref(run/'00_protocol/manifest.json'),'teacher':ref(run/'03_semantic/cache/manifest.json'),'points':ref(pointinfo['points']),
       'scope':'SGI only; no SPC' if kind!='rgb' else 'RGB-only P0 reference; same cameras, resolution and 10000 steps; no semantic loss or OGR'})
    native(request,folder);write(folder/'outputs.json',{'status':'completed','iterations':request['iterations'],'model':str(folder/'final/model.pt'),'ply':str(folder/'final/point_cloud.ply')})
    print('TRAIN COMPLETE',kind,flush=True);return folder
