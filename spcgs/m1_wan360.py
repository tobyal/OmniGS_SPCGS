"""Wan360 -> nearby frames -> registered spherical SfM -> SGI input."""
from collections import Counter
from pathlib import Path
import copy
import os
import shutil
import numpy as np
import cv2
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from .common import ROOT, DATA, read, write, ref, sha, signature, run_command, pcd_write
from .wan360 import generate, settings
from .m1_geometry import (load_anchors, load_features, match_features, register,
                          triangulate_track, epipolar_mask, project_erp, quantiles)

def geometry_settings(cfg):
    result = {
        'structure':str(ROOT.parent/'datasets/iris_xphase1_work_2k/mvg/structure_filtered.json'),
        'features':str(ROOT.parent/'datasets/iris_xphase1_work_2k/mvg'),
        'openmvg_bin':str(ROOT.parent/'deps/openmvg-build/Linux-x86_64-Release'),
        'neighbors':6, 'ratio':.8, 'mutual':True, 'seed':cfg['seed'],
        'min_correspondences':24, 'min_inliers':16, 'min_inlier_ratio':.3,
        'registration_error_px':6., 'min_validation_inliers':4, 'min_validation_ratio':.3,
        'min_coverage_bins':4, 'min_translation':.02, 'max_translation':2.,
        'max_rotation_deg':15., 'min_validation_improvement_px':.3,
        'epipolar_error_px':6., 'triangulation_error_px':5., 'min_parallax_deg':.6,
        'min_real_views':2, 'duplicate_radius':.015, 'max_distance_to_original':2.,
        'outlier_radius':.3, 'outlier_min_neighbors':2,
        'min_rgb_difference':.003, 'min_sharpness_ratio':.15,
    }
    result.update(cfg.get('m1_geometry',{}))
    for key in ['structure','features','openmvg_bin']:
        path=Path(result[key]).expanduser()
        result[key]=str(path if path.is_absolute() else ROOT/path)
    return result

def extract_candidates(folder, cfg, videos):
    """OpenMVG uses the same saved SIFT/RootSIFT describer as original tracks."""
    geo = geometry_settings(cfg)
    feature_dir = folder/'features'
    images = feature_dir/'images'
    images.mkdir(parents=True,exist_ok=True)
    cache = ROOT/'runs/m1_cache/features'
    cache.mkdir(parents=True,exist_ok=True)
    describer = Path(geo['features'])/'image_describer.json'
    shutil.copy2(describer,feature_dir/'image_describer.json')
    template = read(DATA/'openMVG/data_openmvg_train.json')
    view_template=template['views'][0]
    scene = {**template,'root_path':str(images),'views':[],'extrinsics':[],'structure':[],'control_points':[]}
    scene['intrinsics']=copy.deepcopy(template['intrinsics'][:1])
    candidates=[]
    for video in videos:
        for frame in video['frames']:
            if frame['frame_index']==0:
                continue
            name=video['name']+f"_f{frame['frame_index']:04d}"
            source=Path(frame['image'])
            w,h=Image.open(source).size
            dest=images/(name+'.png')
            if dest.is_symlink() and dest.resolve()!=source.resolve():
                raise ValueError(f'Cached candidate points at a different image: {dest}')
            if not dest.exists():dest.symlink_to(source.resolve())
            key=signature({'image':ref(source),'describer':ref(describer),'openmvg':ref(Path(geo['openmvg_bin'])/'openMVG_main_ComputeFeatures')})
            cached=cache/key
            cached.mkdir(exist_ok=True)
            cache_complete=all((cached/('features'+suffix)).is_file() for suffix in ['.feat','.desc'])
            for suffix in ['.feat','.desc']:
                if cache_complete and not (feature_dir/(name+suffix)).exists():
                    (feature_dir/(name+suffix)).symlink_to(cached/('features'+suffix))
            index=len(candidates)
            node=copy.deepcopy(view_template)
            node['key']=index
            node['value']['ptr_wrapper']['id']=2147485000+index
            node['value']['ptr_wrapper']['data'].update(filename=name+'.png',width=w,height=h,id_view=index,id_pose=index,id_intrinsic=0)
            scene['views'].append(node)
            candidates.append({**frame,'name':name,'video_name':video['name'],
                'source_training_view':video['source_training_view'],'direction':video['direction'],
                'width':w,'height':h,'cache':str(cached)})
    if not candidates:
        raise ValueError('No generated candidate frames')
    sizes={(c['width'],c['height']) for c in candidates}
    if len(sizes)!=1:
        raise ValueError('This feature batch requires a consistent generated ERP size')
    w,h=next(iter(sizes))
    scene['intrinsics'][0]['value']['ptr_wrapper']['data']['value0'].update(width=w,height=h)
    write(feature_dir/'views.json',scene)
    env=os.environ.copy();env['OMP_NUM_THREADS']='4'
    run_command([Path(geo['openmvg_bin'])/'openMVG_main_ComputeFeatures','-i',feature_dir/'views.json',
                 '-o',feature_dir,'-m','SIFT','-p','NORMAL','-n','4'],folder/'openmvg_features.log',env)
    for c in candidates:
        for suffix in ['.feat','.desc']:
            target=Path(c['cache'])/('features'+suffix)
            if not target.exists():
                temporary=target.with_name(target.name+f'.{os.getpid()}.tmp')
                shutil.copy2(feature_dir/(c['name']+suffix),temporary);temporary.replace(target)
        c['features']=load_features(feature_dir,c['name'])
    return candidates

def frame_quality(candidate, source):
    image=np.asarray(Image.open(candidate['image']).convert('RGB'))
    real=np.asarray(Image.open(source['image']).convert('RGB').resize((candidate['width'],candidate['height']),Image.Resampling.LANCZOS))
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    realgray=cv2.cvtColor(real,cv2.COLOR_RGB2GRAY)
    sharpness=float(cv2.Laplacian(gray,cv2.CV_64F).var())
    reference=float(cv2.Laplacian(realgray,cv2.CV_64F).var())
    return {'sharpness':sharpness,'source_sharpness':reference,'sharpness_ratio':sharpness/max(reference,1e-8),
            'rgb_difference':float(np.abs(image.astype(float)-real).mean()/255.)}

def preview_matches(folder, source, generated, real_uv, gen_uv, inliers):
    left=Image.open(source['image']).convert('RGB').resize((768,384))
    right=Image.open(generated['image']).convert('RGB').resize((768,384))
    canvas=Image.new('RGB',(1536,408),'white');canvas.paste(left,(0,24));canvas.paste(right,(768,24))
    draw=ImageDraw.Draw(canvas)
    draw.text((8,5),'Real training image | green: geometric inlier; red: rejected',fill='black')
    draw.text((776,5),f"Wan360 {generated['direction']} frame {generated['frame_index']}",fill='black')
    order=np.random.default_rng(0).permutation(len(real_uv))[:80]
    for i in order:
        a=real_uv[i]*[768/2048,384/1024]+[0,24]
        b=gen_uv[i]*[768/generated['width'],384/generated['height']]+[768,24]
        color=(40,210,80) if inliers[i] else (235,70,60)
        draw.line([tuple(a),tuple(b)],fill=color,width=1)
    canvas.save(folder/(generated['name']+'_matches.png'))

def point_previews(folder, original, added, train):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,(a,b,label) in zip(axes,[(0,1,'World XY'),(0,2,'World XZ'),(1,2,'World YZ')]):
        ax.scatter(original[:,a],original[:,b],s=1,c='#377eb8',label=f'Original ({len(original)})')
        if len(added):ax.scatter(added[:,a],added[:,b],s=3,c='#e41a1c',label=f'New ({len(added)})')
        ax.set_title(label);ax.set_aspect('equal');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(folder/'point_distribution.png',dpi=160);plt.close(fig)
    unique = list({camera['name']:camera for camera in train}.values())
    for camera in unique[:4]:
        im=Image.open(camera['image']).convert('RGB').resize((1024,512))
        draw=ImageDraw.Draw(im)
        for points,color,size in [(original,(55,126,184),1),(added,(240,50,40),2)]:
            if not len(points):continue
            uv=project_erp(points,camera,1024,512)
            for u,v in uv:
                if np.isfinite([u,v]).all():
                    x,y=int(round(u))%1024,int(np.clip(round(v),0,511))
                    draw.ellipse((x-size,y-size,x+size,y+size),fill=color)
        draw.rectangle((0,0,650,22),fill='white')
        draw.text((8,4),'Blue: original SfM; red: new validated points. Projection does not imply visibility.',fill='black')
        im.save(folder/(camera['name']+'_new_points.png'))

def report(folder, metrics, reason=None):
    rows=['# M1 Wan360 / spherical SfM report','',f"Status: {'failed' if reason else 'completed'}",'']
    if reason:rows += [f'Failure: {reason}','']
    rows += ['Real cameras remain fixed in the native RMX/OpenMVG world. Wan360 trajectories are conditioning only.',
             'SIFT/RootSIFT and training-only landmarks come from OpenMVG. Pose estimation and triangulation use spherical rays; ERP images are not treated as pinhole images.',
             'Validation correspondences are withheld from pose fitting. No semantic GT, monocular depth or generated RGB supervision is used.','',
             '```json',__import__('json').dumps(metrics,indent=2),'```','',
             '- `wan360_provenance.json`, `generation_jobs.json`, `generated_videos.json`: model/source/video/trajectory provenance.',
             '- `candidate_scores.json`, `registration/*.json`: frame indices, quality, fitted poses and held-out errors.',
             '- `matches/*.npz`, `matching_statistics.json`: descriptor matches and spherical epipolar decisions.',
             '- `selected_views.json`, `invalid_views.json`: selected and rejected frames.',
             '- `observations.json`, `triangulation.json`: point tracks and all filtering decisions.',
             '- `original.ply`, `new_points.ply`, `enhanced.ply`, `enhanced_points.f32`: point-cloud outputs when successful.',
             '- `preview/`: real/generated frames, matches, point distributions and fixed-view projections.','',
             'Only accepted generated views enter downstream CLIP supervision. Generated RGB never enters the RGB reconstruction loss.',
             'A successful small run does not establish correctness of all generated geometry; inspect correspondence and coverage previews before scaling.']
    (folder/'report.md').write_text('\n'.join(rows)+'\n')

def novel_support(candidate, anchors, geo, tree, existing):
    """Rank registered frames by tracks passing the existing geometry tests."""
    tracks={};points=[]
    for name,(pairs,_) in candidate['matches'].items():
        uv=anchors['features'][name]['uv']
        mask,_=epipolar_mask(anchors['cameras'][name],candidate,uv[pairs[:,0]],candidate['features']['uv'][pairs[:,1]],geo)
        for ri,gi in pairs[mask]:tracks.setdefault(int(gi),[]).append((name,int(ri)))
    for gi,links in tracks.items():
        if len(links)<geo['min_real_views'] or any(link in existing for link in links):continue
        cams=[anchors['cameras'][n] for n,_ in links]+[candidate]
        pixels=[anchors['features'][n]['uv'][i] for n,i in links]+[candidate['features']['uv'][gi]]
        point,info=triangulate_track(cams,pixels,[(2048,1024)]*len(links)+[(candidate['width'],candidate['height'])],geo)
        if info['reason']:continue
        distance=tree.query(point)[0]
        if not geo['duplicate_radius']<=distance<=geo['max_distance_to_original']:continue
        if points and np.min(np.linalg.norm(np.asarray(points)-point,axis=1))<geo['duplicate_radius']:continue
        points.append(point)
    return len(points)

def augment(folder,cfg,train,original_xyz,original_rgb,cached_videos=None,existing_features=None):
    geo=geometry_settings(cfg)
    write(folder/'geometry_config.json',geo)
    anchors=load_anchors(geo['structure'],geo['features'],train,original_xyz,folder/'anchor_audit.json')
    # An explicit cache list is a strict reuse path: it never calls generation.
    videos=generate(folder,cfg,train) if cached_videos is None else cached_videos
    existing=set(anchors['by_feature']) if existing_features is None else set(existing_features)
    original_tree=cKDTree(original_xyz)
    if cached_videos is not None:
        assert all(v['source_training_view'] in anchors['cameras'] and v['cache_reused'] for v in videos)
        write(folder/'generated_videos.json',videos)
    # This preview survives complete registration failure and makes generated
    # structure changes inspectable without confusing them with accepted views.
    preview_videos=videos[:8]
    columns=1+min(6,len(settings(cfg)['candidate_frames']))
    sheet=Image.new('RGB',(columns*256,30+len(preview_videos)*150),'white')
    draw=ImageDraw.Draw(sheet)
    draw.text((5,5),'Real input and generated candidates; geometry acceptance is reported separately.',fill='black')
    for row,video in enumerate(preview_videos):
        frames=[f for f in video['frames'] if f['frame_index']>0][:6]
        items=[(anchors['cameras'][video['source_training_view']]['image'],'Real / '+video['direction'])]
        items += [(f['image'],f"Frame {f['frame_index']}") for f in frames]
        for col,(path,label) in enumerate(items):
            x,y=col*256,30+row*150
            draw.text((x+4,y+2),label,fill='black')
            sheet.paste(Image.open(path).convert('RGB').resize((256,128)),(x,y+20))
    sheet.save(folder/'preview/generation_candidates.png')
    metrics={'original_count':len(original_xyz),'videos':len(videos),'generated_sources':len({v['source_training_view'] for v in videos}),
             'videos_generated_this_invocation':sum(not v['cache_reused'] for v in videos),
             'videos_reused':sum(v['cache_reused'] for v in videos),
             'registered_views':0,'selected_views':0,'added':0,'enhanced_count':None,
             'backend':'OpenMVG RootSIFT observations/descriptors; spherical 2D-3D localization and multiview ray triangulation'}
    write(folder/'metrics.json',metrics)
    candidates=extract_candidates(folder,cfg,videos)
    by_name=anchors['cameras'];centers=np.array([v['center'] for v in train])
    scores=[];registered=[];matching_stats=[]
    (folder/'registration').mkdir(exist_ok=True);(folder/'matches').mkdir(exist_ok=True)
    match_cache=ROOT/'runs/m1_cache/matches';match_cache.mkdir(parents=True,exist_ok=True)
    matcher_version=ref(ROOT/'spcgs/m1_geometry.py')
    cv2.setNumThreads(4)
    for candidate in candidates:
        source=by_name[candidate['source_training_view']]
        neighbors=[train[i] for i in np.argsort(np.linalg.norm(centers-source['center'],axis=1))[:geo['neighbors']]]
        matches={}
        for real in neighbors:
            features=anchors['features'][real['name']]
            cache_key=signature({'real':features['fingerprint'],'generated':candidate['features']['fingerprint'],
                                 'ratio':geo['ratio'],'mutual':geo['mutual'],'seed':geo['seed'],
                                 'opencv':cv2.__version__,'implementation':matcher_version})
            cached=match_cache/(cache_key+'.npz')
            if cached.exists():
                with np.load(cached) as saved:pairs,ratios=saved['pairs'],saved['ratio']
            else:
                cv2.setRNGSeed(geo['seed'])
                pairs,ratios=match_features(features,candidate['features'],geo['ratio'],geo['mutual'])
                temporary=cached.with_suffix(f'.{os.getpid()}.tmp.npz')
                np.savez_compressed(temporary,pairs=pairs,ratio=ratios);temporary.replace(cached)
            matches[real['name']]=(pairs,ratios)
            np.savez_compressed(folder/'matches'/(candidate['name']+'__'+real['name']+'.npz'),pairs=pairs,ratio=ratios)
            matching_stats.append({'candidate':candidate['name'],'real_view':real['name'],'matches':len(pairs),'cache':str(cached)})
        quality=frame_quality(candidate,source)
        pose=register(candidate,source,anchors,matches,geo,folder/'registration'/(candidate['name']+'.json'))
        row={k:v for k,v in candidate.items() if k!='features'}
        row.update(quality);row.update(pose)
        if quality['rgb_difference']<geo['min_rgb_difference']:
            row.update(registered=False,reason='near-copy of original image')
        if quality['sharpness_ratio']<geo['min_sharpness_ratio']:
            row.update(registered=False,reason='excessive blur relative to source at the same resolution')
        row['score']=float(row.get('inliers',0)-2*row.get('error_px',{}).get('median',10)-.3*row['frame_index'])
        scores.append(row)
        if row['registered']:
            candidate.update(row);candidate['matches']=matches
            if cfg.get('selection_by_novel_points',False):
                row['novel_support_points']=novel_support(candidate,anchors,geo,original_tree,existing)
                candidate['novel_support_points']=row['novel_support_points']
            registered.append(candidate)
        write(folder/'candidate_scores.json',scores)
        write(folder/'matching_statistics.json',matching_stats)
        metrics.update(candidate_frames=len(candidates),processed_candidates=len(scores),
                       registered_candidates=len(registered),descriptor_matches=sum(s['matches'] for s in matching_stats))
        write(folder/'metrics.json',metrics)
        print('M1 REGISTER',candidate['name'],row['registered'],row.get('inliers',0),row.get('reason'),flush=True)
    metrics.update(candidate_frames=len(candidates),registered_candidates=len(registered),
                   descriptor_matches=sum(s['matches'] for s in matching_stats))
    write(folder/'matching_statistics.json',matching_stats)
    write(folder/'invalid_views.json',[r for r in scores if not r['registered']])
    selected=[]
    for video in videos:
        valid=[v for v in registered if v['video_name']==video['name']]
        if valid:
            chosen=max(valid,key=lambda row:(row.get('novel_support_points',0),row['score']))
            selected_dir=folder/'selected';selected_dir.mkdir(exist_ok=True)
            destination=selected_dir/(chosen['name']+'.png')
            shutil.copy2(chosen['image'],destination)
            chosen['source_candidate_image']=chosen['image']
            chosen['image']=str(destination)
            selected.append(chosen)
    metrics['registered_views']=metrics['selected_views']=len(selected)
    write(folder/'selected_views.json',[{k:v for k,v in c.items() if k not in ['features','matches']} for c in selected])
    if not selected:
        write(folder/'metrics.json',metrics)
        raise RuntimeError('No Wan360 frame passed independent geometric registration and frame-quality checks')
    points=[];colors=[];observations=[];attempts=[];rejected=Counter();used_features=set()
    original_tree=cKDTree(original_xyz)
    for candidate in selected:
        source=by_name[candidate['source_training_view']]
        side=Image.new('RGB',(1024,1050),'white')
        side.paste(Image.open(source['image']).convert('RGB').resize((1024,512)),(0,0))
        side.paste(Image.open(candidate['image']).convert('RGB').resize((1024,512)),(0,536))
        ImageDraw.Draw(side).text((5,516),f"Real above / Wan360 {candidate['direction']} frame {candidate['frame_index']} below",fill='black')
        side.save(folder/'preview'/(candidate['name']+'_real_generated.png'))
        tracks={}
        for name,(pairs,ratio) in candidate['matches'].items():
            real=by_name[name];rf=anchors['features'][name]
            mask,ep_error=epipolar_mask(real,candidate,rf['uv'][pairs[:,0]],candidate['features']['uv'][pairs[:,1]],geo)
            write(folder/'matches'/(candidate['name']+'__'+name+'_geometry.json'),
                  {'matches':len(pairs),'inliers':int(mask.sum()),'spherical_epipolar_px_2k':quantiles(ep_error[mask])})
            np.savez_compressed(folder/'matches'/(candidate['name']+'__'+name+'_geometry.npz'),pairs=pairs,inliers=mask,error=ep_error)
            metrics['geometric_matches']=metrics.get('geometric_matches',0)+int(mask.sum())
            if name==source['name']:
                preview_matches(folder/'preview',real,candidate,rf['uv'][pairs[:,0]],candidate['features']['uv'][pairs[:,1]],mask)
            for ri,gi in pairs[mask]:
                tracks.setdefault(int(gi),[]).append((name,int(ri)))
        real_images={name:np.asarray(Image.open(by_name[name]['image']).convert('RGB')) for name in candidate['matches']}
        for gi,links in tracks.items():
            if len(links)<geo['min_real_views']:
                rejected['insufficient real views']+=1;continue
            if any(link in existing for link in links):
                rejected['already observed original landmark']+=1;continue
            if any(link in used_features for link in links):
                rejected['duplicate real feature track']+=1;continue
            cams=[by_name[name] for name,_ in links]+[candidate]
            pixels=[anchors['features'][name]['uv'][ri] for name,ri in links]+[candidate['features']['uv'][gi]]
            sizes=[(2048,1024)]*len(links)+[(candidate['width'],candidate['height'])]
            point,info=triangulate_track(cams,pixels,sizes,geo)
            record={**info,'generated_view':candidate['name'],'generated_feature_id':gi,
                    'real_observations':[{'view':n,'feature_id':i,'uv':uv.tolist()} for (n,i),uv in zip(links,pixels)],
                    'generated_uv':pixels[-1].tolist(),'xyz':point.tolist() if point is not None and np.isfinite(point).all() else None}
            attempts.append(record)
            if info['reason']:
                rejected[info['reason']]+=1;continue
            distance=float(original_tree.query(point)[0]);record['distance_to_original']=distance
            if distance<geo['duplicate_radius']:
                record['reason']='duplicate original point';rejected[record['reason']]+=1;continue
            if distance>geo['max_distance_to_original']:
                record['reason']='far from original scene support';rejected[record['reason']]+=1;continue
            if points and np.min(np.linalg.norm(np.asarray(points)-point,axis=1))<geo['duplicate_radius']:
                record['reason']='duplicate new point';rejected[record['reason']]+=1;continue
            rgb=[]
            for (name,_),uv in zip(links,pixels):
                image=real_images[name];rgb.append(image[int(np.clip(round(uv[1]),0,1023)),int(round(uv[0]))%2048]/255.)
            points.append(point);colors.append(np.median(rgb,axis=0));observations.append(record);used_features.update(links)
    metrics.update(triangulated_candidates=len(attempts),rejected=dict(rejected),before_density_filter=len(points))
    added=np.asarray(points,dtype='f4').reshape(-1,3);rgb=np.asarray(colors,dtype='f4').reshape(-1,3)
    if len(added):
        # Density check can use original or other new points, excluding the point itself.
        all_tree=cKDTree(np.concatenate([original_xyz,added]))
        counts=all_tree.query_ball_point(added,geo['outlier_radius'],return_length=True)-1
        keep=counts>=geo['outlier_min_neighbors']
        metrics['density_removed']=int((~keep).sum())
        for rec,count,yes in zip(observations,counts,keep):
            rec['neighbors']=int(count)
            if not yes:rec['reason']='isolated point'
        added=added[keep];rgb=rgb[keep];observations=[r for r,k in zip(observations,keep) if k]
    write(folder/'triangulation.json',attempts);write(folder/'observations.json',observations)
    metrics.update(added=len(added),reprojection_px=quantiles([r['max_error_px'] for r in observations]),
                   parallax_deg=quantiles([r['parallax_deg'] for r in observations]))
    pcd_write(folder/'new_points.ply',added,rgb)
    point_previews(folder/'preview',original_xyz,added,[by_name[v['source_training_view']] for v in videos]+train[:4])
    if not len(added):
        write(folder/'metrics.json',metrics)
        raise RuntimeError('Registered Wan360 frames produced no reliable new points after geometric filtering')
    xyz=np.concatenate([original_xyz,added]);merged_rgb=np.concatenate([original_rgb,rgb])
    pcd_write(folder/'enhanced.ply',xyz,merged_rgb)
    np.concatenate([xyz,merged_rgb],1).astype('f4').tofile(folder/'enhanced_points.f32')
    assert np.array_equal(xyz[:len(original_xyz)],original_xyz)
    metrics['enhanced_count']=len(xyz)
    metrics['new_point_distance_to_original']=quantiles(original_tree.query(added)[0])
    exported=[]
    for c in selected:
        # Plain camera records are consumed unchanged by M3 cache and M2 semantics.
        exported.append({k:c[k] for k in ['name','image','rotation','center','source_training_view']})
        exported[-1].update(projection='equirectangular',registered=True,split='generated',
                            registration_evidence=str(folder/'registration'/(c['name']+'.json')),
                            frame_index=c['frame_index'],video_name=c['video_name'])
    write(folder/'registered_generated_views.json',{'coordinate_system':'rmx_openmvg_world','views':exported})
    write(folder/'reconstruction.json',{'schema':'spherical_fixed_real_sfm_v1',
          'coordinate_system':'native_rmx_openmvg_world','real_cameras':train,'generated_cameras':exported,
          'original_points':str(folder/'original.ply'),'new_landmarks':observations,
          'projection':'OpenMVG spherical; integer pixel coordinates; horizontal periodic boundary'})
    report(folder,metrics)
    return {'points':str(folder/'enhanced_points.f32'),'point_count':len(xyz),'source':'Wan360 enhanced SfM',
            'original_ply':str(folder/'original.ply'),'new_ply':str(folder/'new_points.ply'),
            'enhanced_ply':str(folder/'enhanced.ply'),'enhanced_points':str(folder/'enhanced_points.f32'),
            'registered_generated_views':exported,'report':str(folder/'report.md')},metrics
