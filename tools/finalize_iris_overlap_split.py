"""Freeze the compact 18/6 choice and validate with training-only 3D structure."""
import collections
import csv
from pathlib import Path
import shutil
import sys
import numpy as np
from scipy.spatial import ConvexHull

sys.path.insert(0,str(Path(__file__).resolve().parent))
from reselect_iris_interpolation import Diagnostics,OUT,RUN,ROOT,ANALYSIS,read,write,sha,xyz,run_command,BIN
from analyze_iris_52_53_replacements import subset_matches
from run_omnigs_sparse18 import export_cloud,ref


def heldout_structure_support(d,structure_file,tests):
    structure=read(structure_file)['structure'];out=[]
    for t in tests:
        it=d.index[t];R=np.array(d.poses[it]['rotation']);C=np.array(d.poses[it]['center'])
        maps={i:dict(d.matches.get((i,it),np.empty((0,2),int))) for i in d.poses if i!=it}
        candidates={}
        for lm in structure:
            observations=lm['value']['observations']
            assert all(o['key']!=it for o in observations)
            votes=collections.defaultdict(set)
            for obs in observations:
                target=maps[obs['key']].get(obs['value']['id_feat'])
                if target is not None:votes[int(target)].add(obs['key'])
            if not votes:continue
            X=np.array(lm['value']['X']);ray=X-C;depth=np.linalg.norm(ray)
            if not .15<depth<200:continue
            ray/=depth
            for feat,cameras in votes.items():
                if len(cameras)<2:continue
                truth=d.rays[it][feat]
                error=float(np.rad2deg(np.arctan2(np.linalg.norm(np.cross(ray,truth)),ray@truth)))
                if error>.35:continue
                record={'test_feature_id':feat,'landmark_id':lm['key'],'matching_train_cameras':[d.rows[i]['camera_id'] for i in sorted(cameras)],
                        'training_track_views':len(observations),'angular_error_deg':error}
                if feat not in candidates or (len(cameras),-error)>(len(candidates[feat]['matching_train_cameras']),-candidates[feat]['angular_error_deg']):
                    candidates[feat]=record
        ids=sorted(candidates)
        out.append({'test':t,'validated_train_only_landmark_correspondences':len(ids),
                    'median_angular_error_deg':float(np.median([candidates[i]['angular_error_deg'] for i in ids])) if ids else None,
                    'occupancy':d.occupancy(t,ids),'correspondences':[candidates[i] for i in ids]})
    return out


def main():
    d=Diagnostics();design=read(OUT/'design_comparison.json')['compact']
    trains=sorted(design['train'],key=lambda x:d.by[x]['group_index'])
    tests=sorted(design['tests'],key=lambda x:d.by[x]['group_index'])
    assert len(trains)==18 and len(tests)==6 and not set(trains)&set(tests)
    assert all(d.by[x]['group']==1 for x in trains+tests)
    assert sum(d.by[x]['height_band_relative_to_core']=='upper' for x in trains)==9
    assert sum(d.by[x]['height_band_relative_to_core']=='upper' for x in tests)==3
    data,mvg=OUT/'selected_dataset',OUT/'selected_sfm'
    mvg.mkdir(exist_ok=False);(data/'openMVG').mkdir(parents=True)
    for split,ids in [('train',trains),('test',tests)]:
        images=data/(split+'_images');images.mkdir()
        indices={d.index[c] for c in ids};scene=read(OUT/'pool_sfm/sfm_data_train_fixed.json')
        scene['views']=[v for v in scene['views'] if v['key'] in indices]
        scene['extrinsics']=[p for p in scene['extrinsics'] if p['key'] in indices]
        scene['root_path']=str(images)
        for cid in ids:
            source=(OUT/'pool_images'/(d.by[cid]['image_id']+'.png')).resolve()
            (images/source.name).symlink_to(source)
        write(data/f'openMVG/data_openmvg_{split}.json',scene)
        (OUT/f'{split}{len(ids)}.txt').write_text('\n'.join(d.by[x]['image_id'] for x in ids)+'\n')
    shutil.copy2(data/'openMVG/data_openmvg_train.json',mvg/'sfm_data_train_fixed.json')
    subset_matches(OUT/'pool_sfm/matches.fixed_pose.txt',mvg/'matches.fixed_pose.txt',{d.index[x] for x in trains})
    run_command([BIN/'openMVG_main_ComputeStructureFromKnownPoses','-i',mvg/'sfm_data_train_fixed.json','-m',OUT/'pool_sfm',
                 '-f',mvg/'matches.fixed_pose.txt','-d','-r','4','-o',mvg/'structure.json'],OUT/'selected_triangulation.log')
    # Export requires mvg.parent/sfm_quality.json and data/train_images; observations are training-only.
    export_cloud(mvg,data)
    selected=read(mvg/'structure_filtered.json')
    allowed={d.index[x] for x in trains}
    assert all(o['key'] in allowed for p in selected['structure'] for o in p['value']['observations'])
    support=heldout_structure_support(d,mvg/'structure_filtered.json',tests)
    write(OUT/'test_to_train_structure_support.json',support)
    common=sorted(set(tests)&{x['test'] for x in read(OUT/'old_split_test_support.json')})
    old_support=heldout_structure_support(d,RUN/'sfm/structure_filtered.json',common)
    write(OUT/'old_common_test_structure_support.json',old_support)
    new_support=[d.test_support(t,trains) for t in tests]
    write(OUT/'selected_test_feature_support.json',new_support)
    p=np.array([xyz(d.by[x]) for x in trains]);old=read(RUN/'approved_split.json')['selected_cameras']
    oldp=np.array([xyz(x) for x in old if x['candidate_role']=='train']);h=ConvexHull(p[:,:2])
    audit=[]
    for t in tests:
        b=next(x for x in design['brackets'] if x['test']==t)
        audit.append({k:b[k] for k in ['test','before','after','segment_parameter','segment_offset_m','bracket_angle_deg',
                     'distance_before_m','distance_after_m','max_abs_dz_m','matches_before','matches_after',
                     'train_pair_reprojection_supported_features','median_reprojection_error_deg']})
    selected_rows=[]
    for split,ids in [('train',trains),('test',tests)]:
        for i,cid in enumerate(ids,1):
            row=dict(d.by[cid],split=split,split_index=i,pool_view_id=d.index[cid])
            row['selection_role']='trajectory_anchor' if cid in design['anchors'] else 'appearance_support' if split=='train' else 'heldout_trajectory_interpolation'
            selected_rows.append(row)
    manifest={'status':'selected_for_review_no_training','train_ids':trains,'test_ids':tests,'validation_ids':[],
        'selected_cameras':selected_rows,'test_brackets':audit,'train_official':sum(d.by[x]['has_official_semantics'] for x in trains),
        'test_official':sum(d.by[x]['has_official_semantics'] for x in tests),'resolution':[2048,1024],
        'train_height_counts':{'upper':9,'lower':9},'test_height_counts':{'upper':3,'lower':3},
        'xy_bounds_m':{'min':p[:,:2].min(0).tolist(),'max':p[:,:2].max(0).tolist()},
        'xy_hull_area_m2':float(h.volume),'old_xy_hull_area_m2':float(ConvexHull(oldp[:,:2]).volume),
        'test_images_used_for_split_design_matching':True,'test_observations_used_for_initialization':0,
        'initial_cloud':ref(data/'openMVG/scene.ply'),'initial_cloud_quality':read(OUT/'sfm_quality.json'),
        'missing_official_train_ids':[x for x in trains if not d.by[x]['has_official_semantics']],
        'source_inventory':ref(ANALYSIS/'camera_inventory.json'),'coordinate_transform':read(RUN/'dataset/coordinate_transform.json'),
        'scripts':[ref(ROOT/'tools'/f) for f in ['reselect_iris_interpolation.py','finalize_iris_overlap_split.py']],
        'selection_note':'A compact connected industrial corridor is prioritized over broad coverage. Twelve immediate same-height trajectory anchors are fixed; six additional views improve measured test-feature support. This is a geometry-informed selected split, not a random blind benchmark.'}
    write(OUT/'selected_split.json',manifest)
    fields=['split','camera_id','image_id','source_group','x_m','y_m','z_m','height_band_relative_to_core','has_official_semantics','selection_role','pool_view_id']
    with (OUT/'selected_cameras.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(selected_rows)
    # Check train and test directories and SfM observations independently.
    assert len(list((data/'train_images').glob('*.png')))==18 and len(list((data/'test_images').glob('*.png')))==6
    write(OUT/'verification.json',{'training_started':False,'train_test_disjoint':True,'training_images':18,'test_images':6,
          'train_only_structure_observations_verified':True,'selected_points':len(selected['structure']),
          'test_feature_matching_authorized_for_split_design':True,'old_training_still_paused':read(RUN/'status.json')['status']=='paused'})
    print('SELECTED',trains,tests,'area',h.volume,'points',len(selected['structure']),flush=True)
    for s in support:print(s['test'],'3D supported features',s['validated_train_only_landmark_correspondences'],'median',s['median_angular_error_deg'],flush=True)


if __name__=='__main__':main()
