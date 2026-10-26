"""Official-only IRIS split review; cached matching and fixed-pose SfM, no training."""
import collections
import copy
import csv
from pathlib import Path
import shutil
import sys
import numpy as np
from scipy.spatial import ConvexHull

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reselect_iris_interpolation import Diagnostics, OUT as CACHE, ANALYSIS, ROOT, RUN, read, write, xyz, sha, run_command, BIN
from analyze_iris_52_53_replacements import subset_matches
from finalize_iris_overlap_split import heldout_structure_support
from run_omnigs_sparse18 import export_cloud, ref

OUT = ANALYSIS / 'official_overlap_split_20261025'
TRIPLES = [('1-26','1-25','1-29'), ('1-27','1-24','1-28'),
           ('1-60','1-59','1-63'), ('1-61','1-58','1-62'),
           ('1-68','1-67','1-71'), ('1-69','1-66','1-70')]
TESTS = [x[0] for x in TRIPLES]
ANCHORS = sorted({c for x in TRIPLES for c in x[1:]})
EXTRAS = {
    '18_balanced': ['1-34','1-35','1-37','1-44','1-47','1-64'],
    '18_feature_gain': ['1-34','1-35','1-44','1-47','1-64','1-65'],
    '19_balanced_support': ['1-34','1-35','1-37','1-44','1-47','1-64','1-65'],
    '20_extra_upper': ['1-34','1-35','1-36','1-37','1-44','1-47','1-64','1-65'],
}


def components(vertices, edges):
    adj = {i:set() for i in vertices}
    for a,b in edges:
        adj[a].add(b); adj[b].add(a)
    remaining = set(vertices); result = []
    while remaining:
        stack = [min(remaining)]; found = set(stack)
        while stack:
            for j in adj[stack.pop()] - found:
                found.add(j); stack.append(j)
        remaining -= found; result.append(sorted(found))
    return sorted(result, key=lambda x:-len(x))


def official_audit(d):
    inventory = read(ANALYSIS/'camera_inventory.json')
    source = Path(inventory['metadata']['annotation_source'])
    assert sha(source) == inventory['metadata']['annotation_source_sha256']
    coco = read(source); byname = {x['file_name']:x for x in coco['images']}
    counts = collections.Counter(x['image_id'] for x in coco['annotations'])
    rows = []
    for cid in sorted(set(ANCHORS + TESTS + sum(EXTRAS.values(), []))):
        row = d.by[cid]
        assert row['has_official_semantics'] and row['official_face_count'] == 6
        assert sha(row['rmx_source']) == row['rmx_sha256']
        for face in row['official_face_records']:
            record = byname[face['file_name']]
            assert record['id'] == face['official_image_id']
            assert counts[record['id']] == face['object_count'] > 0
        rows.append({'camera_id':cid, 'official_faces_with_objects':6,
                     'official_object_count':row['official_object_count'], 'rmx_hash_verified':True})
    return {'source':ref(source), 'verified_candidates':rows, 'sam_or_review_masks_used':False,
            'definition':'Original IRIS CAD-projected COCO annotations, six face records with annotation objects. Record presence does not establish dense pixel completeness.'}


def evaluate(d, name, trains):
    folder = OUT/'comparisons'/name
    if (folder/'evaluation.json').exists():
        result = read(folder/'evaluation.json'); assert result['train_ids'] == trains
        quality = read(folder/'sfm_quality.json')
        quality['configuration']['pairs'] = f'{len(trains)*(len(trains)-1)//2} exhaustive training-only candidate pairs; only cached geometrically verified matches used'
        write(folder/'sfm_quality.json',quality)
        result['sfm_quality'] = quality
        write(folder/'evaluation.json',result)
        return result
    mvg, data = folder/'sfm', folder/'dataset'
    mvg.mkdir(parents=True, exist_ok=False)
    (data/'train_images').mkdir(parents=True); (data/'openMVG').mkdir()
    indices = {d.index[c] for c in trains}
    assert not set(trains)&set(TESTS)
    assert all(d.by[c]['has_official_semantics'] for c in trains+TESTS)
    scene = copy.deepcopy(d.scene)
    scene['views'] = [v for v in scene['views'] if v['key'] in indices]
    scene['extrinsics'] = [v for v in scene['extrinsics'] if v['key'] in indices]
    scene['root_path'] = str(data/'train_images')
    for c in trains:
        source = (CACHE/'pool_images'/(d.by[c]['image_id']+'.png')).resolve()
        (data/'train_images'/source.name).symlink_to(source)
    write(mvg/'sfm_data_train_fixed.json', scene)
    subset_matches(CACHE/'pool_sfm/matches.fixed_pose.txt', mvg/'matches.fixed_pose.txt', indices)
    run_command([BIN/'openMVG_main_ComputeStructureFromKnownPoses', '-i',mvg/'sfm_data_train_fixed.json',
                 '-m', CACHE/'pool_sfm', '-f',mvg/'matches.fixed_pose.txt', '-d', '-r','4',
                 '-o',mvg/'structure.json'],folder/'triangulation.log')
    export_cloud(mvg,data)
    quality = read(folder/'sfm_quality.json')
    quality['configuration']['pairs'] = f'{len(trains)*(len(trains)-1)//2} exhaustive training-only candidate pairs; only cached geometrically verified matches used'
    write(folder/'sfm_quality.json',quality)
    filtered = read(mvg/'structure_filtered.json')
    edges = set(); observed = set()
    for lm in filtered['structure']:
        ids = [o['key'] for o in lm['value']['observations']]
        assert len(ids)>=3 and set(ids)<=indices
        observed.update(ids)
        for a in ids:
            for b in ids:
                if a<b: edges.add((a,b))
    comps = [[d.rows[i]['camera_id'] for i in c] for c in components(indices,edges)]
    geom = heldout_structure_support(d,mvg/'structure_filtered.json',TESTS)
    support = [d.test_support(t,trains) for t in TESTS]
    result = {'name':name, 'train_ids':trains, 'test_ids':TESTS,
              'height_counts':dict(collections.Counter(d.by[c]['height_band_relative_to_core'] for c in trains)),
              'feature_support':support, 'structure_support':geom,
              'sfm_quality':read(folder/'sfm_quality.json'), 'track_components':comps,
              'all_training_cameras_have_three_view_tracks':observed==indices,
              'test_observations_in_structure':0, 'gaussian_training_started':False}
    write(folder/'evaluation.json', result)
    return result


def compare():
    OUT.mkdir(exist_ok=True); d=Diagnostics()
    write(OUT/'official_annotation_audit.json', official_audit(d))
    valid = [b for b in read(CACHE/'trajectory_brackets.json')
             if all(d.by[b[k]]['has_official_semantics'] for k in ['test','before','after'])]
    write(OUT/'all_official_trajectory_brackets.json', valid)
    results = []
    for name,extra in EXTRAS.items():
        trains = sorted(ANCHORS+extra)
        result = evaluate(d,name,trains); results.append(result)
        print(name, 'heights',result['height_counts'], 'points',result['sfm_quality'],
              'components',list(map(len,result['track_components'])),
              'multi',[x['features_matched_to_at_least_two_train_views'] for x in result['feature_support']],
              '3D',[x['validated_train_only_landmark_correspondences'] for x in result['structure_support']],flush=True)
    write(OUT/'design_comparison.json', results)


def finalize(name):
    d=Diagnostics(); design=read(OUT/'comparisons'/name/'evaluation.json')
    tr=design['train_ids']; te=design['test_ids']; folder=OUT/'comparisons'/name
    assert len(design['track_components'])==1 and design['all_training_cameras_have_three_view_tracks']
    assert all(d.by[c]['has_official_semantics'] for c in tr+te)
    data=OUT/'selected_dataset'; (data/'openMVG').mkdir(parents=True,exist_ok=False)
    for split,ids in [('train',tr),('test',te)]:
        images=data/(split+'_images'); images.mkdir()
        indices={d.index[c] for c in ids}; scene=copy.deepcopy(d.scene)
        scene['views']=[v for v in scene['views'] if v['key'] in indices]
        scene['extrinsics']=[p for p in scene['extrinsics'] if p['key'] in indices]
        scene['root_path']=str(images)
        for cid in ids:
            source=(CACHE/'pool_images'/(d.by[cid]['image_id']+'.png')).resolve()
            (images/source.name).symlink_to(source)
        write(data/f'openMVG/data_openmvg_{split}.json',scene)
        (OUT/f'{split}{len(ids)}.txt').write_text('\n'.join(d.by[c]['image_id'] for c in ids)+'\n',encoding='utf-8')
    shutil.copy2(folder/'dataset/openMVG/scene.ply',data/'openMVG/scene.ply')
    shutil.copy2(folder/'sfm_quality.json',OUT/'sfm_quality.json')
    (OUT/'selected_sfm').symlink_to(folder/'sfm',target_is_directory=True)
    brackets=[d.bracket(*triple) for triple in TRIPLES]
    for b in brackets:
        assert b['before'] in tr and b['after'] in tr
        assert .15 < b['segment_parameter'] < .85 and b['segment_offset_m'] < .1
    selected=[]
    for role,ids in [('train',tr),('test',te)]:
        for i,cid in enumerate(ids,1):
            row=dict(d.by[cid],split=role,split_index=i,pool_view_id=d.index[cid])
            row['selection_role']='heldout_trajectory_interpolation' if role=='test' else 'trajectory_anchor' if cid in ANCHORS else 'additional_shared_observation'
            selected.append(row)
    points=np.array([xyz(d.by[c]) for c in tr]); old=read(CACHE/'selected_split.json')
    s={'status':'candidate_for_review_no_training','design':name,'train_ids':tr,'test_ids':te,'validation_ids':[],
       'selected_cameras':selected,'test_brackets':brackets,'train_official':len(tr),'test_official':len(te),
       'train_height_counts':design['height_counts'],
       'test_height_counts':dict(collections.Counter(d.by[c]['height_band_relative_to_core'] for c in te)),
       'xy_bounds_m':{'min':points[:,:2].min(0).tolist(),'max':points[:,:2].max(0).tolist()},
       'xy_hull_area_m2':float(ConvexHull(points[:,:2]).volume),
       'source_inventory':ref(ANALYSIS/'camera_inventory.json'), 'official_annotation_audit':ref(OUT/'official_annotation_audit.json'),
       'resolution':[2048,1024], 'coordinate_transform':read(RUN/'dataset/coordinate_transform.json'),
       'initial_cloud':ref(data/'openMVG/scene.ply'),'initial_cloud_quality':design['sfm_quality'],
       'test_images_used_for_split_design_matching':True,'test_observations_used_for_initialization':0,
       'gaussian_training_started':False,
       'previous_split_comparison':{'previous_train':old['train_ids'],'previous_test':old['test_ids'],
          'removed_nonofficial_train':[c for c in old['train_ids'] if not d.by[c]['has_official_semantics']],
          'retained_tests':sorted(set(te)&set(old['test_ids'])),
          'old_tests_now_training':sorted(set(tr)&set(old['test_ids']))},
       'selection_note':'All selected images have original official six-face semantics. Three horizontal test stations each have two heights. Same-height chronological neighbors and real matches support interpolation. Not a blind randomized benchmark; no training was run for this analysis.'}
    write(OUT/'selected_split.json',s)
    write(OUT/'selected_test_feature_support.json',design['feature_support'])
    write(OUT/'test_to_train_structure_support.json',design['structure_support'])
    fields=['split','camera_id','image_id','source_group','x_m','y_m','z_m','height_band_relative_to_core',
            'has_official_semantics','official_face_count','official_object_count','selection_role','pool_view_id','official_source']
    with (OUT/'selected_cameras.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(selected)
    write(OUT/'verification.json',{'training_started_by_this_analysis':False,'train_test_disjoint':not set(tr)&set(te),
        'training_images':len(tr),'test_images':len(te),'validation_images':0,'all_selected_have_official_semantics':True,
        'sam_candidates_used_as_official':False,'train_only_structure_observations_verified':True,
        'train_track_connected_components':design['track_components'], 'test_observations_in_structure':0,
        'new_split_requires_fresh_training':True,'reason':'Some new test views belonged to previous training sets; existing checkpoints are not independent evaluations of this split.'})
    print('SELECTED',name,tr,te,flush=True)


if __name__=='__main__':
    if len(sys.argv)==1 or sys.argv[1]=='compare': compare()
    elif sys.argv[1]=='finalize': finalize(sys.argv[2])
