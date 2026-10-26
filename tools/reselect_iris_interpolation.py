"""Choose an IRIS split from real trajectory and appearance overlap, without training.

Test images are explicitly allowed for split-design diagnostics. Geometry used
for any eventual training initialization is reconstructed from training views only.
"""
import itertools
import os
from pathlib import Path
import shutil
import sys
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_omnigs_sparse18 import ROOT, BASE, read, write, sha, bearings, filter_matches
from analyze_iris_52_53_replacements import run_command, BIN, xyz

ANALYSIS = ROOT / 'runs/iris_camera_analysis_20261025'
PRIOR = ANALYSIS / 'replacement_52_53_20261025'
OUT = ANALYSIS / 'trajectory_overlap_split_20261025'
RUN = ROOT / 'runs/omnigs18_6_20261025_v1'


def build_pool():
    OUT.mkdir(exist_ok=False)
    mvg, images = OUT / 'pool_sfm', OUT / 'pool_images'
    mvg.mkdir()
    images.mkdir()
    rows = read(PRIOR / 'pool_rows.json')
    ids = {x['camera_id'] for x in rows}
    inventory = read(ANALYSIS / 'camera_inventory.json')['cameras']
    rows += [x for x in inventory if x['group'] == 1 and x['is_main_core'] and x['camera_id'] not in ids]
    assert len(rows) == 79 and len({x['camera_id'] for x in rows}) == 79
    sfm = read(PRIOR / 'pool_sfm/sfm_data_train_fixed.json')
    sfm['root_path'] = str(images)
    origin = np.array(read(RUN / 'dataset/coordinate_transform.json')['origin_m'])
    basis = np.array([[0., 0., -1.], [1., 0., 0.], [0., -1., 0.]])
    for i, row in enumerate(rows):
        source = BASE / 'datasets/iris_full_A_official_2k/images' / (row['image_id'] + '.png')
        (images / source.name).symlink_to(source)
        if i < 37:
            for suffix in ['.feat', '.desc']:
                shutil.copy2(PRIOR / 'pool_sfm' / (row['image_id'] + suffix), mvg / (row['image_id'] + suffix))
            continue
        assert sha(row['rmx_source']) == row['rmx_sha256']
        raw = Path(row['rmx_source']).read_bytes()
        txt = raw.decode('utf-16' if raw[:2] in (b'\xff\xfe', b'\xfe\xff') else 'utf-8-sig')
        a = [list(map(float, s.split())) for s in txt.splitlines() if s.strip() and not s.startswith('#')]
        C = np.array(a[0]) / 1000.
        np.testing.assert_allclose(C, xyz(row), atol=1e-8, rtol=0)
        R = (Rotation.from_rotvec(np.array(a[1])*a[2][0]).as_matrix() @ basis).T
        sfm['views'].append({'key':i, 'value':{'polymorphic_id':1073741824, 'ptr_wrapper':{'id':2147483649+2*i,
             'data':{'local_path':'','filename':source.name,'width':2048,'height':1024,'id_view':i,'id_pose':i,'id_intrinsic':0}}}})
        sfm['extrinsics'].append({'key':i,'value':{'rotation':R.tolist(),'center':(C-origin).tolist()}})
    sfm['intrinsics'][0]['value']['ptr_wrapper']['id'] = 2147484000
    shutil.copy2(PRIOR / 'pool_images/mask.png', images / 'mask.png')
    shutil.copy2(PRIOR / 'pool_sfm/image_describer.json', mvg / 'image_describer.json')
    write(mvg / 'sfm_data_train_fixed.json', sfm)
    write(OUT / 'pool_rows.json', rows)
    run_command([BIN/'openMVG_main_ComputeFeatures','-i',mvg/'sfm_data_train_fixed.json','-o',mvg,'-m','SIFT','-p','ULTRA','-n','8'], OUT/'features.log')
    (mvg/'new_pairs.txt').write_text(''.join(f'{i} {j}\n' for i,j in itertools.combinations(range(len(rows)),2) if j>=37))
    run_command([BIN/'openMVG_main_ComputeMatches','-i',mvg/'sfm_data_train_fixed.json','-o',mvg/'new_matches.txt',
                 '-p',mvg/'new_pairs.txt','-r','.8','-n','FASTCASCADEHASHINGL2'],OUT/'matching.log')
    (mvg/'matches.putative.txt').write_text((PRIOR/'pool_sfm/matches.putative.txt').read_text()+'\n'+(mvg/'new_matches.txt').read_text())
    filter_matches(mvg)
    write(OUT/'status.json',{'status':'pool_complete','images':len(rows),'gaussian_training_started':False,
          'test_images_permitted_for_split_diagnostics':True,'test_images_permitted_for_training_initialization':False})


class Diagnostics:
    def __init__(self):
        self.rows=read(OUT/'pool_rows.json')
        self.index={x['camera_id']:i for i,x in enumerate(self.rows)}
        self.by={x['camera_id']:x for x in self.rows}
        self.scene=read(OUT/'pool_sfm/sfm_data_train_fixed.json')
        self.poses={p['key']:p['value'] for p in self.scene['extrinsics']}
        self.feats={i:np.loadtxt(OUT/'pool_sfm'/(r['image_id']+'.feat'),ndmin=2)[:,:2] for i,r in enumerate(self.rows)}
        self.rays={i:bearings(xy)@np.array(self.poses[i]['rotation']) for i,xy in self.feats.items()}
        self.matches={}
        with (OUT/'pool_sfm/matches.fixed_pose.txt').open() as f:
            while True:
                line=f.readline()
                if not line:break
                if not line.strip():continue
                i,j=map(int,line.split());n=int(f.readline())
                a=np.array([list(map(int,f.readline().split())) for _ in range(n)],dtype=int)
                self.matches[i,j]=a
                self.matches[j,i]=a[:,::-1]

    def match(self,a,b):
        return self.matches.get((self.index[a],self.index[b]),np.empty((0,2),dtype=int))

    def occupancy(self, cid, feature_ids):
        i=self.index[cid];xy=self.feats[i]
        # Feature support, not dense pixel visibility. Use the feature-mask interior.
        ix=np.clip((xy[:,0]/2048*12).astype(int),0,11)
        iy=np.clip(((xy[:,1]-41)/860*6).astype(int),0,5)
        cells=iy*12+ix
        eligible=np.bincount(cells,minlength=72)>=5
        counts=np.bincount(cells[np.asarray(feature_ids,dtype=int)],minlength=72)
        supported=(counts>=3)&eligible
        return {'supported_cells':int(supported.sum()),'eligible_cells':int(eligible.sum()),
                'fraction':float(supported.sum()/max(eligible.sum(),1)),
                'supported_azimuth_sectors':int((np.bincount(ix[np.asarray(feature_ids,dtype=int)],minlength=12)>=3).sum())}

    def bracket(self,t,a,b):
        p,A,B=[xyz(self.by[x]) for x in [t,a,b]]
        ab=B-A;lam=float((p-A)@ab/(ab@ab));projection=A+lam*ab
        dA,dB=np.linalg.norm(A-p),np.linalg.norm(B-p)
        angle=float(np.rad2deg(np.arccos(np.clip((A-p)@(B-p)/(dA*dB),-1,1))))
        ma,mb=self.match(t,a),self.match(t,b)
        da,db=dict(ma),dict(mb)
        common=sorted(da.keys()&db.keys())
        ab_map=dict(self.match(a,b))
        cycle=np.array([k for k in common if ab_map.get(da[k])==db[k]],dtype=int)
        # Triangulate from A/B only; T is only a held-out reprojection check.
        valid=[];errs=[]
        if len(cycle):
            ia,ib,it=[self.index[x] for x in [a,b,t]]
            ca,cb,ct=[np.array(self.poses[i]['center']) for i in [ia,ib,it]]
            ra=self.rays[ia][[da[k] for k in cycle]];rb=self.rays[ib][[db[k] for k in cycle]]
            delta=cb-ca;dot=np.sum(ra*rb,axis=1);den=1-dot**2
            sa=(ra@delta-dot*(rb@delta))/np.maximum(den,1e-12)
            sb=(dot*(ra@delta)-rb@delta)/np.maximum(den,1e-12)
            X=(ca+sa[:,None]*ra+cb+sb[:,None]*rb)/2
            rt=X-ct;depth=np.linalg.norm(rt,axis=1);rt/=depth[:,None]
            truth=self.rays[it][cycle]
            err=np.rad2deg(np.arctan2(np.linalg.norm(np.cross(rt,truth),axis=1),np.sum(rt*truth,axis=1)))
            ok=(err<=.35)&(sa>.15)&(sb>.15)&(sa<200)&(sb<200)&(depth>.15)&(depth<200)
            valid=cycle[ok].tolist();errs=err[ok].tolist()
        return dict(test=t,before=a,after=b,segment_parameter=lam,segment_offset_m=float(np.linalg.norm(p-projection)),
                    bracket_angle_deg=angle,distance_before_m=float(dA),distance_after_m=float(dB),
                    max_abs_dz_m=float(max(abs(p[2]-A[2]),abs(p[2]-B[2]))),matches_before=len(ma),matches_after=len(mb),
                    shared_test_features=len(common),cycle_consistent_features=len(cycle),
                    train_pair_reprojection_supported_features=len(valid),
                    median_reprojection_error_deg=float(np.median(errs)) if errs else None,
                    union_occupancy=self.occupancy(t,sorted(set(da)|set(db))),
                    verified_occupancy=self.occupancy(t,valid),verified_test_feature_ids=valid)

    def test_support(self,t,trains):
        entries=[];union=set();multi={}
        for a in trains:
            m=self.match(t,a);ids=set(m[:,0].tolist());union|=ids
            for k in ids:multi[k]=multi.get(k,0)+1
            entries.append({'train':a,'geometric_matches':len(m),'distance_m':float(np.linalg.norm(xyz(self.by[t])-xyz(self.by[a]))),
                            'occupancy':self.occupancy(t,sorted(ids))})
        two=[k for k,n in multi.items() if n>=2]
        return dict(test=t,connections=sorted(entries,key=lambda x:-x['geometric_matches']),
                    total_pair_matches=sum(x['geometric_matches'] for x in entries),unique_test_features=len(union),
                    features_matched_to_at_least_two_train_views=len(two),union_occupancy=self.occupancy(t,sorted(union)),
                    two_view_occupancy=self.occupancy(t,two))


def inspect_brackets():
    d=Diagnostics();triples=[]
    for band in ['lower','upper']:
        path=sorted([x for x in d.rows if x['group']==1 and x['height_band_relative_to_core']==band],key=lambda x:x['group_index'])
        for i,t in enumerate(path):
            if not t['has_official_semantics']:continue
            # Only nearby positions along the actual same-height capture sequence.
            for a in path[max(0,i-3):i]:
                for b in path[i+1:i+4]:
                    p,A,B=[xyz(x) for x in [t,a,b]]
                    ab=B-A;lam=(p-A)@ab/(ab@ab)
                    if not .15<=lam<=.85 or np.linalg.norm(p-(A+lam*ab))>.35:continue
                    if max(np.linalg.norm(p-A),np.linalg.norm(p-B))>1.8 or max(abs(p[2]-A[2]),abs(p[2]-B[2]))>.2:continue
                    info=d.bracket(t['camera_id'],a['camera_id'],b['camera_id'])
                    if min(info['matches_before'],info['matches_after'])<30:continue
                    triples.append(info)
    triples.sort(key=lambda x:(x['train_pair_reprojection_supported_features']*x['verified_occupancy']['fraction']),reverse=True)
    write(OUT/'trajectory_brackets.json',triples)
    old=read(RUN/'approved_split.json')['selected_cameras']
    trains=[x['camera_id'] for x in old if x['candidate_role']=='train']
    tests=[x['camera_id'] for x in old if x['candidate_role']=='test']
    write(OUT/'old_split_test_support.json',[d.test_support(t,trains) for t in tests])
    print('BRACKETS_COMPLETE',len(triples),flush=True)


def propose_designs():
    d=Diagnostics()
    designs={
      'compact':[('1-26','1-25','1-29'),('1-34','1-33','1-37'),('1-35','1-32','1-36'),('1-44','1-43','1-47'),('1-59','1-56','1-60'),('1-61','1-58','1-62')],
      'central':[('1-19','1-18','1-20'),('1-26','1-25','1-29'),('1-34','1-33','1-37'),('1-44','1-43','1-47'),('1-60','1-59','1-63'),('1-61','1-58','1-62')],
      'north':[('1-03','1-02','1-06'),('1-13','1-12','1-16'),('1-19','1-18','1-20'),('1-26','1-25','1-29'),('1-34','1-33','1-37'),('1-44','1-43','1-47')]}
    def score(trains,tests):
        support=[d.test_support(t,trains) for t in tests]
        value=sum(np.log1p(s['unique_test_features'])+.8*np.log1p(s['features_matched_to_at_least_two_train_views'])+
                  1.5*s['two_view_occupancy']['fraction'] for s in support)
        return float(value),support
    results={}
    for name,triples in designs.items():
        tests=[x[0] for x in triples];anchors=list(dict.fromkeys(z for x in triples for z in x[1:]))
        assert len(anchors)==12 and not set(anchors)&set(tests)
        points=np.array([xyz(d.by[x]) for x in anchors]);low,high=points.min(0),points.max(0)
        pool=[x['camera_id'] for x in d.rows if x['group']==1 and x['camera_id'] not in tests and
              x['camera_id'] not in ['1-49','1-54'] and np.all(xyz(x)[:2]>=low[:2]-.3) and np.all(xyz(x)[:2]<=high[:2]+.3)]
        chosen=anchors[:];steps=[]
        while len(chosen)<18:
            current,_=score(chosen,tests)
            counts={h:sum(d.by[x]['height_band_relative_to_core']==h for x in chosen) for h in ['upper','lower']}
            options=[]
            for c in pool:
                if c in chosen or counts[d.by[c]['height_band_relative_to_core']]>=9:continue
                val,_=score(chosen+[c],tests);options.append((val-current,c))
            options.sort(reverse=True)
            eligible=[x for x in options if x[0]>=.98*options[0][0]]
            gain,c=max(eligible,key=lambda x:(d.by[x[1]]['has_official_semantics'],x[0]))
            chosen.append(c);steps.append(dict(added=c,gain=gain,top_options=options[:5]))
        value,support=score(chosen,tests)
        results[name]={'tests':tests,'anchors':anchors,'train':chosen,'score':value,'selection_steps':steps,
                      'brackets':[d.bracket(*x) for x in triples],'test_support':support,
                      'train_official':sum(d.by[x]['has_official_semantics'] for x in chosen),
                      'xyz_min':np.min([xyz(d.by[x]) for x in chosen],axis=0).tolist(),
                      'xyz_max':np.max([xyz(d.by[x]) for x in chosen],axis=0).tolist()}
    write(OUT/'design_comparison.json',results)
    print('DESIGNS_COMPLETE',flush=True)


if __name__=='__main__':
    if len(sys.argv)==1:build_pool();inspect_brackets()
    elif sys.argv[1]=='brackets':inspect_brackets()
    elif sys.argv[1]=='designs':propose_designs()
