"""Validate completed F/S/T artifacts and optional text-only cache reuse."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from spcgs.common import ROOT,read,write,ref

def check(path):
    m=read(path);k=len(m['classes']);t=np.fromfile(m['text'],dtype='f4').reshape(k,512)
    assert np.isfinite(t).all();np.testing.assert_allclose(np.linalg.norm(t,axis=1),1,atol=1e-5)
    rows=[]
    for v in m['views']:
        h,w=v['feature_height'],v['feature_width'];f=np.fromfile(v['feature'],dtype='f2').astype('f4').reshape(512,h,w)
        s=np.fromfile(v['relevancy'],dtype='f2').astype('f4').reshape(k,h,w)
        assert np.isfinite(f).all() and np.isfinite(s).all();assert s.min()>=0 and s.max()<=1
        norm_error=float(np.abs(np.linalg.norm(f,axis=0)-1).max());assert norm_error<.001
        fm=read(v['F_manifest']);sm=read(v['S_manifest']);assert fm['status']==sm['status']=='completed'
        scales=np.load(fm['scales'],mmap_mode='r');assert scales.shape==(3,512,h,w) and np.isfinite(scales).all()
        coverage=np.load(fm['coverage']);assert coverage['counts'].min()>0 and (coverage['face_weights'].sum(0)>0).all()
        maps=np.load(sm['maps']);np.testing.assert_array_equal(maps['S'],s.astype('f2'))
        assert np.max(np.abs(maps['relevancy_scales'].astype('f4').mean(0)-s))<.001
        rows.append({'view':v['name'],'F_key':fm['key'],'S_key':sm['key'],'patches':fm['patch_count'],
                     'minimum_coverage':int(coverage['counts'].min()),'F_norm_max_error':norm_error,'normalized_entropy':sm['uncertainty']['normalized_entropy']})
    return {'status':'passed','manifest':ref(path),'view_count':len(rows),'class_count':k,'rows':rows}

def main():
    p=argparse.ArgumentParser();p.add_argument('manifest');p.add_argument('--text-diagnostic');p.add_argument('--output',required=True);a=p.parse_args()
    r=check(ROOT/a.manifest)
    if a.text_diagnostic:
        base=read(ROOT/a.manifest);diag=read(ROOT/a.text_diagnostic)
        matched=[]
        for v in diag['views']:
            b=next(x for x in base['views'] if x['name']==v['name'])
            assert b['F_manifest']==v['F_manifest'] and b['S_manifest']!=v['S_manifest'];matched.append(v['name'])
        assert base['T_manifest']!=diag['T_manifest'];r['text_only_reuse']={'status':'passed','matched_views':matched,'diagnostic':ref(ROOT/a.text_diagnostic)}
    write(ROOT/a.output,r);print({k:v for k,v in r.items() if k!='rows'})
if __name__=='__main__':main()
