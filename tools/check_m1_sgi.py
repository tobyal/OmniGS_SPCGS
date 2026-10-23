"""Verify real M1 output was loaded exactly by a completed short native SGI run."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.spatial import cKDTree
from spcgs.common import ROOT,STAGES,read,write,ply,ref

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default='m1_wan360_test');a=p.parse_args()
    run=ROOT/'runs/industrial_spcgs'/a.run
    m1=run/STAGES['m1'];sgi=run/STAGES['sgi']
    output=read(m1/'outputs.json');metrics=read(m1/'metrics.json')
    assert output['status']=='completed' and metrics['added']>0,'M1 has no successful real enhancement'
    assert read(sgi/'outputs.json')['status']=='completed','SGI did not complete'
    request=read(sgi/'request.json')
    assert Path(request['points']).resolve()==Path(output['points']).resolve(),'SGI loaded a different point source'
    points=np.fromfile(output['points'],dtype='f4').reshape(-1,6)
    original=ply(m1/'original.ply')[0]
    initial=ply(sgi/'initial/point_cloud.ply')[0]
    final,_,attributes=ply(sgi/'final/point_cloud.ply')
    np.testing.assert_array_equal(points[:len(original),:3],original)
    np.testing.assert_array_equal(points[:,:3],initial)
    assert len(points)==output['point_count']==request['point_count']
    assert len(points)==len(original)+metrics['added']
    losses=[json.loads(line) for line in (sgi/'losses.jsonl').read_text().splitlines()]
    def finite(value):
        if isinstance(value,dict):return all(finite(v) for v in value.values())
        if isinstance(value,list):return all(finite(v) for v in value)
        return bool(np.isfinite(value)) if isinstance(value,(float,int)) else True
    assert losses and all(finite(row) for row in losses),'Nonfinite or missing SGI losses'
    assert len(final) and np.isfinite(final).all()
    for name in attributes.dtype.names:
        if attributes[name].dtype.kind=='f':assert np.isfinite(attributes[name]).all(),name
    extent=max(float(np.linalg.norm(np.ptp(initial,axis=0))),1e-8)
    distance=cKDTree(initial).query(final)[0]
    normalized_distance=np.quantile(distance,[.5,.9,.99,1])/extent
    assert normalized_distance[2]<.25,'Most final geometry moved far from the initialization'
    result={'passed':True,'scope':'Short integration check; not a full geometry-quality or reconstruction-quality evaluation',
            'm1_point_input':ref(output['points']),'original_count':len(original),'added':metrics['added'],
            'initial_count':len(initial),'initial_xyz_exactly_matches_m1':True,
            'final_count':len(final),'loss_rows':len(losses),'finite_losses_and_model':True,
            'initial_scene_diagonal':extent,
            'final_to_initial_distance_over_scene_diagonal_p50_p90_p99_max':normalized_distance.tolist(),
            'first_loss':losses[0],'last_loss':losses[-1]}
    write(sgi/'m1_integration_check.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
