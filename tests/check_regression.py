"""Real-scene renderer equivalence and checkpoint-resume regression checks."""
import sys,os,copy,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from spcgs.common import *
from tests.legacy_s_fixture import with_cached_s

def main():
    root=ROOT/'tests/regression';root.mkdir(exist_ok=True)
    source=ROOT/'runs/industrial_spcgs/quick_20261023';cfg=read(source/'config.json');name=cameras('test')[0]['name']
    old=ROOT.parent/'OmniGS_Structure';req={'width':cfg['width'],'height':cfg['height'],'split':'test','views':[name],'moments':False,'diagnostics':False}
    write(root/'legacy_request.json',req);dest=root/'legacy_render';dest.mkdir(exist_ok=True)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',PF_MODE='0',SGS_MODE='0',OMP_NUM_THREADS='2');env.pop('PF_FIELD',None)
    run_command([old/'code/bin/pipefilter_export',old/'configs/baseline.yaml',DATA,ROOT/'baseline/checkpoint.ply',root/'legacy_request.json',dest],root/'legacy.log',env)
    new_request=read(source/'00_baseline/request.json');new_request.update(output=str(root/'new_render'),views=[cameras('test')[0]]);native(new_request,root/'new_render_job')
    a=np.fromfile(dest/'rgb'/(name+'.f32'),dtype='f4');b=np.fromfile(root/'new_render'/(name+'_rgb.f32'),dtype='f4');error=float(np.max(np.abs(a-b)));assert error<1e-6,error
    request=with_cached_s(read(source/'02_sgi_gaussians/request.json'),root/'explicit_S');request.update(iterations=40,save_every=20,densify_until=40,ogr_every=40,output=str(root/'continuous'))
    request['signature']=signature({k:v for k,v in request.items() if k not in ['output','signature']})
    native(request,root/'continuous_job')
    request.update(archive=str(root/'continuous/checkpoint_20/model.pt'),resume=True,output=str(root/'resume'))
    native(request,root/'resume_job')
    import torch
    expected=torch.jit.load(str(root/'continuous/final/model.pt'),map_location='cpu');actual=torch.jit.load(str(root/'resume/final/model.pt'),map_location='cpu')
    errors={k:float((getattr(expected,k).float()-getattr(actual,k).float()).abs().max()) for k in ['xyz','scale','opacity','rotation','dc','rest','semantic','decoder']}
    # GPU atomic adds are not bitwise deterministic; report actual differences.
    assert all(np.isfinite(v) for v in errors.values())
    counters_match=bool(int(expected.added)==int(actual.added) and int(expected.pruned)==int(actual.pruned)) if hasattr(expected,'added') else None
    close=all(v<2e-4 for v in errors.values()) and counters_match is not False
    result={'status':'passed' if close else 'failed','baseline_renderer_max_abs_error':error,'baseline_original_sha256':sha(old/'shared/models/baseline_s0/checkpoint.ply'),'baseline_copy_sha256':sha(ROOT/'baseline/checkpoint.ply'),'resume_max_abs_errors':errors,'resume_tolerance':2e-4,'resume_expected_step':int(expected.step),'resume_actual_step':int(actual.step),'densification_counters_match':counters_match}
    write(root/'results.json',result);print(json.dumps(result,indent=2));assert close,result
if __name__=='__main__':main()
