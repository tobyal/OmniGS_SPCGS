"""Read actual LibTorch archives and verify sparse SGI inheritance/reset."""
import sys,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from spcgs.common import ROOT,read,write,ref

def check(run):
    run=Path(run).resolve();cfg=read(run/'config.json');source=torch.jit.load(cfg['sgi_archive'],map_location='cpu')
    results=[];first=None
    folders=[run/'03_stage_initialization',*sorted((run/'05_spc').glob('c[012]'))]
    for folder in folders:
        path=folder/'initial/model.pt'
        if not path.exists():continue
        model=torch.jit.load(str(path),map_location='cpu');assert model.xyz.shape==(287787,3) and model.semantic.shape==(287787,32) and model.text.shape==(7,512)
        inherited=['xyz','dc','semantic','decoder','bias','classifier','cbias','projection','text']
        errors={k:float((getattr(model,k)-getattr(source,k)).abs().max()) for k in inherited}
        assert all(x==0 for x in errors.values())
        assert model.rest.abs().max()==0 and model.sh==0 and model.step==0
        assert torch.allclose(model.opacity.sigmoid(),torch.full_like(model.opacity,.1),atol=1e-7)
        expected=torch.zeros_like(model.rotation);expected[:,0]=1;assert torch.equal(model.rotation,expected)
        assert all(getattr(model,k).abs().max()==0 for k in ['radii','gradient','denom','exist','added','pruned'])
        assert torch.isfinite(model.scale).all() and torch.exp(model.scale).min()>0
        audit=read(folder/'stage_reset_audit.json');assert audit['optimizer_state_entries']==audit['head_optimizer_state_entries']==0
        keys=inherited+['scale','rotation','opacity','rest']
        if first is None:first={k:getattr(model,k).clone() for k in keys}
        else:assert all(torch.equal(getattr(model,k),first[k]) for k in keys),'Second-stage initial parameters differ'
        results.append({'model':str(path),'gaussians':len(model.xyz),'inherited_max_errors':errors,'reset_and_fresh_optimizers':'passed'})
    assert results
    result={'status':'passed','source':ref(cfg['sgi_archive']),'same_initial_parameters':True,'models':results}
    write(run/'initialization_validation.json',result);print(result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();check(a.run)
