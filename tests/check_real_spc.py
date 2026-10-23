"""Real-data SPC diagnostic on an already reconstructed semantic baseline.

This validates M4/M5 independently; it is NOT labeled SGI -> SPC training.
All prompts, current-model pseudo images, actual losses and final model are saved.
"""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spcgs.common import *
from tests.legacy_s_fixture import with_cached_s

def main():
    source=ROOT/'runs/industrial_spcgs/quick_20261023/03_semantic/s0';out=ROOT/'tests/real_spc';out.mkdir(exist_ok=True)
    c=with_cached_s(read(source/'request.json'),out/'explicit_S');c.update(archive=str(source/'final/model.pt'),stage_reset=False,resume=False,iterations=40,save_every=20,spc2d=True,spc3d=True,isp_every=10,spc_ramp=40,output=str(out/'training'),seed=7,densify_until=0)
    c['signature']=signature({k:v for k,v in c.items() if k not in ['output','signature']});native(c,out/'job')
    rows=[json.loads(x) for x in (out/'training/losses.jsonl').read_text().splitlines()];events=[x for x in rows if x['isp_valid_pairs']]
    result={'scope':'Real-data M4/M5 diagnostic from S0; not SGI pipeline or formal benchmark','events':len(events),'valid_pairs':sum(x['isp_valid_pairs'] for x in rows),'max_inter':max(x['inter'] for x in rows),'max_intra':max(x['intra'] for x in rows),'max_spc3d':max(x['spc3d'] for x in rows),'final_model':str(out/'training/final/model.pt')}
    result['status']='passed' if result['max_inter']>0 and result['max_intra']>0 and result['max_spc3d']>0 else 'failed';write(out/'results.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
