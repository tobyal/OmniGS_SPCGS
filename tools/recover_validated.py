"""Preserve failed pilot and reuse only its independently valid upstream stages."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spcgs.common import *
from spcgs.pipeline import report
old=ROOT/'runs/industrial_spcgs/quick_20261023';new=ROOT/'runs/industrial_spcgs/validated_20261023'
for name in ['c0','c1','c2']:
    p=old/'05_spc'/name/'outputs.json';j=read(p);j.update(status='failed',reason='Post-run audit: LibTorch scalar dtype reuse corrupted restored spatial_lr_scale (15.6794 became 1.0986e9), causing divergent positions. Retained for diagnosis; not usable for SPC conclusions.');write(p,j)
report(old,read(old/'config.json'))
new.mkdir(exist_ok=False)
for name in ['config.json','data_manifest.json','00_baseline','01_sgi_sfm','02_sgi_gaussians','03_semantic']:(new/name).symlink_to(old/name)
write(new/'reuse_provenance.json',{'source':str(old),'reused_stages':['baseline','M1 skipped fallback','CLIP','S0','SGI'],'rerun_stages':['C0','C1','C2'],'reason':'Only second-stage training was affected by scene extent reload. No existing outputs overwritten.'})
