"""Create a compact, self-contained review bundle without large model duplicates."""
import sys,shutil,tarfile,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spcgs.common import *

def main():
    run=ROOT/'runs/industrial_spcgs/pilot_20261023';bundle=ROOT/'delivery';bundle.mkdir(exist_ok=True)
    for name in ['README.md'] :shutil.copy2(ROOT/name,bundle/name)
    for folder in ['07_reports','06_comparison']:
        for p in (run/folder).rglob('*'):
            if not p.is_file() or p.suffix not in ['.md','.json','.csv','.png']:continue
            # Keep comparison images/plots and metrics; omit repeated per-model raw renders.
            if 'renders' in p.parts or 'new_render' in p.parts:continue
            target=bundle/'pilot'/p.relative_to(run);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    examples=['2023-05-11_10.20.43','2023-05-11_11.42.06']
    for stage in ['00_baseline','02_sgi_gaussians','03_semantic/s0','05_spc/c0','05_spc/c1','05_spc/c2']:
        folder=run/stage
        for filename in ['inputs.json','outputs.json','config.json','metrics.json','losses.jsonl','request.json','execution.json']:
            if (folder/filename).exists():target=bundle/'pilot'/stage/filename;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(folder/filename,target)
        for name in examples:
            for suffix in ['_rgb.png','_semantic.png','_labels.png']:
                p=folder/'renders'/(name+suffix)
                if p.exists():target=bundle/'pilot'/stage/'renders'/p.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    for i,p in enumerate(sorted((run/'05_spc/c2/isp').glob('*/outputs.json'))):
        if not any(read(p)['valid']):continue
        for name in ['0.png','1.png','2.png','mask_0.png','mask_1.png','mask_2.png','outputs.json','request.json']:
            target=bundle/'isp_examples'/p.parent.name/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p.parent/name,target)
    checks=[ROOT/'tests/results/native_gradient_checks.json',ROOT/'tests/regression/results.json',ROOT/'tests/real_spc/results.json']
    for p in checks:
        target=bundle/'checks'/p.parent.name/p.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    for name in ['query_new_vocabulary','query_novel_camera']:
        source=ROOT/'runs'/name
        if source.exists():
            for p in source.rglob('*'):
                if p.is_file() and p.suffix in ['.json','.png']:
                    target=bundle/name/p.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    source_files=[]
    for base in ['spcgs','tools','tests','configs','code/include','code/src','code/cuda_rasterizer','code/examples']:
        for p in (ROOT/base).rglob('*'):
            if p.suffix in ['.py','.cpp','.cu','.h','.cuh','.json','.sh'] and '__pycache__' not in p.parts and not any(x in p.parts for x in ['results','regression','real_spc','isp_baseline']):source_files.append(p)
    source_files.extend([ROOT/'code/CMakeLists.txt',ROOT/'README.md'])
    provenance={'source_files':{str(p.relative_to(ROOT)):sha(p) for p in source_files},'native_binary':ref(ROOT/'code/bin/spcgs_native'),
      'original_baseline_sha256':sha(ROOT.parent/'OmniGS_Structure/shared/models/baseline_s0/checkpoint.ply'),'copied_baseline_sha256':sha(ROOT/'baseline/checkpoint.ply'),
      'sam2_commit':(ROOT/'weights/sam2_commit.txt').read_text().strip(),'task_spec':ref(ROOT/'docs/task_spec.txt'),
      'final_models':{stage:str(run/stage/'final/model.pt') for stage in ['02_sgi_gaussians','03_semantic/s0','05_spc/c0','05_spc/c1','05_spc/c2']}}
    assert provenance['original_baseline_sha256']==provenance['copied_baseline_sha256']
    write(ROOT/'SOURCE_MANIFEST.json',provenance);write(bundle/'SOURCE_MANIFEST.json',provenance)
    with tarfile.open(ROOT/'delivery.tar.gz','w:gz') as t:t.add(bundle,arcname='delivery')
    with tarfile.open(ROOT/'source.tar.gz','w:gz') as t:
        for folder in ['spcgs','tools','configs','tests','code']:
            for p in (ROOT/folder).rglob('*'):
                if not p.is_file() or any(x in p.parts for x in ['build','bin','lib','__pycache__','results','regression','real_spc','isp_baseline']):continue
                t.add(p,arcname=str(p.relative_to(ROOT)))
        t.add(ROOT/'README.md',arcname='README.md');t.add(ROOT/'SOURCE_MANIFEST.json',arcname='SOURCE_MANIFEST.json')
    with tarfile.open(ROOT/'final_model.tar.gz','w:gz') as t:
        t.add(run/'05_spc/c2/final',arcname='final_model')
        t.add(run/'05_spc/c2/request.json',arcname='final_model/training_request.json')
        t.add(run/'05_spc/c2/metrics.json',arcname='final_model/evaluation.json')
    print('Delivery',ROOT/'delivery.tar.gz');print('Source',ROOT/'source.tar.gz')
if __name__=='__main__':main()
