"""Artifact checks for the completed frozen20/6 experiment, without retraining."""
from collections import Counter
from pathlib import Path
import json
from .common import ROOT,read,write,ref
from .ipsgr_protocol import check_inputs,MODEL_NAMES
from .ipsgr_training import model_folder


def fresh_generation(run):
    m=check_inputs(run);train={v['name']:v for v in m['train']}
    tasks=read(run/'01_geometry/Wan360/generation_jobs.json')['jobs']
    assignment=read(run/'01_geometry/Wan360/parallel_assignment.json')['workers']
    gpu={name:int(g) for g,names in assignment.items() for name in names};rows=[]
    assert len(tasks)==len(set(j['name'] for j in tasks))==80
    for job in tasks:
        v=train[job['source_training_view']];assert ref(v['image'])==job['source_rgb']
        folder=Path(job['output']);assert folder.resolve().is_relative_to(run.resolve())
        gen=read(folder/'generation.json')
        assert gen['status']=='completed' and gen['signature']==job['signature'] and gen['seed']==42 and gen['source_rgb']==job['source_rgb']
        assert len(gen['frames'])==gen['actual_frames']==49
        assert all(Path(f['image']).is_file() for f in gen['frames']) and Path(gen['video']).exists() and Path(gen['trajectory']).exists()
        rows.append({'name':job['name'],'source':job['source_training_view'],'camera_id':v['camera_id'],'source_sha256':job['source_rgb']['sha256'],'direction':job['direction'],'signature':job['signature'],'GPU':gpu[job['name']],'seconds':gen['seconds'],'peak_allocated_GiB':gen['peak_allocated_vram_bytes']/1024**3,'video':gen['video'],'generation_json':ref(folder/'generation.json'),'frames':49})
    assert len({x['source'] for x in rows})==20
    directions=dict(Counter(x['direction'] for x in rows));assert directions==dict.fromkeys(['forward','backward','left','right'],20)
    result={'status':'passed','jobs':80,'sources':20,'directions':directions,'GPU_jobs':dict(Counter(x['GPU'] for x in rows)),'all_source_files_match_frozen_train_RGB':True,'all_output_directories_inside_new_run':True,'test_sources':0,'old_experiment_videos':0,'all49_lossless_frames_present':True,'script':ref(__file__),'rows':rows}
    write(run/'01_geometry/Wan360/fresh_generation_audit.json',result);return result


def final_models(run):
    m=check_inputs(run);check_inputs(run,'test');allowed={v['name'] for v in m['train']}
    teacher=read(run/'02_semantic_teacher/cache/manifest.json');assert {v['name'] for v in teacher['views']}==allowed
    source=ref(run/'03_layout_bootstrap/final/model.pt');models={};sequence=None
    for name in ['ISLB',*MODEL_NAMES]:
        folder=model_folder(run,name);out=read(folder/'outputs.json');inputs=read(folder/'inputs.json');request=read(folder/'fixed_request.json')
        assert out['status']=='completed' and out['iterations']==10000
        assert out['model']==ref(folder/'final/model.pt')
        assert inputs['dataset']==ref(run/'00_dataset/dataset_manifest.json') and inputs['test_images_used'] is False
        assert {v['name'] for v in request['views']}==allowed and not request['generated_views']
        rows=[json.loads(x) for x in (folder/'losses.jsonl').read_text().splitlines()]
        assert [r['step'] for r in rows]==list(range(1,10001))
        current=[r['view'] for r in rows]
        if sequence is None:sequence=current
        else:assert current==sequence
        if name in ['L-BOOT','CV-2D','IPSGR-Full']:
            assert inputs['initialization_archive']==source and request['stage_reset']
            assert read(folder/'initialization_validation.json')['status']=='passed'
        else:
            expected=run/'01_geometry'/('P0/points.f32' if name in ['B-RGB','B-SEM'] else 'P1/enhanced_points.f32')
            assert inputs['point_cloud']==ref(expected)
        if name!='ISLB':
            ev=read(run/'06_evaluation'/name/'metrics.json')
            assert ev['inputs']['checkpoint']==out['model'] and ev['test_count']==6
            assert {v['id'] for v in ev['views']}=={v['name'] for v in m['test']}
        models[name]={'status':'passed','steps':10000,'gaussians':out['gaussians'],'same_train_camera_sequence':True,'checkpoint':out['model']}
    result={'status':'passed','models':models,'shared_ISLB_source':source,'script':ref(__file__),'dataset_signature':m['dataset_signature'],'teacher_train_only':True}
    write(run/'10_report/final_model_validation.json',result);return result


def source_snapshot(run):
    import shutil,subprocess,time
    dest=run/'10_report/reproducibility/final_code';dest.mkdir(parents=True,exist_ok=True)
    files=list(ROOT.glob('spcgs/ipsgr_*.py'))+list(ROOT.glob('tools/ipsgr_*.py'))
    files += [ROOT/p for p in ['tools/run_ipsgr.py','tools/check_official_teacher.py','tools/check_stage2_initialization.py','tools/run_sparse27.py','tools/run_spc27.py','tools/sam2_worker.py','tools/wan360_worker.py','tools/geometry_worker.py','tools/build.sh','spcgs/common.py','spcgs/pipeline.py','spcgs/m1_geometry.py','spcgs/m1_wan360.py','spcgs/wan360.py','spcgs/official_teacher.py','spcgs/sparse27.py','code/examples/spcgs_native.cpp','configs/iris26_compact7_classes.json','docs/IPSGR_REPAIRED20_6.md']]
    records=[]
    for p in sorted(set(files)):
        target=dest/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);records.append({'repository_path':str(p.relative_to(ROOT)),'snapshot':ref(target)})
    patch=dest.parent/'tracked_changes.patch'
    with patch.open('wb') as f:subprocess.run(['git','diff','--binary','HEAD'],cwd=ROOT,stdout=f,check=True)
    result={'created':time.time(),'base_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'files':records,'tracked_changes':ref(patch),'native_binary':ref(ROOT/'code/bin/spcgs_native'),'note':'Initial source snapshot retained alongside final source. Training input signatures identify the exact implementation and native binary for each model; resource-controller changes do not alter model hyperparameters.'}
    write(dest.parent/'final_code_manifest.json',result);return result
