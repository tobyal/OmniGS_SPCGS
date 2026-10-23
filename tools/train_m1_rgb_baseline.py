"""Original OmniGS baseline training, changing only the sparse initialization."""
import argparse
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spcgs.common import ROOT, read, write, ref, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='m1_rgb_baseline_11901_20261024')
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--baseline-gpu',type=int,default=2)
    args=parser.parse_args()
    if Path(args.run).name != args.run or args.run in ['.','..']:
        parser.error('Use one run directory name')
    run=ROOT/'runs/industrial_spcgs'/args.run
    if run.exists():
        raise FileExistsError(f'Refusing to overwrite {run}')
    run.mkdir(parents=True)
    base=ROOT.parent/'OmniGS_PipeFilter'
    provenance=read(ROOT/'baseline/provenance.json')
    old=provenance['source_training_provenance']
    initial=ROOT/'runs/industrial_spcgs/m1_wan360_56_batch_20261023/01_sgi_sfm/enhanced.ply'
    original=base/'data/dataset/openMVG/scene.ply'
    trainer=base/'code/bin/train_egonerf_ricoh360'
    exporter=base/'code/bin/pipefilter_export'
    baseline=ROOT/'baseline/checkpoint.ply'
    children=[]
    state={'status':'running','phase':'preparing','started':time.time(),'controller_pid':os.getpid()}
    def update(**values):
        state.update(values,updated=time.time());write(run/'status.json',state)
    def launch(command,log,gpu):
        env=os.environ.copy()
        for key in list(env):
            if key.startswith(('PF_','SGS_')):env.pop(key)
        env.update({k:str(v) for k,v in old['filter'].items()})
        env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                   MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
        with log.open('a') as stream:
            child=subprocess.Popen(list(map(str,command)),cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                                   stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(child);return child
    def terminate(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    try:
        import numpy as np
        from plyfile import PlyData
        assert sha(trainer)==old['source_manifest']['training_binary_sha256']
        assert sha(base/'code/lib/libcuda_rasterizer.so')==old['source_manifest']['cuda_library_sha256']
        assert sha(ROOT/'configs/baseline.yaml')==old['config_sha256']
        assert sha(baseline)==provenance['checkpoint_sha256']
        a=PlyData.read(original)['vertex'].data;b=PlyData.read(initial)['vertex'].data
        assert len(a)==3979 and len(b)==11901
        for key in ['x','y','z','red','green','blue']:
            np.testing.assert_array_equal(a[key],b[key][:len(a)])
        assert np.isfinite(np.column_stack([b[k] for k in ['x','y','z']])).all()
        data=run/'dataset';(data/'openMVG').mkdir(parents=True)
        (data/'images').symlink_to((base/'data/dataset/images').resolve(),target_is_directory=True)
        for split in ['train','val','test']:
            shutil.copy2(base/f'data/dataset/openMVG/data_openmvg_{split}.json',data/f'openMVG/data_openmvg_{split}.json')
        shutil.copy2(initial,data/'openMVG/scene.ply')
        config=run/'baseline.yaml';shutil.copy2(ROOT/'configs/baseline.yaml',config)
        protocol=read(ROOT/'data/protocol.json');scene=read(data/'openMVG/data_openmvg_train.json')
        names=sorted(Path(v['value']['ptr_wrapper']['data']['filename']).stem for v in scene['views'])
        assert names==sorted(protocol['splits']['train']) and len(names)==56
        references=[]
        for v in scene['views']:
            d=v['value']['ptr_wrapper']['data'];assert (d['width'],d['height'])==(2048,1024)
            references.append(ref(Path(scene['root_path'])/d['filename']))
        spec={'intentional_change':'Replace original 3979 sparse points with M1-enhanced 11901 points; all RGB training settings unchanged.',
              'train_views':56,'resolution':[2048,1024],'iterations':32010,'seed':0,
              'initial_points':11901,'original_points':3979,'original_xyzrgb_exactly_preserved':True,
              'initialization':'Same native createFromPcd: RGB-to-SH, nearest-neighbor spherical scales, identity rotation, opacity=0.1.',
              'config':ref(config),'trainer':ref(trainer),'exporter':ref(exporter),
              'native_libraries':[ref(p) for p in sorted((base/'code/lib').glob('*.so'))],
              'initial_cloud':ref(data/'openMVG/scene.ply'),'source_m1_cloud':ref(initial),
              'baseline_checkpoint':ref(baseline),'baseline_provenance':provenance,
              'train_cameras':ref(data/'openMVG/data_openmvg_train.json'),'training_images':references,
              'environment':old['filter'],'training_gpu':args.gpu,'baseline_evaluation_gpu':args.baseline_gpu}
        write(run/'experiment.json',spec)
        model=run/'model';model.mkdir()
        command=[trainer,config,data,model,'no_viewer']
        write(run/'training_command.json',{'command':list(map(str,command)),'gpu':args.gpu})
        training_started=time.time()
        training=launch(command,run/'training.log',args.gpu)
        metric_python=Path('/mnt/sda/ubuntu/anaconda3/envs/lkx_odgs/bin/python')
        eval_args=[metric_python,ROOT/'tools/evaluate_rgb_train.py','--dataset',data,'--config',config,'--exporter',exporter]
        previous=launch([*eval_args,'--checkpoint',baseline,'--output',run/'evaluation/baseline_train',
                         '--gpu',str(args.baseline_gpu)],run/'baseline_evaluation.log',args.baseline_gpu)
        update(phase='training',training_pid=training.pid,baseline_evaluation_pid=previous.pid)
        while training.poll() is None:
            update(baseline_evaluation_returncode=previous.poll(),training_seconds=time.time()-training_started)
            time.sleep(15)
        if training.returncode:
            raise RuntimeError(f'Training exited with {training.returncode}; see training.log')
        checkpoint=model/'32010_shutdown/ply/point_cloud/iteration_32010/point_cloud.ply'
        assert checkpoint.is_file()
        cloud=PlyData.read(checkpoint)['vertex'].data
        # Native PLY stores opacity logits; +/-inf activates to a finite 1/0.
        assert all(np.isfinite(cloud[k]).all() for k in cloud.dtype.names if k!='opacity')
        assert not np.isnan(cloud['opacity']).any()
        opacity_logits={'positive_infinity':int(np.isposinf(cloud['opacity']).sum()),
                        'negative_infinity':int(np.isneginf(cloud['opacity']).sum())}
        used=(model/'used_times/keyframe_used_timesfinal.txt').read_text()
        counts={int(k):int(n) for k,n in re.findall(r'^(\d+) (\d+) \d+$',used,re.M)}
        assert set(counts)=={v['key'] for v in scene['views']}
        assert min(counts.values())>0 and sum(counts.values())==32010
        assert 'Loaded number of point3D: 11901' in (run/'training.log').read_text(errors='replace')
        write(run/'training_result.json',{'status':'completed','iterations':32010,'final_gaussians':len(cloud),
             'seconds':time.time()-training_started,'checkpoint':ref(checkpoint),'per_camera_updates':counts,
             'initialization_count_verified_from_native_log':11901,'opacity_logits':opacity_logits})
        current=launch([*eval_args,'--checkpoint',checkpoint,'--output',run/'evaluation/enhanced_train',
                        '--gpu',str(args.gpu)],run/'enhanced_evaluation.log',args.gpu)
        update(phase='evaluating',enhanced_evaluation_pid=current.pid,final_gaussians=len(cloud))
        while current.poll() is None or previous.poll() is None:
            update(enhanced_evaluation_returncode=current.poll(),baseline_evaluation_returncode=previous.poll())
            time.sleep(15)
        if current.returncode or previous.returncode:
            raise RuntimeError('Training completed but evaluation failed; inspect evaluation logs')
        old_metrics=read(run/'evaluation/baseline_train/metrics.json')
        new_metrics=read(run/'evaluation/enhanced_train/metrics.json')
        assert old_metrics['definitions']==new_metrics['definitions']
        assert old_metrics['views']==new_metrics['views']==56
        assert sha(baseline)==provenance['checkpoint_sha256'] and sha(initial)==spec['source_m1_cloud']['sha256']
        comparison={'status':'completed','split':'train','views':56,'resolution':[2048,1024],
            'baseline':old_metrics['aggregate'],'enhanced':new_metrics['aggregate'],
            'delta_enhanced_minus_baseline':{k:new_metrics['aggregate'][k]-v for k,v in old_metrics['aggregate'].items()},
            'baseline_initial_points':3979,'enhanced_initial_points':11901,
            'baseline_final_gaussians':647929,'enhanced_final_gaussians':len(cloud),
            'iterations':32010,'definitions':new_metrics['definitions'],
            'baseline_and_m1_sources_unchanged':True}
        write(run/'comparison.json',comparison)
        update(status='completed',phase='completed',completed=time.time(),comparison=comparison)
        print('COMPLETED',run,flush=True)
    except BaseException as error:
        update(status='failed',reason=str(error));(run/'error.log').write_text(traceback.format_exc());raise
    finally:
        for child in children:
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)


if __name__=='__main__':main()
