"""Train and evaluate original OmniGS from scratch on the approved compact split."""
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

from run_omnigs_sparse18 import ROOT, BASE, read, write, ref, sha


def prepare(run, selection):
    from PIL import Image
    from plyfile import PlyData
    approved = read(selection / 'selected_split.json')
    assert approved['validation_ids'] == []
    rows = approved['selected_cameras']
    assert len(rows) == 24 and len({r['image_id'] for r in rows}) == 24
    assert {r['camera_id'] for r in rows if r['split'] == 'test'} == {
        '1-26', '1-34', '1-35', '1-44', '1-59', '1-61'}
    shutil.copy2(selection / 'selected_split.json', run / 'source_selected_split.json')
    approved['status'] = 'approved_for_from_scratch_training'
    for row in rows:
        row.update(candidate_role=row['split'], role_index=row['split_index'],
                   height_band=row['height_band_relative_to_core'])
    write(run / 'approved_split.json', approved)
    data = run / 'dataset'
    (data / 'openMVG').mkdir(parents=True)
    source_data = selection / 'selected_dataset'
    camera_ids = {}
    sources = []
    for split, count in [('train', 18), ('test', 6)]:
        images = data / (split + '_images')
        images.mkdir()
        scene = read(source_data / f'openMVG/data_openmvg_{split}.json')
        expected = {r['image_id'] for r in rows if r['split'] == split}
        actual = {Path(v['value']['ptr_wrapper']['data']['filename']).stem for v in scene['views']}
        assert actual == expected and len(actual) == count
        camera_ids[split] = {v['key'] for v in scene['views']}
        assert camera_ids[split] == {v['key'] for v in scene['extrinsics']}
        for view in scene['views']:
            v = view['value']['ptr_wrapper']['data']
            assert v['id_view'] == view['key'] and v['id_pose'] == view['key']
            source = (Path(scene['root_path']) / v['filename']).resolve(strict=True)
            with Image.open(source) as im:
                assert im.size == (2048, 1024) and im.mode == 'RGB'
            (images / v['filename']).symlink_to(source)
            sources.append(dict(split=split, image_id=source.stem, **ref(source)))
        scene['root_path'] = str(images)
        write(data / f'openMVG/data_openmvg_{split}.json', scene)
        (data / f'{split}.txt').write_text('\n'.join(r['image_id'] for r in sorted(
            rows, key=lambda r: r['split_index']) if r['split'] == split) + '\n')
    assert camera_ids['train'].isdisjoint(camera_ids['test'])
    structure = read(selection / 'selected_sfm/structure_filtered.json')['structure']
    observed = set()
    for point in structure:
        obs = {o['key'] for o in point['value']['observations']}
        assert len(obs) >= 3 and obs <= camera_ids['train']
        observed.update(obs)
    assert observed == camera_ids['train']
    cloud = source_data / 'openMVG/scene.ply'
    assert sha(cloud) == '864a6c7e04de5367d5d0bcfc2badb2922730ef24d5583b0b3980df3d1d9f27f8'
    assert len(PlyData.read(cloud)['vertex']) == len(structure) == 1347
    shutil.copy2(cloud, data / 'openMVG/scene.ply')
    shutil.copy2(selection / 'sfm_quality.json', run / 'sfm_quality.json')
    shutil.copy2(selection / 'verification.json', run / 'selection_verification.json')
    write(run / 'data_provenance.json', dict(selection=ref(selection / 'selected_split.json'),
        images=sources, initial_cloud=ref(data / 'openMVG/scene.ply'),
        training_structure=ref(selection / 'selected_sfm/structure_filtered.json'),
        camera_jsons=[ref(data / f'openMVG/data_openmvg_{s}.json') for s in ['train', 'test']],
        train_ids=sorted(camera_ids['train']), test_ids=sorted(camera_ids['test']),
        test_observations_in_initial_cloud=0, initial_cloud_points=1347,
        test_images_consulted_for_split_design=True, prior_checkpoint_used=False))
    return data, camera_ids['train']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', default='omnigs_compact18_6_20261025')
    p.add_argument('--gpu', type=int, default=0)
    args = p.parse_args()
    assert Path(args.run).name == args.run and args.run not in ['.', '..']
    run = ROOT / 'runs' / args.run
    run.mkdir(exist_ok=False)
    state = dict(status='running', phase='preparing', started=time.time(), controller_pid=os.getpid())
    child = None
    env = {k: v for k, v in os.environ.items() if not k.startswith(('PF_', 'SGS_'))}
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), PF_MODE='0', SGS_MODE='0', OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')

    def update(**kwargs):
        state.update(kwargs, updated=time.time())
        write(run / 'status.json', state)

    def execute(phase, command):
        nonlocal child
        write(run / (phase + '_command.json'), dict(command=list(map(str, command)),
              environment={k: env[k] for k in ['CUDA_VISIBLE_DEVICES', 'PF_MODE', 'SGS_MODE', 'OMP_NUM_THREADS']}))
        started = time.time()
        with (run / (phase + '.log')).open('w') as stream:
            child = subprocess.Popen(list(map(str, command)), cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        update(phase=phase, child_pid=child.pid, phase_started=started)
        while child.poll() is None:
            update(phase_seconds=time.time() - started)
            time.sleep(5)
        if child.returncode:
            raise RuntimeError(f'{phase} exited {child.returncode}; inspect {run / (phase + ".log")}')
        write(run / (phase + '_timing.json'), dict(seconds=time.time() - started, returncode=child.returncode))

    def stop(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        import numpy as np
        from plyfile import PlyData
        selection = ROOT / 'runs/iris_camera_analysis_20261025/trajectory_overlap_split_20261025'
        trainer = BASE / 'OmniGS_PipeFilter/code/bin/train_egonerf_ricoh360'
        exporter = BASE / 'OmniGS_PipeFilter/code/bin/pipefilter_export'
        rasterizer = BASE / 'OmniGS_PipeFilter/code/lib/libcuda_rasterizer.so'
        assert sha(trainer) == '602e0ddd6b57948bd1cb439d4e2b507d7552bbcd48ac10aff56791f38c823244'
        assert sha(ROOT / 'configs/baseline.yaml') == '1d799fe05ecfec289de8e797712837ad2da18dfa1aaa85241f668806257a346e'
        assert sha(rasterizer) == '0a321ee41aa993679a061bd641a063c5da98e582caa529ba95d660277d5240d5'
        config = run / 'baseline.yaml'
        shutil.copy2(ROOT / 'configs/baseline.yaml', config)
        data, train_ids = prepare(run, selection)
        write(run / 'experiment.json', dict(algorithm='Original OmniGS RGB baseline; native PF_MODE=0 / SGS_MODE=0',
            train_views=18, test_views=6, validation_views=0, iterations=32010, resolution=[2048, 1024],
            seed=0, initial_points=1347,
            initialization='Verified training-only fixed-pose SfM cloud from exactly the approved 18 views; fresh Gaussian optimization',
            trainer=ref(trainer), exporter=ref(exporter), rasterizer=ref(rasterizer), config=ref(config),
            controller=ref(__file__), initial_cloud=ref(data / 'openMVG/scene.ply'), gpu=args.gpu,
            prior_checkpoint_used=False, M1_Wan360_or_M2_M5_enabled=False,
            test_policy='Final 32010 checkpoint only; no validation or test-based checkpoint selection.',
            split_design='Test-image feature overlap was consulted to choose the user-approved interpolation split.'))
        model = run / 'model'
        model.mkdir()
        execute('training', [trainer, config, data, model, 'no_viewer'])
        checkpoint = model / '32010_shutdown/ply/point_cloud/iteration_32010/point_cloud.ply'
        cloud = PlyData.read(checkpoint)['vertex'].data
        assert all(np.isfinite(cloud[k]).all() for k in cloud.dtype.names if k != 'opacity')
        assert not np.isnan(cloud['opacity']).any()
        used = (model / 'used_times/keyframe_used_timesfinal.txt').read_text()
        counts = {int(k): int(n) for k, n in re.findall(r'^(\d+) (\d+) \d+$', used, re.M)}
        assert set(counts) == train_ids and min(counts.values()) > 0 and sum(counts.values()) == 32010, counts
        assert 'Loaded number of point3D: 1347' in (run / 'training.log').read_text(errors='replace')
        write(run / 'training_result.json', dict(status='completed', iterations=32010, final_gaussians=len(cloud),
            checkpoint=ref(checkpoint), per_camera_updates=counts, initial_points=1347,
            opacity_logit_positive_inf=int(np.isposinf(cloud['opacity']).sum()),
            opacity_logit_negative_inf=int(np.isneginf(cloud['opacity']).sum())))
        for split in ['test', 'train']:
            out = run / 'evaluation' / split
            write(out / 'request.json', dict(split=split, camera_json=str(data / f'openMVG/data_openmvg_{split}.json'),
                width=2048, height=1024, views=[], diagnostics=False, moments=False, benchmark=False, training_step=False))
            execute('render_' + split, [exporter, config, data, checkpoint, out / 'request.json', out])
        execute('metrics', [Path('/mnt/sda/ubuntu/anaconda3/envs/lkx_odgs/bin/python'),
            ROOT / 'tools/evaluate_omnigs_sparse18.py', '--run', run, '--gpu', args.gpu])
        execute('report', [sys.executable, ROOT / 'tools/report_omnigs_sparse18.py', '--run', run])
        update(status='completed', phase='report_complete', completed=time.time())
        print('TRAIN_EVALUATE_REPORT_COMPLETE', run, flush=True)
    except BaseException as error:
        update(status='failed', reason=str(error))
        (run / 'error.log').write_text(traceback.format_exc())
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)


if __name__ == '__main__':
    main()
