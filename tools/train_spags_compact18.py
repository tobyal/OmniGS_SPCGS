"""Fresh native SPaGS on the exact compact 18/6 OmniGS comparison inputs."""
import argparse
import copy
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from run_omnigs_sparse18 import ROOT, BASE, read, write, ref, sha

REFERENCE = ROOT / 'runs/omnigs_compact18_6_20261025'
FRAMEWORK = BASE / 'nerficg'


def prepare(run):
    import yaml
    assert read(REFERENCE / 'status.json')['status'] == 'completed'
    for name in ['approved_split.json', 'sfm_quality.json', 'data_provenance.json']:
        shutil.copy2(REFERENCE / name, run / ('reference_' + name if name == 'data_provenance.json' else name))
    dataset = run / 'dataset'
    (dataset / 'imgs').mkdir(parents=True)
    (dataset / 'openMVG').mkdir()
    mappings = []
    for split, count in [('train', 18), ('test', 6)]:
        scene = read(REFERENCE / f'dataset/openMVG/data_openmvg_{split}.json')
        old_poses = {e['key']: e['value'] for e in scene['extrinsics']}
        assert len(scene['views']) == count
        extrinsics = []
        for i, view in enumerate(scene['views']):
            v = view['value']['ptr_wrapper']['data']
            old_id = v['id_pose']
            source = (Path(scene['root_path']) / v['filename']).resolve(strict=True)
            (dataset / 'imgs' / v['filename']).symlink_to(source)
            mappings.append(dict(split=split, name=Path(v['filename']).stem, source_pose_id=old_id, adapted_pose_id=i))
            extrinsics.append(dict(key=i, value=copy.deepcopy(old_poses[old_id])))
            view['key'] = v['id_view'] = v['id_pose'] = i
        scene['extrinsics'] = extrinsics
        scene['root_path'] = str(dataset / 'imgs')
        write(dataset / f'openMVG/data_openmvg_{split}.json', scene)
        shutil.copy2(REFERENCE / f'dataset/{split}.txt', dataset / f'{split}.txt')
    shutil.copy2(REFERENCE / 'dataset/openMVG/scene.ply', dataset / 'openMVG/scene.ply')
    assert sha(dataset / 'openMVG/scene.ply') == '864a6c7e04de5367d5d0bcfc2badb2922730ef24d5583b0b3980df3d1d9f27f8'
    write(run / 'camera_id_adapter.json', mappings)
    source_config = FRAMEWORK / 'configs/iris_xphase1_unified.yaml'
    cfg = yaml.safe_load(source_config.read_text())
    cfg['GLOBAL'].update(RANDOM_SEED=0, GPU_INDICES=[0])
    cfg['DATASET'].update(PATH=str(dataset), IMAGE_SCALE_FACTOR=1.0, NORMALIZE_CUBE=None,
                          NORMALIZE_RECENTER=False, TO_DEVICE=False, BACKGROUND_COLOR=[0., 0., 0.])
    cfg['TRAINING'].update(LOAD_CHECKPOINT=None, MODEL_NAME=run.name, NUM_ITERATIONS=32010,
                           RUN_VALIDATION=False, ACTIVATE_TIMING=True)
    cfg['TRAINING']['BACKUP'].update(INTERVAL=-1, FINAL_CHECKPOINT=True, TRAINING_STATE=False,
                                    RENDER_TESTSET=False, RENDER_TRAINSET=False, RENDER_VALSET=False)
    cfg['TRAINING']['WANDB']['ACTIVATE'] = False
    (run / 'spags.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    source_files = [FRAMEWORK / 'src' / p for p in [
        'Methods/SPaGS/Trainer.py', 'Methods/SPaGS/Model.py', 'Methods/SPaGS/Renderer.py',
        'Methods/SPaGS/Loss.py', 'Datasets/Ricoh360.py', 'Methods/Base/Trainer.py', 'Methods/Base/utils.py']]
    write(run / 'experiment.json', dict(algorithm='SPaGS native implementation in NeRFICG',
        comparison_run=str(REFERENCE), train_views=18, test_views=6, validation_views=0,
        iterations=32010, resolution=[2048, 1024], seed=0, initial_points=1347,
        initial_cloud=ref(dataset / 'openMVG/scene.ply'), config=ref(run / 'spags.yaml'),
        source_config=ref(source_config), controller=ref(__file__), implementation=[ref(p) for p in source_files],
        framework_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=FRAMEWORK, text=True).strip(),
        from_scratch=True, prior_checkpoint_used=False, test_observations_in_initial_cloud=0,
        test_policy='Final 32010-step model only; full ERP; no validation or test-based checkpoint choice.',
        common_settings='Identical 18/6 image pixels, official poses, initial cloud, resolution, seed, black background, SH degree 3 and training budget.',
        native_method_settings='SPaGS retains its default spherical renderer, 3D filter, distance scaling and densification (start=200, gradient threshold=5e-5).'))


def worker(run):
    import numpy as np
    from PIL import Image
    from plyfile import PlyData
    import torch
    sys.argv = [sys.argv[0]]
    sys.modules['ICGui'] = None
    sys.path.insert(0, str(FRAMEWORK / 'src'))
    import Framework
    Framework.setup(config_path=str(run / 'spags.yaml'), require_custom_config=True)
    from Implementations import Methods, Datasets
    from Methods.SPaGS.Trainer import SPaGSTrainer
    from Methods.SPaGS.Model import SPaGSModel
    from Methods.Base.utils import preTrainingCallback, trainingCallback
    dataset = Datasets.getDataset('Ricoh360', str(run / 'dataset'))
    assert len(dataset.data['train']) == 18 and len(dataset.data['test']) == 6 and not dataset.data['val']
    Q = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])
    S = np.diag([1., 1., -1.])
    raw = PlyData.read(run / 'dataset/openMVG/scene.ply')['vertex'].data
    xyz = np.stack([raw[k] for k in ['x', 'y', 'z']], axis=-1)
    rgb = np.stack([raw[k] for k in ['red', 'green', 'blue']], axis=-1) / 255.
    assert len(xyz) == 1347
    np.testing.assert_allclose(dataset.point_cloud.positions.cpu().numpy(), xyz @ Q.T, atol=2e-6)
    np.testing.assert_allclose(dataset.point_cloud.colors.cpu().numpy(), rgb, atol=1e-7)
    names = {}
    max_pose = max_rgb = 0.
    for split in ['train', 'test']:
        scene = read(REFERENCE / f'dataset/openMVG/data_openmvg_{split}.json')
        poses = {e['key']: e['value'] for e in scene['extrinsics']}
        views = {Path(v['value']['ptr_wrapper']['data']['filename']).stem:
                 v['value']['ptr_wrapper']['data'] for v in scene['views']}
        names[split] = sorted(views)
        for name, camera in zip(names[split], dataset.data[split]):
            pose = poses[views[name]['id_pose']]
            expected = np.eye(4)
            expected[:3, :3] = Q @ np.array(pose['rotation']).T @ S
            expected[:3, 3] = Q @ np.array(pose['center'])
            error = float(np.abs(camera.c2w.cpu().numpy() - expected).max())
            max_pose = max(max_pose, error)
            assert error < 2e-6 and (camera.width, camera.height) == (2048, 1024)
            pixels = np.asarray(Image.open(Path(scene['root_path']) / views[name]['filename']), dtype=np.float32) / 255.
            error = float(np.abs(camera.rgb.cpu().numpy().transpose(1, 2, 0) - pixels).max())
            max_rgb = max(max_rgb, error)
            assert error < 1e-7
    write(run / 'adapter_validation.json', dict(status='passed', train=18, test=6, validation=0,
        initial_points=1347, identical_input_cloud=True, maximum_pose_element_error=max_pose,
        maximum_image_pixel_error=max_rgb, camera_world_conversion='Q @ R.T @ diag(1,1,-1); center=Q @ C'))
    print('INPUT_VALIDATED train=18 test=6 val=0 points=1347 pose_error=', max_pose, 'rgb_error=', max_rgb, flush=True)

    class ComparisonTrainer(SPaGSTrainer):
        @preTrainingCallback(priority=30)
        def monitor_setup(self, _, data):
            assert self.model.num_iterations_trained == 0
            assert len(self.model.gaussians.get_positions) == 1347
            self.update_counts = [0] * 18
            original_get = self.train_sampler.get

            def counted_get(*args, **kwargs):
                sample = original_get(*args, **kwargs)
                self.update_counts[sample['sample_id']] += 1
                return sample

            self.train_sampler.get = counted_get
            self.last_loss = None

            def record_loss(module, inputs, output):
                self.last_loss = output.detach()

            self.loss.register_forward_hook(record_loss)
            self.loop_started = time.time()

        @trainingCallback(priority=0)
        @torch.no_grad()
        def progress(self, iteration, data):
            step = iteration + 1
            if step % 100 == 0 or step == self.NUM_ITERATIONS:
                points = self.model.gaussians.get_positions
                assert torch.isfinite(points).all() and torch.isfinite(self.last_loss)
                row = dict(step=step, total=self.NUM_ITERATIONS, gaussians=len(points),
                    loss_last=float(self.last_loss), elapsed_seconds=time.time()-self.loop_started,
                    training_updates=sum(self.update_counts), updated=time.time())
                write(run / 'progress.json', row)
                with (run / 'training_history.jsonl').open('a') as stream:
                    import json
                    stream.write(json.dumps(row) + '\n')
                print('PROGRESS', row, flush=True)

    model = Methods.getModel('SPaGS', name=run.name)
    model.output_directory = run / 'model'
    renderer = Methods.getRenderer('SPaGS', model)
    trainer = ComparisonTrainer(model=model, renderer=renderer)
    started = time.time()
    trainer.run(dataset)
    elapsed = time.time() - started
    assert model.num_iterations_trained == 32010
    assert sum(trainer.update_counts) == 32010 and min(trainer.update_counts) > 0
    checkpoint = run / 'model/checkpoints/final.pt'
    assert checkpoint.is_file()
    # Render the persisted checkpoint using the original inference loader.
    saved = SPaGSModel.load(str(checkpoint))
    assert saved.num_iterations_trained == 32010
    for p in saved.parameters():
        assert torch.isfinite(p).all()
    write(run / 'training_timing.json', dict(seconds=elapsed))
    write(run / 'training_result.json', dict(status='completed', iterations=32010,
        final_gaussians=len(saved.gaussians.get_positions), initial_points=1347, checkpoint=ref(checkpoint),
        per_camera_updates=dict(zip(names['train'], trainer.update_counts)), reloaded_checkpoint_verified=True))
    renderer = Methods.getRenderer('SPaGS', saved)
    saved.eval()
    with torch.inference_mode():
        for split in ['test', 'train']:
            out = run / 'evaluation' / split
            (out / 'rgb').mkdir(parents=True)
            frames = []
            for name, camera in zip(names[split], dataset.data[split]):
                dataset.camera.setProperties(camera.toDefaultDevice())
                pred = renderer.renderImage(dataset.camera, to_chw=True)['rgb'].clamp(0, 1)
                assert tuple(pred.shape) == (3, 1024, 2048) and torch.isfinite(pred).all()
                array = pred.cpu().numpy().astype('<f4')
                array.tofile(out / 'rgb' / (name + '.f32'))
                Image.fromarray(np.rint(array.transpose(1, 2, 0)*255).astype(np.uint8)).save(out / 'rgb' / (name+'.png'))
                frames.append(dict(name=name))
                print('EXPORTED', split, name, flush=True)
            write(out / 'render_manifest.json', dict(width=2048, height=1024, frames=frames, method='SPaGS', checkpoint=ref(checkpoint)))
    Framework.teardown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='spags_compact18_6_20261025')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    assert Path(args.run).name == args.run and args.run not in ['.', '..']
    run = ROOT / 'runs' / args.run
    if args.worker:
        worker(run)
        return
    run.mkdir(exist_ok=False)
    state = dict(status='running', phase='preparing', started=time.time(), controller_pid=os.getpid())
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    child = None

    def update(**kwargs):
        state.update(kwargs, updated=time.time())
        write(run / 'status.json', state)

    def execute(phase, command):
        nonlocal child
        write(run / (phase + '_command.json'), dict(command=list(map(str, command)), physical_gpu=args.gpu))
        with (run / (phase+'.log')).open('w') as stream:
            child = subprocess.Popen(list(map(str, command)), cwd=FRAMEWORK, env=env, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        update(phase=phase, child_pid=child.pid, phase_started=time.time())
        code = child.wait()
        if code:
            raise RuntimeError(f'{phase} exited {code}; inspect {phase}.log')

    try:
        update()
        prepare(run)
        execute('training', ['/mnt/sda/ubuntu/anaconda3/envs/lkx_spags/bin/python', __file__, '--run', args.run, '--worker'])
        execute('metrics', ['/mnt/sda/ubuntu/anaconda3/envs/lkx_odgs/bin/python', ROOT / 'tools/evaluate_omnigs_sparse18.py',
                            '--run', run, '--gpu', args.gpu, '--method', 'SPaGS'])
        execute('report', [sys.executable, ROOT / 'tools/report_spags_compact18.py', '--run', run, '--reference', REFERENCE])
        update(status='completed', phase='report_complete', completed=time.time())
    except BaseException as error:
        update(status='failed', reason=str(error))
        (run / 'error.log').write_text(traceback.format_exc())
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()


if __name__ == '__main__':
    main()
