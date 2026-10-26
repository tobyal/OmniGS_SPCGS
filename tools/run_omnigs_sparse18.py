"""Train the approved IRIS 18/6 split with the original OmniGS RGB baseline.

Fresh training-only SIFT matches and fixed-pose triangulation; no Wan360,
semantic supervision, previous cloud, previous checkpoint, or validation set.
"""
import argparse
import collections
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
W, H = 2048, 1024


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def ref(path):
    path = Path(path)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def bearings(xy):
    import numpy as np
    lon = (xy[:, 0] / W - .5) * 2 * np.pi
    lat = (.5 - xy[:, 1] / H) * np.pi
    return np.column_stack([np.cos(lat) * np.sin(lon), -np.sin(lat), np.cos(lat) * np.cos(lon)])


def prepare(run, proposal):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from PIL import Image
    selected = read(proposal)['selected_cameras']
    train = [x for x in selected if x['candidate_role'] == 'train']
    test = [x for x in selected if x['candidate_role'] == 'test']
    assert len(train) == 18 and len(test) == 6 and len({x['image_id'] for x in selected}) == 24
    data = run / 'dataset'
    (data / 'openMVG').mkdir(parents=True)
    shutil.copy2(proposal, run / 'approved_split.json')
    source_rgb = BASE / 'datasets/iris_full_A_official_2k/images'
    rotations, centers = {}, {}
    basis = np.array([[0., 0., -1.], [1., 0., 0.], [0., -1., 0.]])
    image_refs = []
    for row in selected:
        assert sha(row['rmx_source']) == row['rmx_sha256']
        raw = Path(row['rmx_source']).read_bytes()
        text = raw.decode('utf-16' if raw[:2] in (b'\xff\xfe', b'\xfe\xff') else 'utf-8-sig')
        a = [list(map(float, line.split())) for line in text.splitlines() if line.strip() and not line.startswith('#')]
        assert list(map(len, a)) == [3, 3, 1]
        name = row['image_id']
        centers[name] = np.array(a[0]) / 1000.
        np.testing.assert_allclose(centers[name], [row[k] for k in ['x_m', 'y_m', 'z_m']], atol=1e-8, rtol=0)
        assert abs(np.linalg.norm(a[1]) - 1.) < 1e-6
        rotations[name] = (Rotation.from_rotvec(np.array(a[1]) * a[2][0]).as_matrix() @ basis).T
        source = source_rgb / (name + '.png')
        with Image.open(source) as im:
            assert im.size == (W, H) and im.mode == 'RGB'
        folder = data / (row['candidate_role'] + '_images')
        folder.mkdir(exist_ok=True)
        (folder / source.name).symlink_to(source)
        image_refs.append(dict(image_id=name, camera_id=row['camera_id'], split=row['candidate_role'], **ref(source)))
    origin = np.mean([centers[x['image_id']] for x in train], axis=0)
    template = read(BASE / 'datasets/iris_xphase1_work_2k/mvg/sfm_data_train_fixed.json')
    for split, rows, offset in [('train', train, 0), ('test', test, 18)]:
        scene = {'sfm_data_version': '0.3', 'root_path': str(data / (split + '_images')),
                 'views': [], 'intrinsics': copy.deepcopy(template['intrinsics']),
                 'extrinsics': [], 'structure': [], 'control_points': []}
        for i, row in enumerate(rows, offset):
            name = row['image_id']
            scene['views'].append({'key': i, 'value': {'polymorphic_id': 1073741824,
                'ptr_wrapper': {'id': 2147483649 + 2 * i, 'data': {'local_path': '', 'filename': name + '.png',
                 'width': W, 'height': H, 'id_view': i, 'id_intrinsic': 0, 'id_pose': i}}}})
            scene['extrinsics'].append({'key': i, 'value': {'rotation': rotations[name].tolist(),
                                                       'center': (centers[name] - origin).tolist()}})
        write(data / f'openMVG/data_openmvg_{split}.json', scene)
        (data / (split + '.txt')).write_text('\n'.join(x['image_id'] for x in rows) + '\n')
    # Preserve the established feature-only pole/tripod mask. RGB loss uses the full ERP.
    mask = np.full((H, W), 255, dtype=np.uint8)
    mask[:41] = 0
    mask[901:] = 0
    Image.fromarray(mask).save(data / 'train_images/mask.png')
    mvg = run / 'sfm'
    mvg.mkdir()
    shutil.copy2(data / 'openMVG/data_openmvg_train.json', mvg / 'sfm_data_train_fixed.json')
    write(data / 'coordinate_transform.json', {'origin_m': origin.tolist(), 'origin_source': 'mean of 18 training centers only',
          'rotation': 'R_world_to_camera = (R_rmx @ B).T', 'B': basis.tolist(),
          'center': 'C_rmx_mm / 1000 - origin_m', 'pose_optimization': False})
    write(run / 'data_provenance.json', {'proposal': ref(proposal), 'train_count': 18, 'test_count': 6,
          'validation_count': 0, 'images': image_refs, 'coordinate_transform': ref(data / 'coordinate_transform.json'),
          'feature_image_directory': str(data / 'train_images'), 'previous_cloud_or_checkpoint_used': False,
          'test_used_for_features_matching_triangulation_training_or_model_selection': False})
    return data, mvg


def filter_matches(mvg):
    import numpy as np
    sfm = read(mvg / 'sfm_data_train_fixed.json')
    poses = {p['key']: p['value'] for p in sfm['extrinsics']}
    views = {v['key']: v['value']['ptr_wrapper']['data'] for v in sfm['views']}
    cams = {k: (np.array(poses[v['id_pose']]['rotation']), np.array(poses[v['id_pose']]['center'])) for k, v in views.items()}
    ray = {k: bearings(np.loadtxt(mvg / (Path(v['filename']).stem + '.feat'), ndmin=2)[:, :2]) @ cams[k][0] for k, v in views.items()}
    counts = dict(input_pairs=0, input_matches=0, retained_pairs=0, retained_matches=0)
    edges = []
    with (mvg / 'matches.putative.txt').open() as stream, (mvg / 'matches.fixed_pose.txt').open('w') as out:
        while True:
            line = stream.readline()
            if not line:
                break
            if not line.strip():
                continue
            i, j = map(int, line.split())
            assert i in views and j in views and i != j
            n = int(stream.readline())
            ij = np.array([list(map(int, stream.readline().split())) for _ in range(n)], dtype=int)
            counts['input_pairs'] += 1
            counts['input_matches'] += n
            if n == 0:
                continue
            t = cams[j][1] - cams[i][1]
            if np.linalg.norm(t) < .02:
                continue
            a, b = ray[i][ij[:, 0]], ray[j][ij[:, 1]]
            dot = np.sum(a * b, axis=1)
            den = 1 - dot * dot
            sa = (a @ t - dot * (b @ t)) / np.maximum(den, 1e-12)
            sb = (dot * (a @ t) - b @ t) / np.maximum(den, 1e-12)
            triple = np.abs(np.cross(a, b) @ t)
            ea = triple / (np.linalg.norm(np.cross(a, t), axis=1) + 1e-12)
            eb = triple / (np.linalg.norm(np.cross(b, t), axis=1) + 1e-12)
            keep = ((np.maximum(ea, eb) < np.sin(np.deg2rad(.5))) & (sa > .15) & (sb > .15)
                    & (sa < 200) & (sb < 200) & (den > np.sin(np.deg2rad(1)) ** 2))
            ij = ij[keep]
            if len(ij) < 8:
                continue
            out.write(f'{i} {j}\n{len(ij)}\n')
            for x, y in ij:
                out.write(f'{x} {y}\n')
            counts['retained_pairs'] += 1
            counts['retained_matches'] += len(ij)
            edges.append([i, j, len(ij)])
    components, remaining = [], set(views)
    while remaining:
        seen = {min(remaining)}
        for _ in views:
            for i, j, _ in edges:
                if i in seen or j in seen:
                    seen.update([i, j])
        components.append(sorted(seen))
        remaining -= seen
    # Official poses already place all components in the same coordinate frame.
    # Connectivity is a diagnostic, not a reason to alter the approved split.
    counts.update(edges=edges, connected_components=components, symmetric_epipolar_threshold_deg=.5,
                  min_ray_angle_deg=1, min_pair_matches=8, pose_optimization=False)
    write(mvg / 'fixed_pose_match_report.json', counts)


def export_cloud(mvg, data):
    import numpy as np
    from PIL import Image
    from plyfile import PlyData, PlyElement
    sfm, fixed = read(mvg / 'structure.json'), read(mvg / 'sfm_data_train_fixed.json')
    old = {e['key']: e['value'] for e in fixed['extrinsics']}
    new = {e['key']: e['value'] for e in sfm['extrinsics']}
    assert set(old) == set(new)
    dR = max(np.abs(np.array(old[k]['rotation']) - new[k]['rotation']).max() for k in old)
    dC = max(np.abs(np.array(old[k]['center']) - new[k]['center']).max() for k in old)
    assert dR < 1e-12 and dC < 1e-12
    views = {v['key']: v['value']['ptr_wrapper']['data'] for v in fixed['views']}
    images = {k: np.asarray(Image.open(data / 'train_images' / v['filename']).convert('RGB')) for k, v in views.items()}
    points, colors, errors, lengths, landmarks = [], [], [], [], []
    supports, reasons = collections.Counter(), collections.Counter()
    for item in sfm['structure']:
        lm = item['value']
        X = np.array(lm['X'], dtype=float)
        obs = lm['observations']
        if not np.isfinite(X).all():
            reasons['nonfinite'] += 1
            continue
        if len(obs) < 3:
            reasons['fewer_than_three_views'] += 1
            continue
        good, angle, color, ray = [], [], [], []
        for o in obs:
            k, xy = o['key'], np.array(o['value']['x'])
            assert k in views
            p = old[views[k]['id_pose']]
            R, C = np.array(p['rotation']), np.array(p['center'])
            q = R @ (X - C)
            length = np.linalg.norm(q)
            if length < .15 or length > 200:
                continue
            b = bearings(xy[None])[0]
            err = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(q / length, b)), np.dot(q / length, b)))
            if err > .35:
                continue
            u, v = int(round(xy[0])) % W, int(np.clip(round(xy[1]), 0, H - 1))
            good.append(o)
            angle.append(float(err))
            color.append(images[k][v, u])
            ray.append((X - C) / length)
        if len(good) < 3:
            reasons['angular_or_range_filter'] += 1
            continue
        ray = np.array(ray)
        if np.rad2deg(np.arccos(np.clip((ray @ ray.T).min(), -1, 1))) < 2:
            reasons['triangulation_angle_lt_2deg'] += 1
            continue
        points.append(X)
        colors.append(np.median(color, axis=0))
        errors.extend(angle)
        lengths.append(len(good))
        supports.update(o['key'] for o in good)
        landmarks.append(dict(key=item['key'], value=dict(X=X.tolist(), observations=good)))
    assert len(points) > 100, ('Insufficient validated initialization', len(points))
    xyz, rgb = np.asarray(points, dtype=np.float32), np.asarray(colors, dtype=np.uint8)
    verts = np.zeros(len(xyz), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                                     ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    for i, k in enumerate(['x', 'y', 'z']):
        verts[k] = xyz[:, i]
    for i, k in enumerate(['red', 'green', 'blue']):
        verts[k] = rgb[:, i]
    PlyData([PlyElement.describe(verts, 'vertex')], text=False).write(data / 'openMVG/scene.ply')
    total = len(sfm['structure'])
    sfm['structure'] = landmarks
    write(mvg / 'structure_filtered.json', sfm)
    report = {'input_landmarks': total, 'exported_points': len(points), 'rejections': dict(reasons),
              'max_pose_rotation_element_delta': float(dR), 'max_camera_center_delta_m': float(dC),
              'test_observations_used': 0, 'bundle_adjustment': False, 'supported_train_cameras': len(supports),
              'track_length_quantiles_min_median_p90_max': np.quantile(lengths, [0, .5, .9, 1]).tolist(),
              'angular_error_deg_quantiles_min_median_p90_p99_max': np.quantile(errors, [0, .5, .9, .99, 1]).tolist(),
              'camera_point_support': {views[k]['filename']: int(supports[k]) for k in views},
              'point_bounds_min': xyz.min(axis=0).tolist(), 'point_bounds_max': xyz.max(axis=0).tolist(),
              'configuration': {'features': 'SIFT ULTRA / RootSIFT', 'matcher': 'FASTCASCADEHASHINGL2',
              'ratio': .8, 'pairs': '153 exhaustive training-only pairs', 'minimum_views_per_point': 3,
              'maximum_observation_angular_error_deg': .35, 'minimum_track_angle_deg': 2,
              'point_distance_range_m': [.15, 200], 'feature_mask_rows_included': [41, 900]}}
    write(mvg.parent / 'sfm_quality.json', report)
    print('SFM_COMPLETE', len(points), 'points;', len(supports), 'cameras; median error', np.median(errors), flush=True)
    return len(points)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='omnigs18_6_20261025')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    assert Path(args.run).name == args.run and args.run not in ['.', '..']
    run = ROOT / 'runs' / args.run
    run.mkdir(parents=True, exist_ok=False)
    state = dict(status='running', phase='preparing', started=time.time(), controller_pid=os.getpid())
    child = None
    env = {k: v for k, v in os.environ.items() if not k.startswith(('PF_', 'SGS_'))}
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), PF_MODE='0', SGS_MODE='0', OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')

    def update(**kwargs):
        state.update(kwargs, updated=time.time())
        write(run / 'status.json', state)

    def execute(phase, command, sfm=False):
        nonlocal child
        log = run / (phase + '.log')
        subenv = dict(env)
        if sfm:
            subenv.update(OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        write(run / (phase + '_command.json'), {'command': list(map(str, command)),
              'environment': {k: subenv[k] for k in ['CUDA_VISIBLE_DEVICES', 'PF_MODE', 'SGS_MODE', 'OMP_NUM_THREADS']}})
        started = time.time()
        with log.open('w') as stream:
            child = subprocess.Popen(list(map(str, command)), cwd=ROOT, env=subenv, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        update(phase=phase, child_pid=child.pid, phase_started=started)
        while child.poll() is None:
            update(phase_seconds=time.time() - started)
            time.sleep(10)
        if child.returncode:
            raise RuntimeError(f'{phase} exited {child.returncode}; inspect {log}')
        write(run / (phase + '_timing.json'), {'seconds': time.time() - started, 'returncode': child.returncode})

    def stop(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        import numpy as np
        from plyfile import PlyData
        proposal = ROOT / 'runs/iris_camera_analysis_20261025/candidate_18_6_v1/candidate_split.json'
        trainer = BASE / 'OmniGS_PipeFilter/code/bin/train_egonerf_ricoh360'
        exporter = BASE / 'OmniGS_PipeFilter/code/bin/pipefilter_export'
        assert sha(trainer) == '602e0ddd6b57948bd1cb439d4e2b507d7552bbcd48ac10aff56791f38c823244'
        assert sha(ROOT / 'configs/baseline.yaml') == '1d799fe05ecfec289de8e797712837ad2da18dfa1aaa85241f668806257a346e'
        assert sha(BASE / 'OmniGS_PipeFilter/code/lib/libcuda_rasterizer.so') == '0a321ee41aa993679a061bd641a063c5da98e582caa529ba95d660277d5240d5'
        config = run / 'baseline.yaml'
        shutil.copy2(ROOT / 'configs/baseline.yaml', config)
        data, mvg = prepare(run, proposal)
        binary = BASE / 'deps/openmvg-build/Linux-x86_64-Release'
        execute('features', [binary / 'openMVG_main_ComputeFeatures', '-i', mvg / 'sfm_data_train_fixed.json',
                            '-o', mvg, '-m', 'SIFT', '-p', 'ULTRA', '-n', '8'], sfm=True)
        execute('matching', [binary / 'openMVG_main_ComputeMatches', '-i', mvg / 'sfm_data_train_fixed.json',
                            '-o', mvg / 'matches.putative.txt', '-r', '.8', '-n', 'FASTCASCADEHASHINGL2'], sfm=True)
        update(phase='geometric_filter')
        filter_matches(mvg)
        execute('triangulation', [binary / 'openMVG_main_ComputeStructureFromKnownPoses', '-i', mvg / 'sfm_data_train_fixed.json',
                 '-m', mvg, '-f', mvg / 'matches.fixed_pose.txt', '-d', '-r', '4', '-o', mvg / 'structure.json'], sfm=True)
        update(phase='exporting_cloud')
        initial = export_cloud(mvg, data)
        write(run / 'experiment.json', {'algorithm': 'Original OmniGS RGB baseline; native PF_MODE=0 / SGS_MODE=0',
              'train_views': 18, 'test_views': 6, 'validation_views': 0, 'iterations': 32010, 'resolution': [W, H],
              'seed': 0, 'initial_points': initial, 'initialization': 'Fresh 18-training-view fixed-pose OpenMVG triangulation',
              'trainer': ref(trainer), 'exporter': ref(exporter), 'config': ref(config),
              'controller': ref(__file__), 'initial_cloud': ref(data / 'openMVG/scene.ply'),
              'gpu': args.gpu, 'M1_Wan360_or_M2_M5_enabled': False,
              'test_policy': 'Only final 32010 checkpoint, no validation, no test-driven checkpoint selection.'})
        model = run / 'model'
        model.mkdir()
        execute('training', [trainer, config, data, model, 'no_viewer'])
        checkpoint = model / '32010_shutdown/ply/point_cloud/iteration_32010/point_cloud.ply'
        cloud = PlyData.read(checkpoint)['vertex'].data
        assert all(np.isfinite(cloud[k]).all() for k in cloud.dtype.names if k != 'opacity')
        assert not np.isnan(cloud['opacity']).any()
        used = (model / 'used_times/keyframe_used_timesfinal.txt').read_text()
        counts = {int(k): int(n) for k, n in re.findall(r'^(\d+) (\d+) \d+$', used, re.M)}
        assert set(counts) == set(range(18)) and min(counts.values()) > 0 and sum(counts.values()) == 32010
        assert f'Loaded number of point3D: {initial}' in (run / 'training.log').read_text(errors='replace')
        write(run / 'training_result.json', {'status': 'completed', 'iterations': 32010, 'final_gaussians': len(cloud),
              'checkpoint': ref(checkpoint), 'per_camera_updates': counts, 'initial_points': initial,
              'opacity_logit_positive_inf': int(np.isposinf(cloud['opacity']).sum()),
              'opacity_logit_negative_inf': int(np.isneginf(cloud['opacity']).sum())})
        for split in ['test', 'train']:
            out = run / 'evaluation' / split
            request = {'split': split, 'camera_json': str(data / f'openMVG/data_openmvg_{split}.json'),
                       'width': W, 'height': H, 'views': [], 'diagnostics': False, 'moments': False,
                       'benchmark': False, 'training_step': False}
            write(out / 'request.json', request)
            execute('render_' + split, [exporter, config, data, checkpoint, out / 'request.json', out])
        update(status='completed', phase='rendered', completed=time.time())
        print('TRAINING_AND_RENDERING_COMPLETE', run, flush=True)
    except BaseException as error:
        update(status='failed', reason=str(error))
        (run / 'error.log').write_text(traceback.format_exc())
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)


if __name__ == '__main__':
    main()
