"""Training-free geometric/visual diagnostics for replacing IRIS 1-52 and 1-53."""
import copy
import itertools
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_omnigs_sparse18 import ROOT, BASE, read, write, sha, filter_matches, export_cloud

ANALYSIS = ROOT / 'runs/iris_camera_analysis_20261025'
OUT = ANALYSIS / 'replacement_52_53_20261025'
RUN = ROOT / 'runs/omnigs18_6_20261025_v1'
CANDIDATES = ['1-31', '1-32', '1-33', '1-34', '1-35', '1-45', '1-46', '1-47', '1-48', '1-49',
              '1-50', '1-51', '1-54', '1-56', '1-57', '1-60', '1-70', '2-45', '2-46']
BIN = BASE / 'deps/openmvg-build/Linux-x86_64-Release'


def run_command(command, log):
    env = dict(os.environ, OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with Path(log).open('w') as stream:
        subprocess.run(list(map(str, command)), env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def records():
    all_rows = read(ANALYSIS / 'camera_inventory.json')['cameras']
    by = {x['camera_id']: x for x in all_rows}
    selected = read(RUN / 'approved_split.json')['selected_cameras']
    return by, selected


def xyz(row):
    return np.array([row[k] for k in ['x_m', 'y_m', 'z_m']])


def pool():
    by, selected = records()
    OUT.mkdir(exist_ok=False)
    mvg = OUT / 'pool_sfm'
    images = OUT / 'pool_images'
    mvg.mkdir()
    images.mkdir()
    sfm = read(RUN / 'sfm/sfm_data_train_fixed.json')
    original = [x for x in selected if x['candidate_role'] == 'train']
    assert {v['value']['ptr_wrapper']['data']['filename'] for v in sfm['views']} == {x['image_id'] + '.png' for x in original}
    assert not set(CANDIDATES) & {x['camera_id'] for x in selected}
    pool_rows = [by[x['camera_id']] for x in original] + [by[x] for x in CANDIDATES]
    origin = np.array(read(RUN / 'dataset/coordinate_transform.json')['origin_m'])
    basis = np.array([[0., 0., -1.], [1., 0., 0.], [0., -1., 0.]])
    for i, row in enumerate(pool_rows):
        source = BASE / 'datasets/iris_full_A_official_2k/images' / (row['image_id'] + '.png')
        (images / source.name).symlink_to(source)
        if i < 18:
            for suffix in ['.feat', '.desc']:
                shutil.copy2(RUN / 'sfm' / (row['image_id'] + suffix), mvg / (row['image_id'] + suffix))
            continue
        assert sha(row['rmx_source']) == row['rmx_sha256']
        raw = Path(row['rmx_source']).read_bytes()
        text = raw.decode('utf-16' if raw[:2] in (b'\xff\xfe', b'\xfe\xff') else 'utf-8-sig')
        a = [list(map(float, line.split())) for line in text.splitlines() if line.strip() and not line.startswith('#')]
        C = np.array(a[0]) / 1000
        np.testing.assert_allclose(C, xyz(row), atol=1e-8, rtol=0)
        R = (Rotation.from_rotvec(np.array(a[1]) * a[2][0]).as_matrix() @ basis).T
        sfm['views'].append({'key': i, 'value': {'polymorphic_id': 1073741824, 'ptr_wrapper': {
            'id': 2147483649 + 2 * i, 'data': {'local_path': '', 'filename': source.name, 'width': 2048,
            'height': 1024, 'id_view': i, 'id_intrinsic': 0, 'id_pose': i}}}})
        sfm['extrinsics'].append({'key': i, 'value': {'rotation': R.tolist(), 'center': (C - origin).tolist()}})
    sfm['root_path'] = str(images)
    sfm['intrinsics'][0]['value']['ptr_wrapper']['id'] = 2147484000
    shutil.copy2(RUN / 'dataset/train_images/mask.png', images / 'mask.png')
    shutil.copy2(RUN / 'sfm/image_describer.json', mvg / 'image_describer.json')
    write(mvg / 'sfm_data_train_fixed.json', sfm)
    write(OUT / 'pool_rows.json', pool_rows)
    run_command([BIN / 'openMVG_main_ComputeFeatures', '-i', mvg / 'sfm_data_train_fixed.json', '-o', mvg,
                 '-m', 'SIFT', '-p', 'ULTRA', '-n', '8'], OUT / 'features.log')
    pairs = [(i, j) for i, j in itertools.combinations(range(len(pool_rows)), 2) if j >= 18]
    (mvg / 'new_pairs.txt').write_text(''.join(f'{i} {j}\n' for i, j in pairs))
    run_command([BIN / 'openMVG_main_ComputeMatches', '-i', mvg / 'sfm_data_train_fixed.json', '-o', mvg / 'new_matches.txt',
                 '-p', mvg / 'new_pairs.txt', '-r', '.8', '-n', 'FASTCASCADEHASHINGL2'], OUT / 'matching.log')
    # Preserve every original pair and descriptor exactly for a controlled comparison.
    (mvg / 'matches.putative.txt').write_text((RUN / 'sfm/matches.putative.txt').read_text() + '\n' + (mvg / 'new_matches.txt').read_text())
    filter_matches(mvg)
    edges = read(mvg / 'fixed_pose_match_report.json')['edges']
    keep = set(range(18)) - {10, 11}
    report = []
    for i, row in enumerate(pool_rows):
        if i in keep:
            continue
        connections = [(pool_rows[j if k == i else k]['camera_id'], n) for k, j, n in edges
                       if (k == i and j in keep) or (j == i and k in keep)]
        nearest = sorted((float(np.linalg.norm(xyz(row) - xyz(pool_rows[k]))), pool_rows[k]['camera_id']) for k in keep)
        report.append(dict(camera_id=row['camera_id'], image_id=row['image_id'], source_group=row['source_group'],
              has_official_semantics=row['has_official_semantics'], xyz=xyz(row).tolist(),
              height_band=row['height_band_relative_to_core'], base16_connected_cameras=len(connections),
              base16_geometric_matches=sum(n for _, n in connections), connections=sorted(connections, key=lambda x: -x[1]),
              nearest_base16=nearest[:3], min_distance_to_removed_m=float(min(np.linalg.norm(xyz(row)-xyz(by[k])) for k in ['1-52', '1-53']))))
    write(OUT / 'candidate_connections.json', sorted(report, key=lambda x: -x['base16_geometric_matches']))
    for page, start in enumerate(range(0, len(report), 8), 1):
        entries = report[start:start+8]
        canvas = Image.new('RGB', (1600, 450 * ((len(entries) + 1) // 2)), '#f6f8fa')
        d = ImageDraw.Draw(canvas)
        for k, row in enumerate(entries):
            x, y = (k % 2) * 800, (k // 2) * 450
            im = Image.open(images / (row['image_id'] + '.png')).convert('RGB').resize((780, 390))
            canvas.paste(im, (x+10, y+52))
            text = f"{row['camera_id']} | {'official' if row['has_official_semantics'] else 'no official'} | {row['height_band']} | base links={row['base16_connected_cameras']} / matches={row['base16_geometric_matches']}"
            d.text((x+10, y+8), text, fill='#132b40')
            d.text((x+10, y+28), row['image_id'], fill='#132b40')
        canvas.save(OUT / f'candidate_contact_{page}.jpg', quality=94)
    write(OUT / 'analysis_status.json', {'status': 'pool_complete', 'training_started': False,
          'candidate_count': len(CANDIDATES), 'test_images_used_for_features': 0})


def subset_matches(source, target, indices):
    with Path(source).open() as stream, Path(target).open('w') as out:
        while True:
            line = stream.readline()
            if not line:
                break
            if not line.strip():
                continue
            i, j = map(int, line.split())
            nline = stream.readline()
            obs = [stream.readline() for _ in range(int(nline))]
            if i in indices and j in indices:
                out.write(line + nline + ''.join(obs))


def spatial(rows, tests, core):
    points = np.array([xyz(x) for x in rows])
    hull2, hull3 = ConvexHull(points[:, :2]), ConvexHull(points)
    supports = []
    for test in tests:
        p = xyz(test)
        best = None
        same = [x for x in rows if abs(x['z_m'] - test['z_m']) <= .25]
        for triangle in itertools.combinations(same, 3):
            t = np.array([xyz(x) for x in triangle])
            mat = np.vstack([t[:, :2].T, np.ones(3)])
            if abs(np.linalg.det(mat)) < 1e-9:
                continue
            weights = np.linalg.solve(mat, [p[0], p[1], 1.])
            if min(weights) < -1e-9:
                continue
            distance = np.linalg.norm(t - p, axis=1)
            if best is None or max(distance) < best['max_distance_m']:
                best = dict(ids=[x['camera_id'] for x in triangle], max_distance_m=float(max(distance)), weights=weights.tolist())
        supports.append(dict(camera_id=test['camera_id'], nearest_train_3d_m=float(np.linalg.norm(points - p, axis=1).min()),
              inside_xy_hull=bool(np.all(hull2.equations[:, :2] @ p[:2] + hull2.equations[:, 2] <= 1e-8)),
              inside_xyz_hull=bool(np.all(hull3.equations[:, :3] @ p + hull3.equations[:, 3] <= 1e-8)),
              same_height_enclosing_triangle=best))
    nearest = np.min(np.linalg.norm(np.array([xyz(x) for x in core])[:, None, :] - points[None, :, :], axis=2), axis=1)
    return dict(test_support=supports, main_core_nearest_train_m_quantiles=np.quantile(nearest, [0, .5, .9, 1]).tolist(),
                train_xy_hull_area_m2=float(hull2.volume), official_train_count=sum(x['has_official_semantics'] for x in rows))


def variant(name, replacements):
    by, selected = records()
    original = [x for x in selected if x['candidate_role'] == 'train']
    rows = original if not replacements else [x for x in original if x['camera_id'] not in ['1-52', '1-53']] + [by[k] for k in replacements]
    assert len(rows) == 18
    ids = {x['camera_id'] for x in rows}
    pool_rows = read(OUT / 'pool_rows.json')
    indices = {i for i, x in enumerate(pool_rows) if x['camera_id'] in ids}
    assert len(indices) == 18
    d = OUT / 'variants' / name
    mvg, data = d / 'sfm', d / 'dataset'
    mvg.mkdir(parents=True, exist_ok=False)
    (data / 'train_images').mkdir(parents=True)
    (data / 'openMVG').mkdir()
    sfm = read(OUT / 'pool_sfm/sfm_data_train_fixed.json')
    sfm['views'] = [v for v in sfm['views'] if v['key'] in indices]
    sfm['extrinsics'] = [v for v in sfm['extrinsics'] if v['key'] in indices]
    sfm['root_path'] = str(data / 'train_images')
    for row in rows:
        source = OUT / 'pool_images' / (row['image_id'] + '.png')
        (data / 'train_images' / source.name).symlink_to(source.resolve())
    write(mvg / 'sfm_data_train_fixed.json', sfm)
    subset_matches(OUT / 'pool_sfm/matches.fixed_pose.txt', mvg / 'matches.fixed_pose.txt', indices)
    run_command([BIN / 'openMVG_main_ComputeStructureFromKnownPoses', '-i', mvg / 'sfm_data_train_fixed.json',
                 '-m', OUT / 'pool_sfm', '-f', mvg / 'matches.fixed_pose.txt', '-d', '-r', '4',
                 '-o', mvg / 'structure.json'], d / 'triangulation.log')
    export_cloud(mvg, data)
    s = spatial(rows, [x for x in selected if x['candidate_role'] == 'test'], [x for x in by.values() if x['is_main_core']])
    quality = read(d / 'sfm_quality.json')
    result = dict(name=name, replacements=replacements, training_started=False, sfm=quality, spatial=s,
                  retained_train_ids=[x['camera_id'] for x in rows], unchanged_test_ids=[x['camera_id'] for x in selected if x['candidate_role']=='test'])
    write(d / 'comparison.json', result)
    print(name, 'points', quality['exported_points'], 'supported', quality['supported_train_cameras'],
          'official', s['official_train_count'], 'hull_area', s['train_xy_hull_area_m2'], flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        pool()
    else:
        variant(sys.argv[1], sys.argv[2:])
