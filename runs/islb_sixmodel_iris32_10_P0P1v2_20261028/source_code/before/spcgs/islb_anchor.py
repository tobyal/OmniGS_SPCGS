"""Sparse, measured SfM correspondences with consistent real-view GT.

No SfM edits, generated RGB/CLIP targets, or independent semantic head.
All writes stay in a new _ISLB_anchor_v1 run. Original inputs are read-only.
"""
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from .common import ROOT, read, write, ref
from .m1_geometry import load_anchors, camera_bearings, bearings, angular_pixels, project_erp, quantiles
from .official_teacher import targets

SUFFIX = '_ISLB_anchor_v1'


def sample_label(labels, uv):
    h, w = labels.shape
    u, v = np.rint(uv).astype(int)
    return int(labels[v, u % w]) if 0 <= v < h else 255


def semantic_coordinates(uv, width, height, sw, sh):
    # SfM observes integer-pixel spherical rays. Native ndc2Pix subtracts .5.
    # Do NOT add .5 to SfM coordinates before rescaling.
    return np.asarray(uv) * [sw / width, sh / height] - .5


def depth_buffer(xyz, camera, width):
    uv = project_erp(xyz, camera, width, width // 2)
    ij = np.floor(uv).astype(int)
    ij[:, 0] %= width
    ij[:, 1] = np.clip(ij[:, 1], 0, width // 2 - 1)
    z = np.full((width // 2, width), np.inf)
    np.minimum.at(z, (ij[:, 1], ij[:, 0]), np.linalg.norm(xyz - camera['center'], axis=1))
    return z


def observation_check(xyz, uv, camera, width, height, depth, cfg):
    predicted = camera_bearings(np.asarray(xyz)[None], camera)
    observed = bearings(np.asarray(uv)[None], width, height)
    error = float(angular_pixels(predicted, observed)[0])
    if float((predicted * observed).sum()) <= 0 or error > cfg['max_reprojection_px_2k']:
        return 'reprojection', error
    p = project_erp(np.asarray(xyz)[None], camera, depth.shape[1], depth.shape[0])[0]
    x, y = np.floor(p).astype(int)
    near = depth[np.clip(y, 0, depth.shape[0]-1), x % depth.shape[1]]
    distance = np.linalg.norm(np.asarray(xyz) - camera['center'])
    margin = max(cfg['occlusion_absolute_margin'], cfg['occlusion_relative_margin'] * near)
    if distance > near + margin:
        return 'sparse_depth_occlusion', error
    return None, error


def real_consensus(observations, xyz, cameras, labels, depths, cfg):
    """A conflict is rejected, never voted away. Unique real cameras are required."""
    names = [o['view'] for o in observations]
    if len(set(names)) != len(names) or len(names) < 2:
        return None, 'insufficient_distinct_real_views', []
    classes, errors = [], []
    for o in observations:
        name = o['view']
        if name not in cameras:
            return None, 'not_training_view', errors
        a = labels[name]; h, w = a.shape
        reason, error = observation_check(xyz, o['uv'], cameras[name], w, h, depths[name], cfg)
        errors.append(error)
        if reason:
            return None, 'real_' + reason, errors
        lab = sample_label(a, o['uv'])
        if lab not in range(7):
            return None, 'invalid_real_label', errors
        projected = project_erp(np.asarray(xyz)[None], cameras[name], w, h)[0]
        if sample_label(a, projected) != lab:
            return None, 'real_label_boundary', errors
        classes.append(lab)
    if len(set(classes)) != 1:
        return None, 'real_label_conflict', errors
    return classes[0], None, errors


def prepare(source, run, cfg):
    source, run = Path(source).resolve(), Path(run).resolve()
    assert source != run and run.name.endswith(SUFFIX)
    dest = run / ('01_sparse_anchors' + SUFFIX)
    dest.mkdir()  # refuse to overwrite any prior extraction
    protocol = read(source / '00_dataset/dataset_manifest.json')
    teacher = read(source / '02_semantic_teacher/cache/manifest.json')
    assert teacher['smoothing'] == .05 and teacher['generated_views'] == []
    cameras = {v['name']: v for v in protocol['train']}
    assert set(cameras) == {v['name'] for v in teacher['views']}
    assert not set(cameras) & {v['name'] for v in protocol['test']}
    labels = {}; inputs = []
    for row in protocol['labels']:
        if row['id'] in cameras:
            evidence = ref(row['label']); assert evidence['sha256'] == row['label_sha256']
            inputs.append(evidence); labels[row['id']] = np.array(Image.open(row['label']))
    p0 = source / '01_geometry/P0'; p1 = source / '01_geometry/P1'
    original = np.fromfile(p0 / 'points.f32', 'f4').reshape(-1, 6)
    enhanced = np.fromfile(p1 / 'enhanced_points.f32', 'f4').reshape(-1, 6)
    anchors = load_anchors(p0 / 'structure.json', p0 / 'features', protocol['train'], original[:, :3], dest / ('source_anchor_audit' + SUFFIX + '.json'))
    assert len(anchors['landmarks']) == read(p1 / 'anchor_audit.json')['anchors']
    selected = read(p1 / 'selected_views.json')
    if isinstance(selected, dict): selected = selected['views']
    generated = {v['name']: v for v in selected}
    assert len(generated) == len(selected) and all(v['registered'] for v in selected)
    allcams = {**cameras, **generated}
    depths = {n: depth_buffer(enhanced[:, :3], v, cfg['depth_grid_width']) for n, v in allcams.items()}
    proposed = []; rejected = Counter(); reasons = []
    for v in selected:
        feature_path = p1 / 'features' / (v['name'] + '.feat')
        uv = np.loadtxt(feature_path, dtype='f4', ndmin=2)[:, :2]; inputs.append(ref(feature_path))
        for match in v['correspondences']:
            if not match['inlier']:
                rejected['registration_outlier'] += 1; continue
            aid = match['anchor_id']; a = anchors['landmarks'][aid]
            assert anchors['by_feature'][(match['source_view'], match['source_feature_id'])] == aid
            proposed.append(dict(track='P0:' + str(a['source_landmark_id']), source='P0_registered', xyz=a['xyz'], real_observations=a['observations'], generated_view=v['name'], generated_feature_id=match['generated_feature_id'], generated_uv=uv[match['generated_feature_id']].tolist()))
    for i, a in enumerate(read(p1 / 'observations.json')):
        assert len({o['view'] for o in a['real_observations']}) + 1 >= 3
        if a['generated_view'] in generated:
            proposed.append({**a, 'track': 'P1:' + str(i), 'source': 'P1_added'})
    accepted = []; consensus_cache = {}
    for a in proposed:
        key = a['track']; v = generated[a['generated_view']]
        if key not in consensus_cache:
            consensus_cache[key] = real_consensus(a['real_observations'], a['xyz'], cameras, labels, depths, cfg)
        label, reason, real_errors = consensus_cache[key]
        error = None
        if not reason:
            reason, error = observation_check(a['xyz'], a['generated_uv'], v, v['width'], v['height'], depths[v['name']], cfg)
            if reason: reason = 'generated_' + reason
        if reason:
            rejected[reason] += 1
            reasons.append({'track': key, 'view': v['name'], 'reason': reason})
            continue
        xy = semantic_coordinates(a['generated_uv'], v['width'], v['height'], cfg['semantic_width'], cfg['semantic_height'])
        accepted.append({**a, 'class_id': label, 'semantic_xy': xy.tolist(), 'real_errors_px_2k': real_errors, 'generated_error_px_2k': error, 'real_support': len(a['real_observations'])})
    # A rendered pixel cannot receive contradictory targets. Retain one best
    # measured correspondence per cell, preserving its continuous sample position.
    cells = defaultdict(list)
    for a in accepted:
        x, y = np.floor(np.asarray(a['semantic_xy']) + .5).astype(int)
        cells[(a['generated_view'], int(x) % cfg['semantic_width'], int(np.clip(y, 0, cfg['semantic_height']-1)))].append(a)
    final = []
    for rows in cells.values():
        if len({a['class_id'] for a in rows}) != 1:
            rejected['semantic_cell_conflict'] += len(rows); continue
        rows.sort(key=lambda a: (a['generated_error_px_2k'], -a['real_support'], a['track']))
        final.append(rows[0]); rejected['same_cell_duplicate'] += len(rows)-1
    by_view = defaultdict(list)
    for a in final: by_view[a['generated_view']].append(a)
    views = []
    for name, rows in sorted(by_view.items()):
        base = dest / (name + SUFFIX)
        xy = np.array([a['semantic_xy'] for a in rows], dtype='f4')
        labs = np.array([a['class_id'] for a in rows], dtype='i8')
        xy.tofile(str(base) + '_xy.f32'); labs.tofile(str(base) + '_labels.i64')
        v = generated[name]
        views.append({k: v[k] for k in ['name', 'rotation', 'center', 'source_training_view', 'direction']} | {'anchor_xy': str(base) + '_xy.f32', 'anchor_labels': str(base) + '_labels.i64', 'anchor_count': len(rows)})
    assert len(final) and len(views)
    text = np.fromfile(teacher['text'], 'f4').reshape(7, 512)
    F, S, _ = targets(np.arange(7, dtype='u1')[None], text, teacher['smoothing'])
    S.reshape(7, 7).tofile(dest / ('class_scores' + SUFFIX + '.f16'))
    inputs += [ref(p) for p in [p0/'points.f32', p0/'structure.json', p1/'enhanced_points.f32', p1/'selected_views.json', p1/'observations.json', source/'00_dataset/dataset_manifest.json', source/'02_semantic_teacher/cache/manifest.json', Path(teacher['text']), Path(teacher['decoder'])]]
    summary = {'anchor_observations': len(final), 'unique_tracks': len({a['track'] for a in final}), 'generated_views': len(views), 'registered_selected_views': len(selected), 'class_counts': {n: sum(a['class_id'] == k for a in final) for k, n in enumerate(teacher['classes'])}, 'source_counts': dict(Counter(a['source'] for a in final)), 'real_support_counts': dict(Counter(a['real_support'] for a in final)), 'proposals': len(proposed), 'rejections': dict(rejected), 'generated_reprojection_px_2k': quantiles([a['generated_error_px_2k'] for a in final]), 'real_reprojection_px_2k': quantiles([e for a in final for e in a['real_errors_px_2k']]), 'smoothing': teacher['smoothing'], 'geometry_unchanged': True, 'dense_generated_GT': False, 'occlusion_limit': 'Sparse P1 depth cells reject obvious foreground conflicts; this does not establish dense visibility.', 'sampling': 'continuous bilinear, horizontal circular padding; native coordinate = SfM uv * semantic_size / native_size - 0.5'}
    manifest = {'views': views, 'summary': summary, 'config': cfg, 'source_run': str(source), 'inputs': inputs, 'class_scores': str(dest / ('class_scores' + SUFFIX + '.f16')), 'smoothing': teacher['smoothing']}
    write(dest / ('anchors' + SUFFIX + '.json'), final)
    write(dest / ('rejected' + SUFFIX + '.json'), reasons)
    write(dest / ('manifest' + SUFFIX + '.json'), manifest)
    preview(dest, final, generated, protocol['classes']['classes'])
    print(summary, flush=True)
    return manifest


def preview(dest, anchors, views, classes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    counts = Counter(a['generated_view'] for a in anchors)
    chosen = [x[0] for x in counts.most_common(4)]
    fig, axes = plt.subplots(len(chosen), 1, figsize=(16, 5*len(chosen)))
    colors = np.asarray([c['color'] for c in classes]) / 255
    for ax, name in zip(np.atleast_1d(axes), chosen):
        v = views[name]; rows = [a for a in anchors if a['generated_view'] == name]
        ax.imshow(Image.open(v['image'])); uv = np.array([a['generated_uv'] for a in rows])
        ax.scatter(uv[:, 0], uv[:, 1], c=colors[[a['class_id'] for a in rows]], s=28, edgecolors='black', linewidths=.4)
        ax.set_title(f'{name}: {len(rows)} sparse anchors'); ax.axis('off')
    fig.legend(handles=[Patch(color=colors[k], label=c['name']) for k,c in enumerate(classes)], loc='lower center', ncol=7)
    fig.tight_layout(rect=(0,.03,1,1)); fig.savefig(dest/('anchor_overlays'+SUFFIX+'.png'), dpi=125); plt.close(fig)
