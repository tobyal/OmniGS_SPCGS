"""Spherical geometry in the actual OpenMVG ERP convention (integer pixels).

Real camera poses remain fixed. New poses use measured 2D--3D correspondences,
with independent held-out correspondences for acceptance. No monocular depth.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from .common import read, write, ref

FACES = [np.eye(3), np.array([[0,0,1],[0,1,0],[-1,0,0]]), np.diag([-1,1,-1]),
         np.array([[0,0,-1],[0,1,0],[1,0,0]]), np.array([[1,0,0],[0,0,-1],[0,1,0]]),
         np.array([[1,0,0],[0,0,1],[0,-1,0]])]

def bearings(uv, width, height):
    uv = np.asarray(uv, dtype=float)
    lon = (uv[:, 0] / width - .5) * 2 * np.pi
    lat = (uv[:, 1] / height - .5) * np.pi
    return np.column_stack([np.cos(lat)*np.sin(lon), np.sin(lat), np.cos(lat)*np.cos(lon)])

def camera_bearings(xyz, camera):
    v = (np.asarray(xyz) - camera['center']) @ np.asarray(camera['rotation']).T
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)

def project_erp(xyz, camera, width, height):
    v = camera_bearings(xyz, camera)
    return np.column_stack([(np.arctan2(v[:,0], v[:,2])/(2*np.pi)+.5)*width,
                            (np.arcsin(np.clip(v[:,1],-1,1))/np.pi+.5)*height])

def angular_pixels(a, b, reference_width=2048):
    # atan2 is stable at small angles and at the longitude seam.
    return np.arctan2(np.linalg.norm(np.cross(a,b),axis=-1), np.sum(a*b,axis=-1)) * reference_width/(2*np.pi)

def quantiles(values):
    return dict(zip(['min','p10','median','p90','max'], np.quantile(values,[0,.1,.5,.9,1]).tolist())) if len(values) else {}

def load_features(folder, name):
    uv = np.loadtxt(Path(folder)/(name+'.feat'), dtype=np.float32, ndmin=2)
    raw = (Path(folder)/(name+'.desc')).read_bytes()
    n = int(np.frombuffer(raw[:8],dtype='<u8')[0])
    if n != len(uv) or len(raw) != 8+n*128:
        raise ValueError(f'Unsupported OpenMVG SIFT descriptor encoding: {name}')
    desc = np.frombuffer(raw[8:],dtype=np.uint8).reshape(n,128).astype(np.float32)
    fingerprint=hashlib.sha256(raw+uv.tobytes()).hexdigest()
    return {'uv': uv[:,:2], 'desc': desc, 'fingerprint':fingerprint}

def match_features(real, generated, ratio=.8, mutual=True):
    if len(real['desc']) < 2 or len(generated['desc']) < 2:
        return np.empty((0,2),dtype=int), np.empty(0)
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1,trees=5),dict(checks=96))
    pairs = matcher.knnMatch(real['desc'], generated['desc'], k=2)
    matches = [(a.queryIdx,a.trainIdx,a.distance/max(b.distance,1e-6))
               for a,b in pairs if a.distance < ratio*b.distance]
    if mutual and matches:
        reverse = matcher.knnMatch(generated['desc'], real['desc'], k=1)
        back = {row[0].queryIdx: row[0].trainIdx for row in reverse if row}
        matches = [row for row in matches if back.get(row[1]) == row[0]]
    matches.sort(key=lambda row:row[2])
    return (np.array([m[:2] for m in matches],dtype=int).reshape(-1,2),
            np.array([m[2] for m in matches]))

def load_anchors(structure_path, feature_folder, train, original_xyz, output):
    scene = read(structure_path)
    names = {v['key']:Path(v['value']['ptr_wrapper']['data']['filename']).stem for v in scene['views']}
    cameras = {v['name']:v for v in train}
    rows = [v for v in scene['structure'] if len(v['value']['observations']) >= 3
            and all(names[o['key']] in cameras for o in v['value']['observations'])]
    if not rows:
        raise ValueError('No training-only observed landmarks in source SfM')
    raw = np.array([v['value']['X'] for v in rows])
    transforms = [np.eye(3),np.diag([1,-1,-1])]
    tree = cKDTree(original_xyz)
    distances = [tree.query(raw@t.T)[0] for t in transforms]
    index = int(np.argmin([np.median(x) for x in distances]))
    xyz = raw @ transforms[index].T
    matched = distances[index] < 2e-5
    if matched.sum() < min(100,len(original_xyz)):
        raise ValueError('Source observations cannot be aligned to the current native point cloud')
    features = {n:load_features(feature_folder,n) for n in cameras}
    anchors, by_feature, errors = [], {}, []
    for i, row in enumerate(rows):
        if not matched[i]:
            continue
        anchor_id = len(anchors)
        obs = []
        for o in row['value']['observations']:
            name = names[o['key']]
            uv = np.array(o['value']['x'])
            fid = o['value']['id_feat']
            if fid >= len(features[name]['uv']) or np.linalg.norm(features[name]['uv'][fid]-uv) > .02:
                raise ValueError(f'Feature IDs do not match the SfM observation file: {name}')
            cam = cameras[name]
            err = angular_pixels(camera_bearings(xyz[i:i+1],cam),bearings(uv[None],2048,1024))[0]
            errors.append(err)
            by_feature[(name,fid)] = anchor_id
            obs.append({'view':name,'feature_id':fid,'uv':uv.tolist()})
        anchors.append({'source_landmark_id':row['key'],'xyz':xyz[i].tolist(),'observations':obs})
    if np.median(errors) > 4:
        raise ValueError('Native camera/point/ERP conventions fail observed-landmark reprojection check')
    write(output, {'source':ref(structure_path),'feature_folder':str(feature_folder),
                   'world_conversion':transforms[index].tolist(),'anchors':len(anchors),
                   'original_points':len(original_xyz),'reprojection_spherical_px_2k':quantiles(errors),
                   'observation_policy':'all observations train-only; landmarks must coincide with original native cloud',
                   'projection':'OpenMVG spherical integer-pixel coordinates; +X right, +Y down, +Z panorama center'})
    return {'xyz':np.array([a['xyz'] for a in anchors]),'landmarks':anchors,
            'by_feature':by_feature,'features':features,'cameras':cameras}

def pnp_candidates(xyz, observed, source, threshold, seed):
    result = [source]
    cv2.setRNGSeed(int(seed))
    for face in FACES:
        local = observed @ face
        mask = local[:,2] > .5
        if mask.sum() < 8:
            continue
        xy = local[mask,:2]/local[mask,2:]
        ok, rv, tv, ids = cv2.solvePnPRansac(np.ascontiguousarray(xyz[mask]),
             np.ascontiguousarray(xy),np.eye(3),None,iterationsCount=1500,
             reprojectionError=threshold*2*np.pi/2048*1.6,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
        if ok:
            rot = face @ cv2.Rodrigues(rv)[0]
            center = -cv2.Rodrigues(rv)[0].T @ tv.ravel()
            result.append({'rotation':rot.tolist(),'center':center.tolist()})
    return result

def refine_pose(xyz, observed, camera, threshold):
    x0 = np.r_[Rotation.from_matrix(camera['rotation']).as_rotvec(),camera['center']]
    def residual(x):
        rotated = (xyz-x[3:])@Rotation.from_rotvec(x[:3]).as_matrix().T
        predicted = rotated/np.maximum(np.linalg.norm(rotated,axis=1,keepdims=True),1e-10)
        return (predicted-observed).ravel()
    fit = least_squares(residual,x0,loss='soft_l1',f_scale=threshold*2*np.pi/2048,max_nfev=160)
    return {'rotation':Rotation.from_rotvec(fit.x[:3]).as_matrix().tolist(),'center':fit.x[3:].tolist()}

def register(generated, source, anchor_map, matches, cfg, output):
    # Conflicting matches never get duplicated to artificially increase support.
    assignments = []
    for name, (pairs, scores) in matches.items():
        for (ri,gi),score in zip(pairs,scores):
            aid = anchor_map['by_feature'].get((name,int(ri)))
            if aid is not None:
                assignments.append((float(score),int(gi),aid,name,int(ri)))
    assignments.sort()
    used_g, used_a, rows = set(),set(),[]
    for score,gi,aid,name,ri in assignments:
        if gi in used_g or aid in used_a:
            continue
        used_g.add(gi);used_a.add(aid)
        rows.append({'generated_feature_id':gi,'anchor_id':aid,'source_view':name,'source_feature_id':ri,'ratio':score})
    result = {'name':generated['name'],'source_training_view':source['name'],
              'correspondences':rows,'registered':False,'matches_2d3d':len(rows)}
    if len(rows) < cfg['min_correspondences']:
        result['reason'] = 'insufficient 2D-3D matches'
        write(output,result);return result
    xyz = anchor_map['xyz'][[r['anchor_id'] for r in rows]]
    uv = generated['features']['uv'][[r['generated_feature_id'] for r in rows]]
    observed = bearings(uv,generated['width'],generated['height'])
    permutation = np.random.default_rng(cfg['seed']).permutation(len(rows))
    validation = permutation[:max(6,len(rows)//4)]
    training = permutation[len(validation):]
    threshold = cfg['registration_error_px']
    candidates = pnp_candidates(xyz[training],observed[training],source,threshold,cfg['seed'])
    ranked = []
    for camera in candidates:
        e = angular_pixels(camera_bearings(xyz[training],camera),observed[training])
        ranked.append(((e<threshold).sum(),-float(np.median(e)),camera))
    camera = max(ranked,key=lambda r:r[:2])[2]
    camera = refine_pose(xyz[training],observed[training],camera,threshold)
    error = angular_pixels(camera_bearings(xyz,camera),observed)
    fit_ids = training[error[training] < threshold*2]
    if len(fit_ids) >= 8:
        camera = refine_pose(xyz[fit_ids],observed[fit_ids],camera,threshold*.5)
    error = angular_pixels(camera_bearings(xyz,camera),observed)
    inliers = error < threshold
    valid = error[validation] < threshold
    displacement = np.array(camera['center'])-source['center']
    distance = float(np.linalg.norm(displacement))
    rot_delta = Rotation.from_matrix(np.array(camera['rotation'])@np.array(source['rotation']).T).magnitude()*180/np.pi
    bins = np.floor(uv[inliers]/[generated['width']/8,generated['height']/4]).astype(int)
    coverage = len(set(map(tuple,bins)))
    prior = np.array(generated.get('conditioning_c2w',np.eye(4)))[:3,3]
    prior = np.asarray(source['rotation']).T@np.diag([1,-1,1])@prior
    direction_cos = float(np.dot(displacement,prior)/max(np.linalg.norm(prior)*distance,1e-12))
    source_err = angular_pixels(camera_bearings(xyz,source),observed)
    improvement = float(np.median(source_err[validation][valid]-error[validation][valid])) if valid.any() else 0.
    reasons=[]
    if inliers.sum() < cfg['min_inliers'] or inliers.mean() < cfg['min_inlier_ratio']: reasons.append('insufficient registration inliers')
    if valid.sum() < cfg['min_validation_inliers'] or valid.mean() < cfg['min_validation_ratio']: reasons.append('held-out geometric validation failed')
    if coverage < cfg['min_coverage_bins']: reasons.append('inliers concentrated in too few ERP regions')
    if not cfg['min_translation'] <= distance <= cfg['max_translation']: reasons.append('translation outside configured world-coordinate range')
    if rot_delta > cfg['max_rotation_deg']: reasons.append('large orientation departure')
    if improvement < cfg['min_validation_improvement_px']: reasons.append('estimated pose does not improve held-out projection over source pose')
    for row,e,inside in zip(rows,error,inliers):
        row.update(error_spherical_px_2k=float(e),inlier=bool(inside))
    result.update(camera,registered=not reasons,reason='; '.join(reasons) if reasons else None,
                  inliers=int(inliers.sum()),inlier_ratio=float(inliers.mean()),
                  fit_indices=training.tolist(),validation_indices=validation.tolist(),
                  validation_inliers=int(valid.sum()),validation_total=len(validation),
                  validation_error_px=quantiles(error[validation][valid]),error_px=quantiles(error[inliers]),
                  coverage_bins=coverage,translation=distance,rotation_delta_deg=float(rot_delta),
                  motion_direction_cosine=direction_cos,heldout_improvement_px=improvement,
                  conditioning_role='generation only; direction cosine is diagnostic, not an acceptance criterion; absolute pose fitted from training-only 2D-3D correspondences')
    write(output,result)
    return result

def triangulate_track(cameras, pixels, sizes, cfg):
    rays = np.array([bearings(np.array([uv]),w,h)[0]@np.array(cam['rotation'])
                     for cam,uv,(w,h) in zip(cameras,pixels,sizes)])
    centers = np.array([cam['center'] for cam in cameras])
    projectors = np.eye(3)[None]-rays[:,:,None]*rays[:,None,:]
    matrix = projectors.sum(0)
    if np.linalg.cond(matrix) > 1e8:
        return None, {'reason':'degenerate rays'}
    point = np.linalg.solve(matrix,np.einsum('nij,nj->i',projectors,centers))
    def residual(p):
        delta=p-centers
        return (delta/np.maximum(np.linalg.norm(delta,axis=1,keepdims=True),1e-10)-rays).ravel()
    point = least_squares(residual,point,loss='soft_l1',f_scale=cfg['triangulation_error_px']*2*np.pi/2048,max_nfev=30).x
    delta = point-centers
    errors = angular_pixels(delta/np.maximum(np.linalg.norm(delta,axis=1,keepdims=True),1e-10),rays)
    parallax = np.rad2deg(np.arccos(np.clip(rays[:-1]@rays[-1],-1,1)))
    info={'error_spherical_px_2k':errors.tolist(),'max_error_px':float(errors.max()),
          'parallax_deg':float(parallax.max()),'real_views':len(cameras)-1,'reason':None}
    if not np.isfinite(point).all() or np.any(np.sum(delta*rays,axis=1) <= 0):info['reason']='invalid point or negative ray depth'
    elif errors.max() > cfg['triangulation_error_px']:info['reason']='reprojection'
    elif parallax.max() < cfg['min_parallax_deg']:info['reason']='low real-generated parallax'
    return point,info

def epipolar_mask(real, generated, real_uv, generated_uv, cfg):
    a = bearings(real_uv,2048,1024)@np.array(real['rotation'])
    b = bearings(generated_uv,generated['width'],generated['height'])@np.array(generated['rotation'])
    baseline = np.asarray(generated['center'])-real['center']
    plane = np.cross(a,baseline)
    plane /= np.maximum(np.linalg.norm(plane,axis=1,keepdims=True),1e-12)
    error = np.arcsin(np.clip(np.abs(np.sum(plane*b,axis=1)),0,1))*2048/(2*np.pi)
    return error < cfg['epipolar_error_px'],error
