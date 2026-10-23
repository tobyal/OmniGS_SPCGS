"""Real-data localization check using one withheld training camera's RGB.

No generated video/point counts are claimed here. Matching does not use the
query image's original feature observations; its known pose is evaluation only.
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.spatial.transform import Rotation
from spcgs.common import ROOT,DATA,cameras,ply,write
from spcgs.m1_wan360 import geometry_settings,extract_candidates
from spcgs.m1_geometry import load_anchors,match_features,register

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--extract-downsampled',action='store_true');args=parser.parse_args()
    folder=ROOT/('runs/m1_diagnostics/relocalization_1024' if args.extract_downsampled else 'runs/m1_diagnostics/relocalization');folder.mkdir(parents=True,exist_ok=True)
    cfg=geometry_settings({'seed':42});train=cameras('train');source=train[0];query=train[1]
    cfg.update(min_direction_cosine=-1.,max_rotation_deg=45.,max_translation=4.)
    anchors=load_anchors(cfg['structure'],cfg['features'],train,ply(DATA/'openMVG/scene.ply')[0],folder/'anchors.json')
    nearby=sorted([v for v in train if v['name']!=query['name']],key=lambda v:np.linalg.norm(np.array(v['center'])-query['center']))[:6]
    generated={**query,'width':2048,'height':1024,'features':anchors['features'][query['name']]}
    if args.extract_downsampled:
        from PIL import Image
        image=folder/'real_query_not_generated.png'
        Image.open(query['image']).convert('RGB').resize((1024,512),Image.Resampling.LANCZOS).save(image)
        video={'name':'real_query_diagnostic','source_training_view':source['name'],'direction':'real_pose_diagnostic',
               'frames':[{'frame_index':1,'trajectory_index':1,'image':str(image)}]}
        generated=extract_candidates(folder,{'seed':42},[video])[0]
    matches={v['name']:match_features(anchors['features'][v['name']],generated['features'],cfg['ratio']) for v in nearby}
    result=register(generated,source,anchors,matches,cfg,folder/'registration.json')
    # A perfect descriptor match to the input image must not be accepted as a
    # translated observation just because the conditioning requested motion.
    copy_matches={source['name']:(np.column_stack([np.arange(len(anchors['features'][source['name']]['uv']))]*2),
                                  np.zeros(len(anchors['features'][source['name']]['uv'])))}
    copy_view={**source,'width':2048,'height':1024,'features':anchors['features'][source['name']]}
    copy_result=register(copy_view,source,anchors,copy_matches,cfg,folder/'original_copy_rejection.json')
    error=float(np.linalg.norm(np.array(result['center'])-query['center'])) if 'center' in result else None
    rot=float(Rotation.from_matrix(np.array(result['rotation'])@np.array(query['rotation']).T).magnitude()*180/np.pi) if 'rotation' in result else None
    summary={'scope':'real held-out training-camera localization, not Wan360 validation',
             'query':query['name'],'matching_views':[v['name'] for v in nearby],
             'registered':result['registered'],'center_error_world_units':error,'rotation_error_deg':rot,
             'matches_2d3d':result['matches_2d3d'],'inliers':result.get('inliers'),
             'validation_inliers':result.get('validation_inliers'),
             'original_copy_rejected':not copy_result['registered'],'copy_translation':copy_result.get('translation')}
    summary['passed']=bool(result['registered'] and error<.1 and rot<1. and not copy_result['registered'])
    write(folder/'results.json',summary);print(summary,flush=True)
    if not summary['passed']:raise RuntimeError('Real camera localization regression failed')

if __name__=='__main__':main()
