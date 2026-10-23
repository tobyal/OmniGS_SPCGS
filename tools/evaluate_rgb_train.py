"""Render and evaluate every real training camera using a common RGB protocol."""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spcgs.common import ROOT, read, write, ref


def main():
    parser = argparse.ArgumentParser()
    for name in ['dataset', 'config', 'checkpoint', 'output', 'exporter']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['PF_MODE'] = '0'
    os.environ.pop('PF_FIELD', None)
    import numpy as np
    import torch
    from PIL import Image
    odgs = ROOT.parent / 'ODGS'
    sys.path.insert(0, str(odgs))
    from utils.loss_utils import ssim
    from utils.image_utils import psnr
    from lpipsPyTorch.modules.lpips import LPIPS
    torch.set_num_threads(2)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    camera_file = args.dataset / 'openMVG/data_openmvg_train.json'
    scene = read(camera_file)
    views = {Path(v['value']['ptr_wrapper']['data']['filename']).stem:
             v['value']['ptr_wrapper']['data'] for v in scene['views']}
    assert len(views) == 56
    request = {'split': 'train', 'camera_json': str(camera_file), 'width': 2048, 'height': 1024,
               'views': [], 'diagnostics': False, 'moments': False, 'benchmark': False, 'training_step': False}
    write(out / 'request.json', request)
    write(out / 'status.json', {'status': 'running', 'phase': 'render', 'started': started})
    command = [args.exporter, args.config, args.dataset, args.checkpoint, out/'request.json', out]
    with (out/'render.log').open('w') as log:
        subprocess.run(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT, check=True)
    manifest = read(out/'render_manifest.json')
    assert manifest['width'] == 2048 and manifest['height'] == 1024
    assert {r['name'] for r in manifest['frames']} == set(views)
    assert len(manifest['frames']) == 56
    network = LPIPS('alex', '0.1').cuda().eval()
    rows = []
    with torch.inference_mode():
        for name in sorted(views):
            source = Path(scene['root_path']) / views[name]['filename']
            with Image.open(source) as image:
                assert image.size == (2048, 1024)
                gt = np.asarray(image.convert('RGB'), dtype=np.float32) / 255.
            rgb = np.fromfile(out/'rgb'/(name+'.f32'), dtype='<f4').reshape(3,1024,2048)
            assert np.isfinite(rgb).all() and rgb.min() >= 0 and rgb.max() <= 1
            a = torch.from_numpy(rgb.copy()).cuda()
            b = torch.from_numpy(gt.transpose(2,0,1).copy()).cuda()
            mse = (a-b).square().mean()
            row = {'name': name, 'PSNR': float(-10*torch.log10(mse.clamp_min(1e-12))),
                   'SSIM': float(ssim(a,b)),
                   'LPIPS': float(network(a[None]*2-1,b[None]*2-1).mean()),
                   'legacy_channel_mean_PSNR': float(psnr(a,b).mean()),
                   'legacy_input01_LPIPS': float(network(a[None],b[None]).mean()),
                   'reference': str(source), 'render': str(out/'rgb'/(name+'.png'))}
            assert all(np.isfinite(row[k]) for k in ['PSNR','SSIM','LPIPS'])
            rows.append(row)
            write(out/'status.json', {'status':'running','phase':'metrics','processed':len(rows),'total':56})
            print('TRAIN METRIC', name, row['PSNR'], row['SSIM'], row['LPIPS'], flush=True)
    keys = ['PSNR','SSIM','LPIPS','legacy_channel_mean_PSNR','legacy_input01_LPIPS']
    definitions = {'aggregation':'Unweighted arithmetic mean of per-view metrics over all 56 real training cameras.',
        'PSNR':'-10 log10 mean squared error over all RGB pixels/channels; images in [0,1].',
        'SSIM':'Existing ODGS Gaussian-window implementation: window=11, sigma=1.5, zero padding.',
        'LPIPS':'AlexNet v0.1, full 2048x1024 image; [0,1] RGB converted to [-1,1] before LPIPS.',
        'legacy_channel_mean_PSNR':'Prior evaluator: compute PSNR separately per RGB channel, then average.',
        'legacy_input01_LPIPS':'Prior evaluator compatibility only: passes [0,1] RGB directly to LPIPS.',
        'render':'Native baseline renderer, PF_MODE=0, float RGB clamped to [0,1]; no JPEG metric input.',
        'scope':'Training reconstruction metrics, not held-out generalization.'}
    result = {'status':'completed','split':'train','views':56,'resolution':[2048,1024],
              'seconds':time.time()-started,'checkpoint':ref(args.checkpoint),'exporter':ref(args.exporter),
              'camera_json':ref(camera_file),'torch':torch.__version__,
              'metric_sources':[ref(odgs/p) for p in ['utils/loss_utils.py','utils/image_utils.py',
                              'lpipsPyTorch/modules/lpips.py','lpipsPyTorch/modules/networks.py']],
              'aggregate':{k:float(np.mean([r[k] for r in rows])) for k in keys},
              'per_view':rows,'definitions':definitions}
    write(out/'metrics.json',result)
    with (out/'per_view.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(out/'status.json',{'status':'completed','processed':56,'total':56,'seconds':time.time()-started})


if __name__ == '__main__':
    main()
