"""Evaluate the final original OmniGS checkpoint on the approved 18/6 split."""
import argparse
import csv
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_omnigs_sparse18 import ROOT, read, write, ref


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--method', default='OmniGS')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import numpy as np
    import torch
    from PIL import Image
    odgs = ROOT.parent / 'ODGS'
    sys.path.insert(0, str(odgs))
    from utils.loss_utils import ssim
    from lpipsPyTorch.modules.lpips import LPIPS
    torch.set_num_threads(2)
    run = args.run.resolve()
    started = time.time()
    selected = read(run / 'approved_split.json')['selected_cameras']
    identities = {x['image_id']: x for x in selected}
    network = LPIPS('alex', '0.1').cuda().eval()
    for split, count in [('test', 6), ('train', 18)]:
        out = run / 'evaluation' / split
        camera_file = run / f'dataset/openMVG/data_openmvg_{split}.json'
        scene = read(camera_file)
        views = {Path(v['value']['ptr_wrapper']['data']['filename']).stem: v['value']['ptr_wrapper']['data'] for v in scene['views']}
        manifest = read(out / 'render_manifest.json')
        assert len(views) == len(manifest['frames']) == count
        assert (manifest['width'], manifest['height']) == (2048, 1024)
        assert {f['name'] for f in manifest['frames']} == set(views)
        assert set(views) == {x['image_id'] for x in selected if x['candidate_role'] == split}
        rows = []
        with torch.inference_mode():
            for name in sorted(views):
                source = Path(scene['root_path']) / views[name]['filename']
                with Image.open(source) as image:
                    assert image.size == (2048, 1024)
                    gt = np.asarray(image.convert('RGB'), dtype=np.float32) / 255.
                rgb = np.fromfile(out / 'rgb' / (name + '.f32'), dtype='<f4').reshape(3, 1024, 2048)
                assert np.isfinite(rgb).all() and rgb.min() >= 0 and rgb.max() <= 1
                a = torch.from_numpy(rgb.copy()).cuda()
                b = torch.from_numpy(gt.transpose(2, 0, 1).copy()).cuda()
                mse = (a - b).square().mean()
                row = {'camera_id': identities[name]['camera_id'], 'name': name,
                       'height_band': identities[name]['height_band'],
                       'PSNR': float(-10 * torch.log10(mse.clamp_min(1e-12))),
                       'SSIM': float(ssim(a, b)), 'LPIPS': float(network(a[None] * 2 - 1, b[None] * 2 - 1).mean()),
                       'reference': str(source), 'render_float': str(out / 'rgb' / (name + '.f32'))}
                assert all(np.isfinite(row[k]) for k in ['PSNR', 'SSIM', 'LPIPS'])
                rows.append(row)
                write(run / 'evaluation/status.json', {'status': 'running', 'phase': split,
                      'processed': len(rows), 'total': count, 'updated': time.time()})
                print(split, row, flush=True)
        definitions = {'aggregation': f'Unweighted arithmetic mean over all {count} {split} views.',
              'PSNR': '-10 log10 MSE over all RGB pixels/channels in [0,1].',
              'SSIM': 'Existing ODGS implementation; Gaussian window=11, sigma=1.5, zero padding.',
              'LPIPS': 'AlexNet v0.1; full 2048x1024 images; RGB converted to [-1,1].',
              'render': f'{args.method} native float RGB clamped to [0,1], black background; no JPEG metric input.',
              'pixels': 'Full ERP including poles and tripod; no metric mask or crop.',
              'scope': 'Held-out final checkpoint evaluation' if split == 'test' else 'Training reconstruction diagnostic'}
        result = {'status': 'completed', 'split': split, 'views': count, 'resolution': [2048, 1024],
                  'checkpoint': read(run / 'training_result.json')['checkpoint'], 'camera_json': ref(camera_file),
                  'aggregate': {k: float(np.mean([r[k] for r in rows])) for k in ['PSNR', 'SSIM', 'LPIPS']},
                  'per_view': rows, 'definitions': definitions, 'torch': torch.__version__,
                  'evaluator': ref(__file__), 'metric_sources': [ref(odgs / p) for p in
                    ['utils/loss_utils.py', 'lpipsPyTorch/modules/lpips.py', 'lpipsPyTorch/modules/networks.py']]}
        write(out / 'metrics.json', result)
        with (out / 'per_view.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write(run / 'evaluation/status.json', {'status': 'completed', 'seconds': time.time() - started})


if __name__ == '__main__':
    main()
