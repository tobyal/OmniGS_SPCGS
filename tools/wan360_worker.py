"""Isolated genuine Wan360 inference; one model load for all requested videos.

Uses the pinned official inference.py trajectory and WanVideoPipeline. The saved
trajectory is a conditioning signal only, never a registered world pose.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('request')
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text())
    cfg = request['config']
    os.environ.setdefault('DIFFSYNTH_SKIP_DOWNLOAD', 'true')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    model_root = Path(cfg['repository']) / 'model'
    sys.path.insert(0, str(model_root))
    spec = importlib.util.spec_from_file_location('wan360_official_inference', model_root / 'inference.py')
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    import numpy as np
    import torch
    from PIL import Image
    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
    from diffsynth.utils.data import save_video
    weights = Path(cfg['weights'])
    models = [weights / 'Wan360.safetensors', weights / 'Wan2.2_VAE.pth',
              weights / 'models_t5_umt5-xxl-enc-bf16.pth']
    for path in models:
        if not path.is_file():
            raise FileNotFoundError(path)
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device='cuda',
        model_configs=[ModelConfig(path=str(path)) for path in models],
        tokenizer_config=ModelConfig(path=str(weights / 'google/umt5-xxl')),
        redirect_common_files=False,
    )
    pipe.dit.is_360 = True
    for job in request['jobs']:
        out = Path(job['output'])
        out.mkdir(parents=True, exist_ok=True)
        first = Image.open(job['image']).convert('RGB')
        if first.width != first.height * 2:
            raise ValueError('Wan360 input must be ERP, width = 2 * height')
        first = first.resize((cfg['width'], cfg['height']), Image.Resampling.LANCZOS)
        poses = official.camera_trajectory(job['direction'], cfg['num_frames'], cfg['distance'])
        np.savez(out / 'conditioning_trajectory.npz', data=poses)
        prompt = cfg['prompt'].replace('{direction}', job['direction'])
        started = time.time()
        torch.cuda.reset_peak_memory_stats()
        print('GENERATE', job['name'], flush=True)
        with torch.inference_mode():
            video = pipe(prompt=prompt, negative_prompt=cfg['negative_prompt'],
                         input_image=first.copy(), height=cfg['height'], width=cfg['width'],
                         num_frames=cfg['num_frames'], num_inference_steps=cfg['steps'],
                         cfg_scale=cfg['cfg_scale'], sigma_shift=cfg['sigma_shift'], seed=job['seed'],
                         tiled=True, is_360=True, cam_c2w=poses)
        if len(video) != len(poses):
            raise RuntimeError(f'Video/trajectory length mismatch: {len(video)} vs {len(poses)}')
        frames = out / 'frames'
        frames.mkdir(exist_ok=True)
        # Lossless snapshots come from the exact frames passed to the video encoder.
        # Frame index zero is the conditioned starting frame, never a new view.
        mapping = []
        # Retain every lossless frame so later candidate-range changes reuse the
        # same expensive video, without extracting a lossy re-encoded frame.
        for i in range(len(video)):
            frame = video[i]
            if frame.size != (cfg['width'], cfg['height']):
                raise RuntimeError('Unexpected generated ERP dimensions')
            frame.save(frames / f'{i:04d}.png')
            mapping.append({'frame_index': i, 'trajectory_index': i, 'time_seconds': i / cfg['fps'],
                            'image': str(frames / f'{i:04d}.png'), 'conditioning_c2w': poses[i].tolist()})
        save_video(video, str(out / 'video.mp4'), fps=cfg['fps'], quality=cfg['video_quality'])
        result = {**job, 'status': 'completed', 'config': cfg, 'prompt': prompt,
                  'seconds': time.time() - started, 'video': str(out / 'video.mp4'),
                  'trajectory': str(out / 'conditioning_trajectory.npz'), 'frames': mapping,
                  'trajectory_role': 'uncalibrated camera-motion conditioning; NOT a world pose',
                  'actual_frames': len(video), 'torch': torch.__version__, 'cuda': torch.version.cuda,
                  'python':sys.version,'gpu':torch.cuda.get_device_name(0),
                  'peak_allocated_vram_bytes':torch.cuda.max_memory_allocated()}
        temporary = out / 'generation.json.tmp'
        temporary.write_text(json.dumps(result, indent=2))
        temporary.replace(out / 'generation.json')
        print('COMPLETED', job['name'], result['seconds'], flush=True)
        del video
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
