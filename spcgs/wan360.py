"""Versioned Wan360 inference requests and reusable video/trajectory caches."""
import os
import subprocess
from pathlib import Path
import numpy as np
from .common import ROOT, read, write, ref, signature, run_command

CODE_REVISION = '628784299fd5c01394be9beeae052e63bc533ee5'

def settings(cfg):
    defaults = {
        'python': str(ROOT / '.venv_wan360/bin/python'),
        'repository': str(ROOT / 'vendor/MUGEN'), 'weights': str(ROOT / 'weights/Wan360'),
        'cache': str(ROOT / 'runs/m1_cache/wan360'), 'views': 1, 'view_names': [],
        'directions': ['forward', 'backward', 'left', 'right'], 'distance': 0.8,
        'width': 1024, 'height': 512, 'num_frames': 49, 'steps': 30, 'fps': 16,
        'cfg_scale': 5.0, 'sigma_shift': 5.0, 'video_quality': 8,
        'candidate_frames': [4, 8, 12, 16, 20], 'reuse': True,
        'prompt': 'A photorealistic 360 degree panoramic video of the same static indoor industrial mechanical plant. '
                  'The camera translates slowly {direction}. Preserve the existing metal pipes, insulated ducts, '
                  'pumps, electrical cabinets, steel supports, red columns and concrete floor exactly. '
                  'Fixed lighting. Stationary machinery. No people and no moving objects.',
        'negative_prompt': 'blur, low quality, watermark, subtitles, people, moving objects, '
                           'changing equipment, deforming pipes, morphing structures, outdoor garden, trees',
    }
    defaults.update(cfg.get('wan360', {}))
    for key in ['python','repository','weights','cache']:
        path=Path(defaults[key]).expanduser()
        defaults[key]=str(path if path.is_absolute() else ROOT/path)
    return defaults

def plan_generation(folder, cfg, train):
    """Prepare identical signed jobs and cache links without launching inference."""
    w = settings(cfg)
    if w['width'] != 2 * w['height'] or w['height'] % 32:
        raise ValueError('Wan360 needs 2:1 ERP, height divisible by 32')
    if w['num_frames'] < 5 or (w['num_frames'] - 1) % 4:
        raise ValueError('Wan360 num_frames must be 4n+1')
    if not w['candidate_frames'] or any(i <= 0 or i >= w['num_frames'] for i in w['candidate_frames']):
        raise ValueError('Candidate indices must be inside the video and exclude frame zero')
    if not np.isfinite(w['distance']) or w['distance'] <= 0:
        raise ValueError('Positive finite conditioning distance required')
    if not w['directions'] or not set(w['directions']) <= {'forward', 'backward', 'left', 'right'}:
        raise ValueError('Unsupported motion direction')
    by_name = {v['name']: v for v in train}
    if w['view_names']:
        if not set(w['view_names']) <= set(by_name):
            raise ValueError('Generation sources must belong to this run training split')
        sources = [by_name[n] for n in w['view_names']]
    else:
        sources = [train[i] for i in np.linspace(0, len(train) - 1, min(w['views'], len(train)), dtype=int)]
    if not sources:
        raise ValueError('No training source selected')
    input_images=folder/'input_images';input_images.mkdir(parents=True,exist_ok=True)
    for source in sources:
        path=Path(source['image']).resolve()
        link=input_images/(source['name']+path.suffix)
        if not link.exists():link.symlink_to(path)
    commit = subprocess.check_output(['git', '-C', w['repository'], 'rev-parse', 'HEAD'], text=True).strip()
    if commit != CODE_REVISION:
        raise ValueError(f'Expected validated MUGEN revision {CODE_REVISION}; got {commit}')
    weights_manifest = Path(w['weights']) / 'provenance.json'
    if not weights_manifest.exists():
        raise FileNotFoundError('Run tools/download_wan360.py to obtain and hash official weights')
    weights = read(weights_manifest)
    for name, info in weights['files'].items():
        path = Path(w['weights']) / name
        if not path.is_file() or path.stat().st_size != info['bytes']:
            raise ValueError(f'Missing/truncated Wan360 asset: {name}')
    provenance = {'repository_commit': commit, 'official_weights': weights,
                  'worker': ref(ROOT / 'tools/wan360_worker.py'),
                  'coordinate_note': 'Wan360 uses +Y up, OmniGS/OpenMVG +Y down. Translation prior is converted with diag(1,-1,1); its scale is not metric.'}
    package_record=Path(w['weights'])/'python_packages.txt'
    if package_record.is_file():provenance['runtime_packages']=ref(package_record)
    write(folder / 'wan360_provenance.json', provenance)
    jobs, pending = [], []
    for source in sources:
        for direction in w['directions']:
            identity = {'source': ref(source['image']), 'config': {k: v for k, v in w.items() if k not in ['cache', 'reuse', 'views', 'view_names', 'directions', 'candidate_frames']},
                        'seed': cfg['seed'], 'direction': direction, 'provenance': provenance}
            key = signature(identity)
            output = Path(w['cache']) / key
            if not w['reuse']:
                output = folder / 'generation' / (source['name'] + '_' + direction)
            job = {'name': source['name'] + '_' + direction, 'source_training_view': source['name'],
                   'image': source['image'], 'direction': direction, 'seed': cfg['seed'],
                   'output': str(output), 'signature': key}
            stage_link = folder / 'generation' / job['name']
            stage_link.parent.mkdir(parents=True, exist_ok=True)
            output.mkdir(parents=True, exist_ok=True)
            if output != stage_link and not stage_link.exists():
                stage_link.symlink_to(output, target_is_directory=True)
            cached = output / 'generation.json'
            complete = cached.exists() and read(cached).get('signature') == key
            if complete:
                result = read(cached)
                complete = all(Path(p).is_file() for p in [result['video'], result['trajectory'], *[f['image'] for f in result['frames']]])
            jobs.append(job)
            if not complete:
                pending.append(job)
    write(folder / 'generation_jobs.json', {'jobs': jobs, 'config': w, 'sources': sources})
    return w, jobs, pending


def generate(folder, cfg, train):
    w, jobs, pending = plan_generation(folder, cfg, train)
    if pending:
        request = folder / 'wan360_request.json'
        write(request, {'config': w, 'jobs': pending})
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(cfg['gpu']), OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1')
        env.update(HF_HOME=str(ROOT/'weights/wan360_hf_cache'),
                   TORCH_HOME=str(ROOT/'weights/wan360_torch_cache'),
                   TRITON_CACHE_DIR=str(ROOT/'runs/m1_cache/triton'))
        run_command([w['python'], ROOT / 'tools/wan360_worker.py', request], folder / 'wan360.log', env)
    result = [read(Path(job['output']) / 'generation.json') for job in jobs]
    pending_names={job['name'] for job in pending}
    for video in result:
        video['cache_reused']=video['name'] not in pending_names
        video['available_lossless_frames']=len(video['frames'])
        video['candidate_frames_requested']=w['candidate_frames']
        video['frames']=[frame for frame in video['frames'] if frame['frame_index'] in {0,*w['candidate_frames']}]
    write(folder / 'generated_videos.json', result)
    return result
