"""Generate every source first, then run independent CPU SfM and global M1.

Run on the Linux training server. Existing signed Wan360 caches are preserved.
Each source has its own directory; global M1 retains cross-source deduplication.
"""
import argparse
from collections import deque
import copy
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spcgs.common import ROOT, default_config, read, write, ref
from spcgs.pipeline import prepare
from spcgs.wan360 import plan_generation


def complete_video(job, frames):
    path = Path(job['output']) / 'generation.json'
    if not path.is_file():
        return False
    result = read(path)
    return (result.get('status') == 'completed' and result.get('signature') == job['signature']
            and len(result.get('frames', [])) == frames
            and all(Path(p).is_file() for p in [result['video'], result['trajectory'],
                                               *[f['image'] for f in result['frames']]]))


def cpu_env():
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--config', default='configs/m1_wan360_56.json')
    parser.add_argument('--gpus', default='0,2')
    parser.add_argument('--sfm-workers', type=int, default=4)
    parser.add_argument('--stage', choices=['all', 'generation', 'sfm'], default='all')
    args = parser.parse_args()
    if not args.run or Path(args.run).name != args.run or args.run in ['.', '..']:
        parser.error('--run must be a single directory name')
    gpus = [int(g) for g in args.gpus.split(',')]
    if len(set(gpus)) != len(gpus) or min(gpus) < 0 or args.sfm_workers < 1:
        parser.error('Use unique nonnegative GPU IDs and a positive SfM worker count')
    cfg = default_config()
    cfg.update(read(args.config))
    if cfg['m1_mode'] != 'wan360' or not cfg.get('wan360', {}).get('reuse', True):
        parser.error('This batch requires m1_mode=wan360 and reusable video caches')
    run = ROOT / 'runs/industrial_spcgs' / args.run
    batch = run / 'batch'
    batch.mkdir(parents=True, exist_ok=True)
    lock = (batch / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    active = []
    status = {'run': args.run, 'controller_pid': os.getpid(), 'started': time.time(),
              'status': 'running', 'phase': 'planning', 'gpu_ids': gpus,
              'sfm_workers': args.sfm_workers, 'source_results': {}}

    def update(**changes):
        status.update(changes, updated=time.time())
        write(run / 'batch_status.json', status)

    def launch(command, log, env):
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('a') as stream:
            stream.write('\nCOMMAND ' + ' '.join(map(str, command)) + '\n')
            stream.flush()
            child = subprocess.Popen(list(map(str, command)), cwd=ROOT, env=env,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        active.append(child)
        return child

    def stop(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        train, _ = prepare(run, cfg)
        w, jobs, pending = plan_generation(run / 'generation', cfg, train)
        names = list(dict.fromkeys(j['source_training_view'] for j in jobs))
        source_configs = {}
        for name in names:
            source = run / 'views' / name
            source.mkdir(parents=True, exist_ok=True)
            view_cfg = copy.deepcopy(cfg)
            view_cfg['wan360'].update(views=1, view_names=[name])
            config_path = batch / 'view_configs' / (name + '.json')
            write(config_path, view_cfg)
            source_configs[name] = config_path
            view_jobs = [j for j in jobs if j['source_training_view'] == name]
            for job in view_jobs:
                link = source / 'generation' / job['direction']
                link.parent.mkdir(exist_ok=True)
                if not link.exists():
                    link.symlink_to(Path(job['output']), target_is_directory=True)
            write(source / 'source_manifest.json', {'source': next(v for v in train if v['name'] == name),
                  'jobs': view_jobs, 'candidate_frames': w['candidate_frames'],
                  'all_lossless_frames_retained': w['num_frames'], 'sfm_config': str(config_path)})
        write(batch / 'plan.json', {'config': cfg, 'sources': names, 'jobs': jobs,
              'initial_pending': [j['name'] for j in pending], 'gpu_ids': gpus,
              'sfm_workers': args.sfm_workers, 'stages': ['generation_barrier', 'per_source_sfm', 'global_sfm'],
              'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'code': [ref(ROOT / p) for p in ['spcgs/wan360.py', 'spcgs/m1_geometry.py',
                        'spcgs/m1_wan360.py', 'tools/run_m1_batch.py', 'tools/wan360_worker.py']]})
        update(sources=len(names), total_videos=len(jobs), initially_cached_videos=len(jobs)-len(pending),
               completed_videos=sum(complete_video(j, w['num_frames']) for j in jobs))
        if args.stage != 'sfm':
            update(phase='generation')
            workers = []
            for i, gpu in enumerate(gpus):
                shard = pending[i::len(gpus)]
                if not shard:
                    continue
                request = batch / f'gpu_{gpu}_request.json'
                write(request, {'config': w, 'jobs': shard})
                env = cpu_env()
                env.update(CUDA_VISIBLE_DEVICES=str(gpu), HF_HOME=str(ROOT/'weights/wan360_hf_cache'),
                           TORCH_HOME=str(ROOT/'weights/wan360_torch_cache'),
                           TRITON_CACHE_DIR=str(ROOT/f'runs/m1_cache/triton_gpu{gpu}'))
                child = launch([w['python'], ROOT/'tools/wan360_worker.py', request], batch/f'gpu_{gpu}.log', env)
                workers.append((gpu, child))
            while workers:
                update(completed_videos=sum(complete_video(j, w['num_frames']) for j in jobs),
                       generation_workers=[{'gpu': gpu, 'pid': p.pid, 'returncode': p.poll()} for gpu, p in workers])
                failed = [(gpu, p.returncode) for gpu, p in workers if p.returncode not in [None, 0]]
                if failed:
                    raise RuntimeError(f'Wan360 worker failed: {failed}; inspect batch/gpu_*.log')
                if all(p.returncode == 0 for _, p in workers):
                    break
                time.sleep(15)
        incomplete = [j['name'] for j in jobs if not complete_video(j, w['num_frames'])]
        if incomplete:
            raise RuntimeError(f'Generation barrier not satisfied: {len(incomplete)} incomplete videos')
        write(batch / 'generation_complete.json', {'completed': time.time(), 'videos': len(jobs),
              'lossless_frames': len(jobs)*w['num_frames'], 'sources': names})
        update(completed_videos=len(jobs), phase='generation_complete')
        print(f'GENERATION COMPLETE: {len(jobs)} videos for {len(names)} sources', flush=True)
        if args.stage == 'generation':
            update(status='completed', completed=time.time())
            return

        # This phase starts only after all sources satisfy the generation barrier.
        update(phase='per_source_sfm')
        queue, running = deque(names), {}
        while queue or running:
            while queue and len(running) < args.sfm_workers:
                name = queue.popleft()
                folder = run / 'views' / name / '01_sgi_sfm'
                output = folder / 'outputs.json'
                if output.exists() and read(output).get('status') == 'completed':
                    status['source_results'][name] = {'status': 'completed', 'reused': True,
                        'metrics': read(folder/'metrics.json'), 'output': str(output)}
                    continue
                command = [sys.executable, '-m', 'spcgs.pipeline', '--run', f'{args.run}/views/{name}',
                           '--config', source_configs[name], '--stage', 'm1']
                running[name] = launch(command, batch/'sfm_logs'/(name+'.log'), cpu_env())
                status['source_results'][name] = {'status': 'running', 'pid': running[name].pid}
            for name, process in list(running.items()):
                if process.poll() is None:
                    continue
                folder = run / 'views' / name / '01_sgi_sfm'
                output = read(folder/'outputs.json') if (folder/'outputs.json').exists() else {}
                metrics = read(folder/'metrics.json') if (folder/'metrics.json').exists() else {}
                state = 'completed' if process.returncode == 0 and output.get('status') == 'completed' else 'failed'
                status['source_results'][name] = {'status': state, 'returncode': process.returncode,
                    'reason': output.get('reason'), 'metrics': metrics, 'output': str(folder/'outputs.json')}
                print('SOURCE SFM', name, state, metrics.get('added'), flush=True)
                del running[name]
            update(sfm_finished=sum(r['status'] in ['completed','failed'] for r in status['source_results'].values()),
                   sfm_failed=sum(r['status']=='failed' for r in status['source_results'].values()))
            if queue or running:
                time.sleep(5)
        write(batch / 'source_results.json', status['source_results'])

        # Reuse feature/match caches, retaining existing cross-source track/point
        # deduplication and density filtering instead of concatenating local clouds.
        update(phase='global_sfm')
        aggregate = launch([sys.executable, '-m', 'spcgs.pipeline', '--run', args.run,
                            '--config', run/'config.json', '--stage', 'm1'], batch/'global_sfm.log', cpu_env())
        while aggregate.poll() is None:
            metrics = run/'01_sgi_sfm/metrics.json'
            update(global_pid=aggregate.pid, global_metrics=read(metrics) if metrics.exists() else {})
            time.sleep(15)
        output = read(run/'01_sgi_sfm/outputs.json')
        if aggregate.returncode != 0 or output.get('status') != 'completed':
            raise RuntimeError(f'Global M1 failed: {output.get("reason")}; inspect batch/global_sfm.log')
        update(status='completed', phase='completed', completed=time.time(),
               global_metrics=read(run/'01_sgi_sfm/metrics.json'), global_outputs=output)
        print('BATCH COMPLETE', run, flush=True)
    except BaseException as error:
        update(status='failed', reason=str(error), failed=time.time())
        (batch/'controller_error.log').write_text(traceback.format_exc())
        raise
    finally:
        for process in active:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        lock.close()


if __name__ == '__main__':
    main()
