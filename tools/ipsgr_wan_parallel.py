"""Disjoint multi-GPU execution of the already signed fresh80 Wan360 jobs."""
import sys,os,json,time,signal,subprocess,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spcgs.common import ROOT,read,write,ref
from spcgs.ipsgr_protocol import DEFAULT_RUN

def complete(job):
    p=Path(job['output'])/'generation.json'
    if not p.exists():return False
    try:
        r=read(p)
        return r['signature']==job['signature'] and all(Path(x).is_file() for x in [r['video'],r['trajectory'],*[v['image'] for v in r['frames']]])
    except (ValueError,KeyError):return False

def main():
    run=Path(sys.argv[1]) if len(sys.argv)>1 else DEFAULT_RUN;folder=run/'01_geometry/Wan360';request=read(folder/'wan360_request.json');jobs=request['jobs'];cfg=request['config'];assert len(jobs)==80
    original_info=read(run/'01_geometry/wan_process.json');original=original_info['pid'];groups={0:jobs[:28]+jobs[47:49],1:jobs[28:47],2:jobs[54:]+jobs[49:54]};workers={};logs=[];previous=read(folder/'parallel_status.json') if (folder/'parallel_status.json').exists() else {};started=previous.get('started',time.time());attempts={0:0,1:0,2:0}
    original_group=original
    def alive(pid):
        try:
            os.kill(pid,0)
            return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
        except (ProcessLookupError,FileNotFoundError):return False
    class Adopted:
        def __init__(self,pid):self.pid=pid
        def poll(self):return None if alive(self.pid) else 0
    for gpu in [1,2]:
        pid=previous.get('workers',{}).get(str(gpu),{}).get('pid')
        if pid and alive(pid):
            cmd=Path(f'/proc/{pid}/cmdline').read_text();assert str(ROOT/'tools/wan360_worker.py') in cmd
            workers[gpu]=Adopted(pid);attempts[gpu]=1
    def launch(gpu):
        pending=[j for j in groups[gpu] if not complete(j)]
        if not pending:return
        attempts[gpu]+=1;rq=folder/f'parallel_gpu{gpu}_attempt{attempts[gpu]}.json';write(rq,{'config':cfg,'jobs':pending});log=folder/f'parallel_gpu{gpu}_attempt{attempts[gpu]}.log';f=log.open('a');logs.append(f)
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1',HF_HOME=str(ROOT/'weights/wan360_hf_cache'),TORCH_HOME=str(ROOT/'weights/wan360_torch_cache'),TRITON_CACHE_DIR=str(ROOT/'runs/m1_cache/triton'))
        p=subprocess.Popen([cfg['python'],str(ROOT/'tools/wan360_worker.py'),str(rq)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);workers[gpu]=p
        print('START',gpu,p.pid,len(pending),flush=True)
    if (folder/'parallel_assignment.json').exists():write(folder/f'parallel_assignment_before_{time.time_ns()}.json',read(folder/'parallel_assignment.json'))
    write(folder/'parallel_assignment.json',{'original_pid':original,'workers':{str(g):[j['name'] for j in group] for g,group in groups.items()},'training_gpu_reserved':3,'all_jobs_disjoint':True,'generation_config_unchanged':True,'worker':ref(ROOT/'tools/wan360_worker.py'),'rebalance_note':'GPU1 slow under shared load: last5 tasks to GPU2, and another2 still-unstarted tasks to the newly free GPU0. GPU1 stops only after its first19 outputs commit. All assignments remain disjoint; original GPU0 worker already stopped after28.'})
    for gpu in [1,2]:
        if gpu not in workers:launch(gpu)
    while True:
        state={str(g):{'complete':sum(complete(j) for j in group),'total':len(group),'pid':original if g==0 and original else workers[g].pid if g in workers else None} for g,group in groups.items()}
        write(folder/'parallel_status.json',{'started':started,'updated':time.time(),'workers':state,'complete':sum(v['complete'] for v in state.values()),'total':80})
        # The original worker was started with all80 jobs. End only our process
        # group once its assigned first28 are safely and atomically committed.
        if original and all(complete(j) for j in jobs[:28]):
            if alive(original):os.killpg(original_group,signal.SIGTERM)
            print('ORIGINAL_SHARD_FINISHED',original,flush=True);original=None
        if state['1']['complete']==len(groups[1]) and 1 in workers and workers[1].poll() is None:
            os.killpg(workers[1].pid,signal.SIGTERM);print('GPU1_ASSIGNED_SHARD_FINISHED',flush=True)
        if all(complete(j) for j in jobs):break
        for gpu,group in groups.items():
            if all(complete(j) for j in group):continue
            if gpu==0 and original:
                if alive(original):continue
                original=None
            if gpu in workers and workers[gpu].poll() is None:continue
            if attempts[gpu]>=3:raise RuntimeError(f'GPU{gpu} worker failed three times; logs preserved')
            launch(gpu)
        time.sleep(2)
    videos=[]
    for job in jobs:
        v=read(Path(job['output'])/'generation.json');v.update(cache_reused=False,available_lossless_frames=len(v['frames']),candidate_frames_requested=cfg['candidate_frames']);v['frames']=[f for f in v['frames'] if f['frame_index'] in {0,*cfg['candidate_frames']}];videos.append(v)
    write(folder/'generated_videos.json',videos);write(folder/'parallel_completed.json',{'status':'completed','jobs':len(videos),'seconds':time.time()-started,'seconds_since_first_single_gpu_launch':time.time()-original_info['started'],'summed_video_generation_seconds':sum(v['seconds'] for v in videos),'assignments':ref(folder/'parallel_assignment.json'),'all_fresh_for_this_experiment':True})
    print('ALL80_COMPLETE',flush=True)

if __name__=='__main__':main()
