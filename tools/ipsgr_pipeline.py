"""Dependency runner for the authorized complete experiment; no metric-driven tuning."""
import sys,subprocess,time,traceback,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spcgs.common import ROOT,read,write
from spcgs.ipsgr_protocol import DEFAULT_RUN,MODEL_NAMES

def main():
    run=DEFAULT_RUN;logs=[];jobs={};started=time.time();events=[]
    def state(path):return read(path).get('status') if path.exists() else None
    # A controller restart adopts its recorded stage processes; it never launches
    # another copy of a still-running training job.
    class Adopted:
        def __init__(self,event):self.event=event;self.pid=event['pid'];self.returncode=None
        def poll(self):
            proc=Path('/proc')/str(self.pid)
            if proc.exists():
                try:
                    cmd=(proc/'cmdline').read_text();status=(proc/'stat').read_text().split()[2]
                    if status!='Z' and 'tools/run_ipsgr.py' in cmd and str(run) in cmd:return None
                except FileNotFoundError:pass
            key=self.event['key']
            if key=='M1':ok=state(run/'01_geometry/P1/outputs.json')=='completed'
            elif key.startswith('evaluate-'):ok=(run/'06_evaluation'/key[len('evaluate-'):]/'metrics.json').exists()
            else:
                from spcgs.ipsgr_training import model_folder
                ok=state(model_folder(run,key)/'outputs.json')=='completed'
            self.returncode=0 if ok else 1;return self.returncode
    if (run/'10_report/runner_events.json').exists():
        events=read(run/'10_report/runner_events.json')
        for event in events:jobs[event['key']]=Adopted(event)
        if (run/'10_report/live_status.json').exists():started=read(run/'10_report/live_status.json')['started']
        print('ADOPTED_RECORDED_JOBS',[(k,p.pid) for k,p in jobs.items()],flush=True)
    def spawn(key,stage,gpu=0,model=None):
        if key in jobs:return
        args=[str(ROOT/'.venv/bin/python'),'tools/run_ipsgr.py','--run',str(run),'--stage',stage,'--gpu',str(gpu)]
        if model:args+=['--model',model]
        f=(run/'10_report'/(key+'_driver.log')).open('a');logs.append(f);p=subprocess.Popen(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);jobs[key]=p;events.append({'key':key,'pid':p.pid,'gpu':gpu,'started':time.time(),'args':args});write(run/'10_report/runner_events.json',events);print('LAUNCHED',key,p.pid,gpu,flush=True)
    while True:
        status={n:state(run/'05_models'/n/'outputs.json') for n in MODEL_NAMES};layout=state(run/'03_layout_bootstrap/outputs.json');wan=(run/'01_geometry/Wan360/parallel_completed.json').exists();p1=state(run/'01_geometry/P1/outputs.json');smoke=state(run/'04_cross_view/smoke-IPSGR-Full/outputs.json')
        for key,p in jobs.items():
            if p.poll() not in [None,0]:raise RuntimeError(f'{key} exited{p.returncode}; inspect its driver log')
        if any(status[n]=='failed' for n in ['B-RGB','B-SEM']):raise RuntimeError('A baseline failed; evidence retained')
        prefetch=state(run/'01_geometry/prefetch/status.json')
        if wan and prefetch=='completed' and p1!='completed':spawn('M1','m1')
        if p1=='completed':
            if layout!='completed':spawn('ISLB','train',2,'ISLB')
            if status['G-AUG']!='completed':spawn('G-AUG','train',0,'G-AUG')
        if layout=='completed':
            if smoke!='completed':spawn('smoke-IPSGR-Full','train',2,'smoke-IPSGR-Full')
            if status['L-BOOT']!='completed' and status['B-SEM']=='completed':spawn('L-BOOT','train',3,'L-BOOT')
        if smoke=='completed':
            if status['CV-2D']!='completed':spawn('CV-2D','train',0 if status['G-AUG']=='completed' else 1,'CV-2D')
            if status['IPSGR-Full']!='completed':spawn('IPSGR-Full','train',2,'IPSGR-Full')
        # Evaluation is serialized on GPU0 once its generation process has ended.
        ws=read(run/'01_geometry/Wan360/parallel_status.json')
        gpu0ready=ws['workers']['0']['complete']==ws['workers']['0']['total']
        evaluations=[k for k,p in jobs.items() if k.startswith('evaluate-') and p.poll() is None]
        if gpu0ready and not evaluations:
            for name in MODEL_NAMES:
                if status[name]=='completed' and not (run/'06_evaluation'/name/'metrics.json').exists() and 'evaluate-'+name not in jobs:
                    spawn('evaluate-'+name,'evaluate',0,name);break
        write(run/'10_report/live_status.json',{'started':started,'updated':time.time(),'wan':ws['complete'],'P1':p1,'ISLB':layout,'smoke':smoke,'models':status,'evaluated':[n for n in MODEL_NAMES if (run/'06_evaluation'/n/'metrics.json').exists()]})
        if all((run/'06_evaluation'/n/'metrics.json').exists() for n in MODEL_NAMES):break
        time.sleep(10)
    from spcgs.ipsgr_evaluate import high_resolution_check
    from spcgs.ipsgr_report import report
    cfg=read(run/'experiment_config.json');cfg['gpu']=2;high_resolution_check(run,cfg);report(run)
    print('EXPERIMENT_COMPLETE',flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        write(DEFAULT_RUN/'10_report/runner_failure.json',{'time':time.time(),'traceback':traceback.format_exc()});raise
