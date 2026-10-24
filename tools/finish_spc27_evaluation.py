"""Continue this experiment on its two assigned GPUs as stages finish."""
import argparse,json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.run_spc27 import ROOT,RUN

def wait_completed(model):
    while True:
        path=RUN/'05_spc'/model/'outputs.json'
        status=json.loads(path.read_text()).get('status') if path.exists() else None
        if status=='completed':return
        if status=='failed':raise RuntimeError(f'{model} failed; inspect {path}')
        time.sleep(30)

def execute(*args):
    print('RUN',*args,flush=True)
    subprocess.run([sys.executable,*args],cwd=ROOT,check=True)

def evaluate(model,gpu):
    execute('tools/run_spc27.py','--stage','render','--model',model,'--gpu',str(gpu))
    for action in ['lpips','query']:
        execute('-m','spcgs.spc27_evaluate',str(RUN),'--action',action,'--model',model,'--gpu',str(gpu))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--queue',choices=['c1','c2'],required=True);a=p.parse_args()
    if a.queue=='c1':
        print('Waiting for C0 to release GPU 2',flush=True);wait_completed('c0')
        execute('tools/run_spc27.py','--stage','c1','--gpu','2')
        execute('tools/check_stage2_initialization.py',str(RUN))
        evaluate('c1',2)
    else:
        print('Waiting for C2 to release GPU 0',flush=True);wait_completed('c2')
        evaluate('c2',0);wait_completed('c0');evaluate('c0',0)
    print('QUEUE COMPLETED',a.queue,flush=True)
