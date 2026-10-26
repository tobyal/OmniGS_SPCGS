"""Explicit fresh IPSGR experiment stages using the frozen repaired20/6 protocol."""
import argparse, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spcgs.ipsgr_protocol import DEFAULT_RUN, audit, check_inputs
from spcgs.common import read

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=DEFAULT_RUN);p.add_argument('--stage',required=True,choices=['audit','wan','p0','m1','teacher','train','evaluate','report']);p.add_argument('--gpu',type=int,default=0);p.add_argument('--model',default='IPSGR-Full');a=p.parse_args();run=a.run.resolve()
    if a.stage=='audit':return audit(run)
    manifest=check_inputs(run);cfg=read(run/'experiment_config.json');cfg['gpu']=a.gpu
    if a.stage=='wan':
        from spcgs.wan360 import generate
        folder=run/'01_geometry/Wan360';folder.mkdir(exist_ok=True)
        return generate(folder,cfg,manifest['train'])
    if a.stage in ['p0','m1']:
        from spcgs.ipsgr_geometry import build_p0,build_p1
        return (build_p0 if a.stage=='p0' else build_p1)(run,cfg)
    if a.stage=='teacher':
        from spcgs.ipsgr_training import teacher
        return teacher(run,cfg)
    if a.stage=='train':
        from spcgs.ipsgr_training import train
        return train(run,cfg,a.model)
    if a.stage=='evaluate':
        from spcgs.ipsgr_evaluate import evaluate
        return evaluate(run,cfg,a.model)
    if a.stage=='report':
        from spcgs.ipsgr_report import report
        return report(run)

if __name__=='__main__':main()
