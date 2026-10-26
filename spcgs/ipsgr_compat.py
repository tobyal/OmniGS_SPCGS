"""Current protocol dispatch for the historical sparse27/spc27 CLI names."""
import argparse
from pathlib import Path
from .common import read
from .ipsgr_protocol import DEFAULT_RUN,audit

def main(stage_family):
    p=argparse.ArgumentParser(description='Current repaired20/6 IPSGR protocol; --legacy27 explicitly selects historical code.')
    p.add_argument('--run',type=Path,default=DEFAULT_RUN);p.add_argument('--stage',required=True);p.add_argument('--gpu',type=int,default=0);p.add_argument('--model',default='IPSGR-Full');a=p.parse_args();run=a.run.resolve()
    if a.stage in ['protocol','prepare','audit']:return audit(run)
    cfg=read(run/'experiment_config.json');cfg['gpu']=a.gpu
    if a.stage in ['p0','m1']:
        from .ipsgr_geometry import build_p0,build_p1
        return (build_p0 if a.stage=='p0' else build_p1)(run,cfg)
    if a.stage=='teacher':
        from .ipsgr_training import teacher
        return teacher(run,cfg)
    aliases={'sgi':'ISLB','rgb':'B-RGB','c0':'L-BOOT','c1':'CV-2D','c2':'IPSGR-Full','short':'smoke-IPSGR-Full'}
    if a.stage in aliases or a.stage=='train':
        from .ipsgr_training import train
        return train(run,cfg,aliases.get(a.stage,a.model))
    if a.stage in ['render','evaluate']:
        from .ipsgr_evaluate import evaluate
        return evaluate(run,cfg,aliases.get(a.model,a.model))
    if a.stage=='report':
        from .ipsgr_report import report
        return report(run)
    p.error('Use protocol/p0/wan/m1/teacher/sgi/rgb/c0/c1/c2/render/report, or tools/run_ipsgr.py.')
