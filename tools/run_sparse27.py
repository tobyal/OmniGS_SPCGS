"""Explicit sparse experiment stages. Never generates videos or launches SPC."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spcgs.common import ROOT,default_config,read,write
from spcgs.sparse27 import prepare_protocol,build_p0,build_m1,train

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default='sparse27_official_sgi_20261024/opacity_lr_fixed');p.add_argument('--stage',choices=['protocol','p0','m1','teacher','short','sgi','rgb','render','report'],required=True);p.add_argument('--gpu',type=int,default=0);p.add_argument('--model',choices=['short','sgi','rgb'],default='sgi');p.add_argument('--initial',action='store_true');p.add_argument('--representatives',action='store_true');a=p.parse_args()
    run=ROOT/'runs/industrial_spcgs'/a.run;cfg=default_config('full');cfg.update(seed=42,gpu=a.gpu,train_views=27)
    if a.stage=='protocol':prepare_protocol(run)
    elif a.stage=='p0':build_p0(run,cfg)
    elif a.stage=='m1':build_m1(run,cfg)
    elif a.stage=='teacher':
        from spcgs.official_teacher import extract
        extract(run,cfg)
    elif a.stage=='render':
        from spcgs.sparse27_evaluate import render_stage
        render_stage(run,a.model,a.initial,a.representatives,a.gpu)
    elif a.stage=='report':
        from spcgs.sparse27_evaluate import report
        report(run)
    else:train(run,cfg,a.stage)

if __name__=='__main__':main()
