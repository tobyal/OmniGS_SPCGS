"""Native integration test: hold F/T fixed and change only cached S.

Runs two one-step fits on existing M1 points; keeps all requests and renders.
This is a data-flow test, not an industrial semantic accuracy benchmark.
"""
import argparse,sys,copy,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from spcgs.common import ROOT,read,write,native,signature
from spcgs.pipeline import base_request
from spcgs.features import init_semantics
from tools.m2_semantics import config

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    cfg=config('configs/m2_industrial_multiscale.json');cfg.update(width=512,height=256,semantic_width=256,semantic_height=128)
    cache=read(ROOT/a.cache);cfg['classes']=cache['classes'];out=ROOT/a.output;out.mkdir(parents=True,exist_ok=True)
    m1=read(ROOT/cfg['m1_source']);xyz=np.fromfile(m1['points'],dtype='f4').reshape(-1,6)[:,:3]
    initial=init_semantics(xyz,cache,out/'initial_semantic.f32');v=cache['views'][0];s=np.fromfile(v['relevancy'],dtype='f2').reshape(len(cache['classes']),v['relevancy_height'],v['relevancy_width'])
    s[::-1].copy().tofile(out/'permuted_S.f16');rows=[]
    for name in ['original','permuted']:
        view=copy.deepcopy(v)
        if name=='permuted':view['relevancy']=str(out/'permuted_S.f16')
        req={**base_request(cfg),'action':'train','output':str(out/name),'iterations':1,'save_every':1,'views':[view],
             'points':m1['points'],'point_count':m1['point_count'],'extent':1.,'text':cache['text'],'decoder':cache['decoder'],
             'initial_semantic':initial,'densify_until':0,'ogr':False,'spc2d':False,'spc3d':False,'generated_views':[]}
        req['signature']=signature(req);native(req,out/name)
        row=json.loads((out/name/'losses.jsonl').read_text().splitlines()[0]);rows.append(row)
    assert abs(rows[0]['semantic_cosine']-rows[1]['semantic_cosine'])<1e-6,'F cosine changed with fixed initialization'
    assert abs(rows[0]['semantic_ce']-rows[1]['semantic_ce'])>1e-5,'CE is insensitive to cached S'
    assert all(r['semantic_grad_norm']>0 and r['decoder_grad_norm']>0 for r in rows)
    # Independently recompute CE from native initial logits and the saved S.
    folder=out/'original'
    native({**base_request(cfg),'action':'render','archive':str(folder/'initial/model.pt'),'output':str(folder/'initial_render'),'views':[v]},folder/'render_job')
    logits=np.fromfile(folder/'initial_render'/(v['name']+'_logits.f16'),dtype='f2').astype('f4').reshape(s.shape)
    z=s.astype('f4');target=np.exp(z-z.max(0));target/=target.sum(0)
    if v.get('semantic_target')=='hard_argmax':target=np.eye(len(cache['classes']),dtype='f4')[s.argmax(0)].transpose(2,0,1)
    z=logits-logits.max(0);logprob=z-np.log(np.exp(z).sum(0));ce=float(-(target*logprob).sum(0).mean())
    assert abs(ce-rows[0]['semantic_ce'])<3e-4,'Native CE differs from independent soft-target CE'
    write(out/'results.json',{'status':'passed','original':rows[0],'permuted_S_only':rows[1],'independent_CE_from_fp16_render':ce,
                             'note':'Only S rows were reversed; F, T, Gaussian initialization, seed and camera were held fixed.'})
    print(out/'results.json')
if __name__=='__main__':main()
