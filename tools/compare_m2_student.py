"""Saved F/S/T versus actual native Gaussian renders; no ground-truth claims."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from spcgs.common import ROOT,read,write,ref
from spcgs.clip_dense import unit,save_responses,spatial_relevancy
from spcgs.visuals import PALETTE

def softmax(x):
    a=np.exp(x-x.max(0));return a/a.sum(0)

def compare(stage,cache,cfg,views):
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    stage=Path(stage);out=stage/'validation';out.mkdir(exist_ok=True);h,w=cfg['semantic_height'],cfg['semantic_width'];k=len(cache['classes']);rows=[]
    try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    except OSError:font=ImageFont.load_default()
    for v in views:
        name=v['name'];fh,fw=v['feature_height'],v['feature_width']
        def resized(x):return np.stack([cv2.resize(z,(w,h),interpolation=cv2.INTER_LINEAR) for z in x])
        f=unit(resized(np.fromfile(v['feature'],dtype='f2').astype('f4').reshape(512,fh,fw)),0)
        s=resized(np.fromfile(v['relevancy'],dtype='f2').astype('f4').reshape(k,v['relevancy_height'],v['relevancy_width']));target=softmax(s);teacher=s.argmax(0)
        rgb=Image.open(v['image']).convert('RGB').resize((768,384));canvas=Image.new('RGB',(1536,2050),'white');draw=ImageDraw.Draw(canvas)
        def label_panel(labels):return Image.blend(rgb,Image.fromarray(PALETTE[labels]).resize(rgb.size,Image.Resampling.NEAREST),.65)
        panels=[('RGB',rgb),('Cached S teacher',label_panel(teacher))];row={'view':name,'states':{}}
        for state in ['initial','final']:
            folder=out/state;low=np.fromfile(folder/(name+'_features.f16'),dtype='f2').astype('f4').reshape(32,-1)
            decoder=np.fromfile(stage/state/'decoder.f32',dtype='f4').reshape(512,32);bias=np.fromfile(stage/state/'decoder_bias.f32',dtype='f4')
            decoded=unit((decoder@low+bias[:,None]).reshape(512,h,w),0);cos=(decoded*f).sum(0)
            logits=np.fromfile(folder/(name+'_logits.f16'),dtype='f2').astype('f4').reshape(k,h,w);probs=softmax(logits);labels=logits.argmax(0)
            ids=np.fromfile(folder/(name+'_ids.i32'),dtype='i4').reshape(h,w);valid=ids>=0
            ce=-(target*np.log(probs+1e-12)).sum(0);entropy=-(target*np.log(target+1e-12)).sum(0);kl=ce-entropy
            row['states'][state]={'feature_cosine':float(cos.mean()),'soft_ce':float(ce.mean()),'target_entropy':float(entropy.mean()),'target_KL':float(kl.mean()),
                'S_argmax_agreement':float((labels==teacher).mean()),'valid_fraction':float(valid.mean()),'agreement_on_valid':float((labels[valid]==teacher[valid]).mean()) if valid.any() else None,
                'relevancy_probability_MAE':float(np.abs(probs-target).mean())}
            np.savez_compressed(folder/(name+'_teacher_student.npz'),feature_cosine=cos,teacher_S=s,teacher_probabilities=target,student_logits=logits,student_probabilities=probs,valid=valid,target_KL=kl)
            panels.append((state+' Gaussian class head',label_panel(labels)))
            heat=cv2.applyColorMap((np.clip((cos-.98)/.02,0,1)*255).astype('u1'),cv2.COLORMAP_TURBO)[:,:,::-1]
            panels.append((state+' cosine to F (fixed display range: .98 blue, 1 red)',Image.fromarray(heat).resize(rgb.size)))
            save_responses(v['image'],probs,cache['classes'],folder/(name+'_class_probabilities.jpg'))
            save_responses(v['image'],spatial_relevancy(probs),[c+' (spatial minmax)' for c in cache['classes']],folder/(name+'_relative_responses.jpg'))
        panels.extend([('Initial rendered RGB',Image.open(out/'initial'/(name+'_rgb.png')).resize(rgb.size)),('Final rendered RGB',Image.open(out/'final'/(name+'_rgb.png')).resize(rgb.size))])
        for i,(title,im) in enumerate(panels):
            x=i%2*768;y=i//2*410;draw.text((x+5,y+2),title,fill='black',font=font);canvas.paste(im,(x,y+24))
        for i,c in enumerate(cache['classes']):
            x=i%3*500+12;y=1660+i//3*42;draw.rectangle((x,y,x+20,y+20),fill=tuple(PALETTE[i]));draw.text((x+28,y),c,fill='black',font=font)
        canvas.save(out/(name+'_comparison.jpg'),quality=94);rows.append(row)
    losses=[__import__('json').loads(x) for x in (stage/'losses.jsonl').read_text().splitlines()]
    # Keep the original supervision panels and expose existing regularizers,
    # generated-view loss and model growth so a falling CE cannot hide them.
    keys=['semantic_cosine','semantic_ce','rgb','local3d','generated_semantic','total','gaussians','semantic_grad_norm','decoder_grad_norm']
    fig,ax=plt.subplots(3,3,figsize=(15,10))
    for a,key in zip(ax.flat,keys):
        a.plot([x['step'] for x in losses],[x[key] for x in losses],alpha=.6,lw=.7);a.set_title(key);a.set_xlabel('iteration');a.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'losses.png',dpi=150);plt.close(fig)
    summary={'views':rows,'note':'Teacher agreement/KL are supervision-fitting diagnostics, not mIoU or semantic accuracy. Labels can be unstable because S softmax is nearly uniform.',
             'mean_initial':{key:float(np.mean([r['states']['initial'][key] for r in rows])) for key in rows[0]['states']['initial'] if key!='agreement_on_valid'},
             'mean_final':{key:float(np.mean([r['states']['final'][key] for r in rows])) for key in rows[0]['states']['final'] if key!='agreement_on_valid'},
             'training_first':losses[0],'training_last':losses[-1],'model':ref(stage/'final/model.pt'),'status':'completed'}
    write(out/'metrics.json',summary);return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('cache');p.add_argument('config');a=p.parse_args();c=read(ROOT/a.cache)
    compare(ROOT/a.stage,c,read(ROOT/a.config),c['views'])
