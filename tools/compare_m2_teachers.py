"""Same RGB: legacy features/old labels, shared new text, dense cosine and S."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image,ImageDraw
from spcgs.common import ROOT,read,write,ref
from spcgs.clip_dense import unit,spatial_relevancy,save_responses
from spcgs.visuals import PALETTE

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='runs/industrial_semantics_20261023/01_clip');a=p.parse_args();root=ROOT/a.root
    new=read(root/'features_pilot/manifest.json');old=read(root/'legacy_pilot/manifest.json');out=root/'teacher_comparison';out.mkdir(exist_ok=True)
    text=np.fromfile(new['text'],dtype='f4').reshape(-1,512);metrics=[]
    for v in new['views']:
        o=next(o for o in old['views'] if o['name']==v['name']);name=v['name'];h,w=v['feature_height'],v['feature_width']
        oldmaps=np.load(Path(o['feature']).parent/(name+'_relevancy.npz'));oldfeat=unit(np.fromfile(o['feature'],dtype='f2').astype('f4').reshape(512,h,w),0)
        shared=text@oldfeat.reshape(512,-1);shared=shared.reshape(len(text),h,w);oldrel=spatial_relevancy(shared)
        sm=read(v['S_manifest']);newmaps=np.load(sm['maps']);np.savez_compressed(out/(name+'_legacy_shared_text.npz'),cosine=shared.astype('f2'),relevancy=oldrel.astype('f2'))
        save_responses(v['image'],oldrel,new['classes'],out/(name+'_legacy_responses.jpg'))
        import shutil
        shutil.copy2(Path(sm['maps']).parent/'responses.jpg',out/(name+'_dense_responses.jpg'))
        rgb=Image.open(v['image']).convert('RGB').resize((768,384));canvas=Image.new('RGB',(1536,1420),'white');draw=ImageDraw.Draw(canvas)
        panels=[('RGB',None),('OLD local_crops + original 12 classes',oldmaps['labels']),('OLD local_crops + NEW prompt ensemble (cosine)',shared.argmax(0)),('NEW multiscale + same text (raw cosine)',newmaps['cosine'].argmax(0)),('NEW multiscale cached S (spatial minmax, mean scales)',newmaps['S'].argmax(0))]
        for i,(title,labels) in enumerate(panels):
            x=i%2*768;y=i//2*412;draw.text((x+6,y+5),title,fill='black')
            im=rgb if labels is None else Image.blend(rgb,Image.fromarray(PALETTE[labels]).resize(rgb.size,Image.Resampling.NEAREST),.65)
            canvas.paste(im,(x,y+24))
        for i,c in enumerate(new['classes']):
            x=780+(i%2)*375;y=850+(i//2)*32;draw.rectangle((x,y,x+18,y+18),fill=tuple(PALETTE[i]));draw.text((x+24,y+3),f'{i}: {c}',fill='black')
        draw.text((8,1250),'OLD 12-class legend (top-right panel ONLY):',fill='black')
        for i,c in enumerate(old['classes']):
            x=8+(i%6)*250;y=1280+(i//6)*40;draw.rectangle((x,y,x+18,y+18),fill=tuple(PALETTE[i]));draw.text((x+24,y+3),c,fill='black')
        canvas.save(out/(name+'.jpg'),quality=95)
        metrics.append({'view':name,'old_new_raw_label_agreement':float((shared.argmax(0)==newmaps['cosine'].argmax(0)).mean()),'new_raw_vs_S_label_agreement':float((newmaps['cosine'].argmax(0)==newmaps['S'].argmax(0)).mean()),'S_uncertainty':sm['uncertainty'],'note':'Agreement and entropy are diagnostics, not semantic accuracy.'})
    write(out/'metrics.json',{'new':ref(root/'features_pilot/manifest.json'),'old':ref(root/'legacy_pilot/manifest.json'),'views':metrics})
    print(out)
if __name__=='__main__':main()
