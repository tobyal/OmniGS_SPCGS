"""Visual diagnostics only: inspect each cached scale without changing targets."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image,ImageDraw
from spcgs.common import ROOT,read,write
from spcgs.visuals import PALETTE

def main():
    p=argparse.ArgumentParser();p.add_argument('manifest');p.add_argument('--output',required=True);a=p.parse_args()
    cache=read(ROOT/a.manifest);out=ROOT/a.output;out.mkdir(parents=True,exist_ok=True);rows=[]
    for v in cache['views']:
        sm=read(v['S_manifest']);m=np.load(sm['maps']);fm=read(v['F_manifest'])
        rgb=Image.open(v['image']).convert('RGB').resize((768,384));canvas=Image.new('RGB',(1536,2040),'white');d=ImageDraw.Draw(canvas)
        panels=[('RGB',None),('Mean scales: raw cosine',m['cosine'])]
        for s in range(3):panels.extend([(f'Scale {s}: raw cosine',m['cosine_scales'][s]),(f'Scale {s}: spatial minmax',m['relevancy_scales'][s])])
        for i,(title,scores) in enumerate(panels):
            x=i%2*768;y=i//2*410;d.text((x+6,y+5),title,fill='black')
            im=rgb if scores is None else Image.blend(rgb,Image.fromarray(PALETTE[scores.argmax(0)]).resize(rgb.size,Image.Resampling.NEAREST),.65)
            canvas.paste(im,(x,y+24))
        for i,name in enumerate(cache['classes']):
            x=i%3*500+12;y=1650+i//3*42;d.rectangle((x,y,x+20,y+20),fill=tuple(PALETTE[i]));d.text((x+28,y+4),name,fill='black')
        canvas.save(out/(v['name']+'.jpg'),quality=94)
        rows.append({'view':v['name'],'face_size':fm['recipe']['face_size'],'patch_count':fm['patch_count'],'uncertainty':sm['uncertainty'],
                     'scales_label_agreement':[float((m['cosine_scales'][0].argmax(0)==m['cosine_scales'][i].argmax(0)).mean()) for i in (1,2)]})
    write(out/'metrics.json',{'views':rows})
if __name__=='__main__':main()
