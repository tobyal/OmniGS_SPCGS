from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .common import project
PALETTE=np.array([[228,26,28],[77,175,74],[55,126,184],[255,127,0],[152,78,163],[255,255,51],[166,86,40],[247,129,191],[153,153,153],[0,200,180],[120,100,240],[210,190,120]],dtype='u1')
def semantic_preview(rgb,labels,classes,out):
    im=Image.open(rgb).convert('RGB');im.thumbnail((768,384));seg=Image.fromarray(PALETTE[labels%len(PALETTE)]).resize(im.size,Image.Resampling.NEAREST)
    invalid=np.array(Image.fromarray((labels>=len(classes)).astype('u1')).resize(im.size,Image.Resampling.NEAREST))>0
    pixels=np.array(seg);pixels[invalid]=np.array(im)[invalid];seg=Image.fromarray(pixels)
    canvas=Image.new('RGB',(im.width,im.height*2+22*((len(classes)+5)//6)+11),'white');canvas.paste(im,(0,0));canvas.paste(Image.blend(im,seg,.55),(0,im.height));draw=ImageDraw.Draw(canvas)
    for i,c in enumerate(classes):
        x=(i%6)*(im.width//6);y=im.height*2+(i//6)*22;draw.rectangle((x,y+2,x+12,y+14),fill=tuple(PALETTE[i%len(PALETTE)]));draw.text((x+16,y),c,fill='black')
    Path(out).parent.mkdir(parents=True,exist_ok=True);canvas.save(out)
def projection_preview(xyz,rgb,cam,out):
    im=Image.open(cam['image']).convert('RGB');im.thumbnail((1024,512));u,v,r=project(xyz,cam,im.height,im.width);draw=ImageDraw.Draw(im)
    ids=np.argsort(r)[::-1];ids=ids[::max(1,len(ids)//20000)]
    for i in ids:
        x=int(round(u[i]))%im.width;y=int(round(v[i]));c=tuple((rgb[i]*255).astype('u1'));draw.ellipse((x-1,y-1,x+1,y+1),fill=c)
    Path(out).parent.mkdir(parents=True,exist_ok=True);im.save(out)
