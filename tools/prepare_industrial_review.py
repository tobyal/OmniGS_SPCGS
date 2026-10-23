"""Prepare training-only visual material for direct human/Codex inspection.

This script projects RGB and inventories metadata. It performs no recognition,
does not read segmentation pixels, and never marks an image as visually reviewed.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
import cv2
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spcgs.common import ROOT, cameras, read, write, ref

FACES = [('front', np.eye(3)), ('right', np.array([[0,0,1],[0,1,0],[-1,0,0]])),
         ('back', np.diag([-1,1,-1])), ('left', np.array([[0,0,-1],[0,1,0],[1,0,0]])),
         ('up', np.array([[1,0,0],[0,0,-1],[0,1,0]])),
         ('down', np.array([[1,0,0],[0,0,1],[0,-1,0]]))]

def perspective(rgb, rotation, size=512, fov=100):
    t = ((np.arange(size, dtype=np.float32)+.5)/size*2-1)*np.tan(np.deg2rad(fov/2))
    x,y = np.meshgrid(t,t)
    d = np.stack([x,y,np.ones_like(x)],-1) @ rotation.T
    d /= np.linalg.norm(d,axis=-1,keepdims=True)
    h,w = rgb.shape[:2]
    u = ((np.arctan2(d[...,0],d[...,2])/(2*np.pi)+.5)*w-.5).astype('f4')
    v = np.clip((np.arcsin(d[...,1])/np.pi+.5)*h-.5,0,h-1).astype('f4')
    padded = np.concatenate([rgb[:,-1:],rgb,rgb[:,:1]],axis=1)
    return cv2.remap(padded,u+1,v,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='runs/industrial_semantics_20261023/00_scene_review')
    p.add_argument('--annotations',default=str(ROOT.parent/'datasets/images_xphase_1_cubemap_semantics_1500/_source/annotations_segmentation_cubemaps_xphase.json'))
    a=p.parse_args();out=ROOT/a.output
    train=cameras('train');val=cameras('val');test=cameras('test')
    assert len(train)==56 and not set(v['name'] for v in train)&set(v['name'] for v in val+test)
    out.mkdir(parents=True,exist_ok=True);(out/'sheets').mkdir(exist_ok=True)
    source=read(a.annotations)
    groups=defaultdict(list)
    for c in source['categories']:groups[(c['id'],c['supercategory'])].append(c['name'])
    names={v['name'] for v in train}
    correspondence={name:[] for name in names}
    for im in source['images']:
        stem=Path(im['file_name']).stem
        name,face=stem.rsplit('_',1)
        if name in names:correspondence[name].append({k:im[k] for k in ['id','file_name','width','height']})
    official={'source':ref(a.annotations),'info':source.get('info'),
              'note':'Repeated category IDs identify CAD instance names; group by id and supercategory. Category metadata only, no segmentation masks used.',
              'categories':[{'id':i,'name':s,'instance_names':n,'instance_name_count':len(n)} for (i,s),n in sorted(groups.items())],
              'training_cubemap_correspondence':correspondence}
    write(out/'official_categories.json',official)
    rows=[]
    for index,cam in enumerate(train,1):
        name=cam['name'];folder=out/'images'/name;folder.mkdir(parents=True,exist_ok=True)
        image=Image.open(cam['image']).convert('RGB');rgb=np.asarray(image)
        assert image.size==(2048,1024)
        erp=folder/'erp.png'
        if not erp.exists():erp.symlink_to(Path(cam['image']).resolve())
        sheet=Image.new('RGB',(1536,1840),'white');draw=ImageDraw.Draw(sheet)
        draw.text((10,6),f'{index:02d}/56 TRAIN {name} | ERP + six 100-degree views | +Y down',fill='black')
        sheet.paste(image.resize((1536,768),Image.Resampling.LANCZOS),(0,24))
        views=[]
        for j,(face,rotation) in enumerate(FACES):
            im=Image.fromarray(perspective(rgb,rotation));dest=folder/(face+'.png');im.save(dest)
            x=(j%3)*512;y=792+(j//3)*524
            draw.text((x+5,y+3),face,fill='black');sheet.paste(im,(x,y+12))
            views.append({'name':face,'image':str(dest),'fov_degrees':100,'rotation_camera_from_face':rotation.tolist()})
        contact=out/'sheets'/f'{index:02d}_{name}.jpg';sheet.save(contact,quality=94)
        rows.append({'index':index,'name':name,'split':'train','rgb':ref(cam['image']),
                     'erp':str(erp),'perspective_views':views,'contact_sheet':str(contact),
                     'visual_analysis_status':'prepared_not_reviewed'})
    write(out/'manifest.json',{'stage':'A','status':'materials_prepared','observer_required':'Codex native image viewing; no external VLM',
                              'train_count':56,'val_count':len(val),'test_count':len(test),
                              'projection':'ERP pixel centers, horizontal circular border; six overlapping 100-degree pinhole views',
                              'official_categories':str(out/'official_categories.json'),'views':rows})
    print('Prepared',len(rows),'training RGB contact sheets')
    print([(c['id'],c['name'],c['instance_name_count']) for c in official['categories']])
    print('Training panoramas with official cubemap metadata:',sum(bool(v) for v in correspondence.values()))

if __name__=='__main__':main()
