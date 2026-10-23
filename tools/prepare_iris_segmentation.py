"""Decode official COCO masks for exactly the 56 train cameras; no inference."""
import argparse,sys,shutil
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
import cv2
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spcgs.common import ROOT,cameras,read,write,sha
from spcgs.segmentation import *

def preview(root,name,labels,official):
    store=Store(root);folder=store.folder(name);rgb=np.array(Image.open(store.items[name]['rgb']).convert('RGB'));p=palette(store.config)
    for kind,arr in [('current',labels),('official',official)]:
        overlay=rgb.copy();known=arr!=255;overlay[known]=(.55*rgb[known]+.45*p[arr[known]]).astype('u1')
        Image.fromarray(overlay).resize((1024,512)).save(folder/(kind+'_preview.jpg'),quality=92)
    Image.fromarray(rgb).resize((512,256)).save(folder/'thumb.jpg',quality=88)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default=str(DEFAULT_RUN));a=parser.parse_args();root=Path(a.run)
    if (root/'manifest.json').exists():raise FileExistsError('Existing dataset: use a new run directory')
    root.mkdir(parents=True,exist_ok=True);cfg=read(CONFIG);write(root/'classes.json',cfg)
    source=ROOT.parent/'datasets/images_xphase_1_cubemap_semantics_1500';src=source/'_source/annotations_segmentation_cubemaps_xphase.json'
    d=read(src);mapping={i:c['id'] for c in cfg['classes'] for i in c['official_ids']};ann=defaultdict(list)
    for x in d['annotations']:ann[x['image_id']].append(x)
    ims={Path(x['file_name']).stem:x for x in d['images']};train=cameras('train');assert len(train)==56
    rows=[{'id':c['name'],'index':i,'rgb':c['image'],'rgb_sha256':sha(c['image'])} for i,c in enumerate(train,1)]
    write(root/'manifest.json',{'schema':'iris_erp_semantic_v1','classes':'iris7_v1','width':2048,'height':1024,'ignore_id':255,
        'source_annotations':str(src),'source_sha256':sha(src),'split':'train','images':rows})
    store=Store(root);audit=[]
    for row in rows:
        name=row['id'];folder=store.folder(name);folder.mkdir(parents=True,exist_ok=True)
        rawfaces={};mappedfaces={};faceaudit=[]
        for code,(face,rot) in FACE_CODES.items():
            im=ims.get(name+'_'+code);annotations=ann[im['id']] if im else []
            raw=np.full((1500,1500),255,np.uint8);overlap=np.zeros_like(raw,bool)
            objects=[]
            for obj in annotations:
                mask=decode_rle(obj['segmentation']);cl=obj['category_id']
                overlap |= mask&(raw!=255)&(raw!=cl)
                raw[mask]=cl
                objects.append({'annotation_id':obj['id'],'official_class':cl,'mapped_class':mapping.get(cl,255),'pixels':int(mask.sum()),'bbox':obj['bbox']})
            raw[overlap]=255
            mapped=np.full_like(raw,255)
            for old,new in mapping.items():mapped[raw==old]=new
            rawfaces[face]=raw;mappedfaces[face]=mapped
            faceaudit.append({'face':face,'official_image_id':im['id'] if im else None,'has_record':im is not None,
                'object_count':len(objects),'official_coverage':float((raw!=255).mean()),'mapped_coverage':float((mapped!=255).mean()),
                'unlabeled_pixels':int((raw==255).sum()),'unmapped_pixels':int(((raw!=255)&(mapped==255)).sum()),
                'overlap_conflict_pixels':int(overlap.sum()),'objects':objects})
        raw=from_faces(rawfaces);labels=from_faces(mappedfaces)
        Image.fromarray(raw).save(folder/'official_raw.png');Image.fromarray(labels).save(folder/'official_mapped.png')
        write(folder/'coverage.json',faceaudit)
        store.save(name,labels,(labels!=255).astype('u1'),None,actor='pipeline',action='official_import')
        s=store.state(name);s['issues']=['官方 CAD 投影可能遗漏实拍遮挡物；未知区域不是背景。','待检查细杆、桥架、灯具附近边界及遮挡。'];s['review_state']='issues';write(folder/'state.json',s)
        audit.append({'id':name,'index':row['index'],'faces':faceaudit,'erp_official_coverage':float((raw!=255).mean()),'erp_mapped_coverage':float((labels!=255).mean())})
        preview(root,name,labels,labels)
        print(row['index'],name,'mapped',round(float((labels!=255).mean()),3),flush=True)
    write(root/'coverage_audit.json',audit)
    # Verify all six original reference orientations against actual 2K train RGB.
    rgb=np.array(Image.open(rows[0]['rgb']).convert('RGB'));alignment=[]
    for code,(face,rot) in FACE_CODES.items():
        ref=source/'_source/reference_rgb'/f'{rows[0]["id"]}_{code}.png'
        reference=np.array(Image.open(ref).convert('RGB').resize((256,256)))
        projected=perspective(rgb,rot,256,90)
        corr=float(np.corrcoef(reference.mean(-1).ravel(),projected.mean(-1).ravel())[0,1])
        alignment.append({'face':face,'correlation':corr,'mae':float(np.abs(reference.astype(float)-projected).mean())})
        if corr<.93:raise ValueError('Official alignment failed '+face)
    write(root/'projection_alignment.json',alignment)
    print('Alignment',alignment)

if __name__=='__main__':main()
