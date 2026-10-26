"""Audit the current 20+6 split directly from original IRIS COCO labels."""
from pathlib import Path
import collections,json,sys,math
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont,ImageOps
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spcgs.common import ROOT,read,write,ref
from spcgs.segmentation import decode_rle
from spcgs.clip_dense import FACES,perspective
from prepare_iris_fullclass_gpt import project_faces,EXPLAIN,COLORS

OUT=ROOT/'runs/iris26_semantic_consolidation_20261025'
DATA=ROOT.parent/'datasets/iris_official20_6_no_tripod_2k_20261025'
COCO=ROOT.parent/'datasets/images_xphase_1_cubemap_semantics_1500/_source/annotations_segmentation_cubemaps_xphase.json'
CODES=dict(zip('FRBLUD',[n for n,_ in FACES]))
IDS=sorted(EXPLAIN)

def save(p,a):
    p.parent.mkdir(parents=True,exist_ok=True)
    im=Image.fromarray(a) if isinstance(a,np.ndarray) else a
    im.save(p,**({'quality':94} if p.suffix=='.jpg' else {}))

def main():
    OUT.mkdir(exist_ok=True);rows=read(ROOT/'runs/iris_tripod_cleanup_20261025/manifest.json')['images']
    assert len(rows)==26 and sum(r['split']=='train' for r in rows)==20
    d=read(COCO);ii={Path(i['file_name']).stem:i for i in d['images']};anns=collections.defaultdict(list)
    for a in d['annotations']:anns[a['image_id']].append(a)
    palette=np.zeros((256,3),'u1');palette[255]=[35,40,50]
    for k,col in zip(IDS,COLORS):palette[k]=col
    weights=np.broadcast_to(np.cos((.5-(np.arange(1024)+.5)/1024)*np.pi)[:,None],(1024,2048))
    stats=[];objects=[]
    for r in rows:
        cid=r['camera_id'];name=r['image_id'];folder=OUT/'source'/cid
        rgb=np.array(Image.open(DATA/'images'/(name+'.png')).convert('RGB'))
        valid=np.array(Image.open(DATA/'valid_masks'/(name+'.png')))>0
        faces={};annmaps={};conflict_faces={}
        for code,face in CODES.items():
            im=ii[name+'_'+code];assert im['width']==im['height']==1500
            lab=np.full((1500,1500),255,'u1');annmap=np.zeros((1500,1500),'i4');count=np.zeros_like(lab)
            frgb=perspective(rgb,dict(FACES)[face],1500,90)
            for a in anns[im['id']]:
                m=decode_rle(a['segmentation']);assert int(m.sum())==int(a['area'])
                lab[m]=a['category_id'];annmap[m]=a['id'];count+=m
                if a['category_id'] in [9,13]:
                    bbox=list(map(int,a['bbox']));x,y,w,h=bbox
                    pad=max(25,int(max(w,h)*.12));box=[max(0,x-pad),max(0,y-pad),min(1500,x+w+pad),min(1500,y+h+pad)]
                    if box[2]-box[0]<120:box[0]=max(0,x-60);box[2]=min(1500,x+w+60)
                    if box[3]-box[1]<120:box[1]=max(0,y-60);box[3]=min(1500,y+h+60)
                    ov=frgb.copy();ov[m]=(.5*frgb[m]+.5*np.array([255,0,170])).astype('u1')
                    panel=Image.new('RGB',(400,228),'#eef2f5');dr=ImageDraw.Draw(panel)
                    for j,arr in enumerate([frgb,ov]):
                        crop=Image.fromarray(arr).crop(box);crop.thumbnail((196,196));panel.paste(crop,(j*200+(200-crop.width)//2,28+(196-crop.height)//2))
                    text=f"{cid} {code} ann {a['id']} | {a['area']}px"
                    dr.text((5,5),text,font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13),fill='#153448')
                    save(OUT/'review/objects'/f"{a['id']}.jpg",panel)
                    objects.append({'camera_id':cid,'image_id':name,'face':face,'face_code':code,'annotation_id':a['id'],'official_id':a['category_id'],'pixels_face':int(a['area']),'bbox_xywh':bbox,'crop_box':box,'preview':f"objects/{a['id']}.jpg"})
            conflict=count>1;lab[conflict]=255;annmap[conflict]=0
            faces[face]=lab;annmaps[face]=annmap;conflict_faces[face]=conflict.astype('u1')
            save(folder/'faces'/face/'labels.png',lab)
            np.save(folder/'faces'/face/'annotation_ids.npy',annmap)
            save(folder/'faces'/face/'rgb.jpg',frgb)
        label=project_faces(faces);save(folder/'labels_original.png',label)
        # Same validated owner maps, but preserve int32 annotation IDs.
        import prepare_iris_fullclass_gpt as proj
        ann_erp=np.zeros((1024,2048),'i4')
        for face,pick,yy,xx in proj.PROJECT_MAPS:ann_erp[pick]=annmaps[face][yy,xx]
        np.save(folder/'annotation_ids.npy',ann_erp)
        effective=label.copy();effective[~valid]=255;save(folder/'labels_effective.png',effective)
        save(OUT/'review/original_semantics'/(cid+'.png'),palette[effective])
        save(OUT/'review/rgb'/(cid+'.jpg'),rgb)
        for cl in IDS+[255]:
            m=effective==cl
            stats.append({'camera_id':cid,'split':r['split'],'class_id':cl,'pixels':int(m.sum()),'erp_fraction':float(m.mean()),'sphere_fraction':float(weights[m].sum()/weights.sum())})
        print('AUDITED',cid,flush=True)
    summary=[]
    for cl in IDS+[255]:
        arr=[s for s in stats if s['class_id']==cl]
        summary.append({'id':cl,'zh':EXPLAIN[cl][0] if cl!=255 else 'ignore','erp_percent':sum(s['erp_fraction'] for s in arr)/26*100,'sphere_percent':sum(s['sphere_fraction'] for s in arr)/26*100,'views':sum(s['pixels']>0 for s in arr),'train_views':sum(s['pixels']>0 and s['split']=='train' for s in arr),'test_views':sum(s['pixels']>0 and s['split']=='test' for s in arr),'pixels':sum(s['pixels'] for s in arr)})
    write(OUT/'source_audit.json',{'source_coco':ref(COCO),'source_split':ref(ROOT/'runs/iris_camera_analysis_20261025/official_overlap_split_20261025/selected_split.json'),'rows':rows,'classes':summary,'per_image':stats,'misc_objects':objects,'artifact_masks_applied':True,'semantics_source':'official_COCO_only; no SAM candidate semantics imported'})
    ordered=sorted([o for o in objects if o['official_id']==9],key=lambda x:-x['pixels_face'])
    # Cards remain readable; sort by size and retain all, including unresolved tiny ones.
    for start in range(0,len(ordered),24):
        chunk=ordered[start:start+24];sheet=Image.new('RGB',(1600,6*228),'#f1f4f7')
        for j,o in enumerate(chunk):sheet.paste(Image.open(OUT/'review'/o['preview']),((j%4)*400,(j//4)*228))
        save(OUT/'review'/f'misc_sheet_{start//24+1:02d}.jpg',sheet)
    write(OUT/'review/misc_index.json',ordered)
    write(OUT/'review/original_statistics.json',summary)
    print(json.dumps({'classes':summary,'misc_count':len(ordered),'misc_over_2000':sum(o['pixels_face']>2000 for o in ordered),'misc_over_2000_area_share':sum(o['pixels_face'] for o in ordered if o['pixels_face']>2000)/sum(o['pixels_face'] for o in ordered)},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
