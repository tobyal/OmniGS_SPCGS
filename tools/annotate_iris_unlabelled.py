"""Frozen SAM2 transfer from two official neighbours; uncertain pixels stay ignore.

No semantic training and no original data writes. Output is assisted annotation.
"""
import argparse,gc,json,shutil,sys,tempfile,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spcgs.segmentation import from_faces
from sam2.build_sam import build_sam2_video_predictor

REPORT=ROOT/'runs/iris_fullclass_gpt_annotation_20261025'
OUT=REPORT/'assisted_annotations'
NAMES=['front','right','back','left','up','down']
TARGETS=['1-32','1-33','1-43','1-45','1-46','1-56']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def render(rgb,labels,dest,palette):
    dest.mkdir(parents=True,exist_ok=True);Image.fromarray(labels).save(dest/'labels.png')
    colored=palette[labels];Image.fromarray(colored).save(dest/'semantic.jpg',quality=93,subsampling=0)
    overlay=(.56*rgb+.44*colored).astype('uint8');overlay[labels==255]=(rgb[labels==255]*.3).astype('uint8')
    Image.fromarray(overlay).save(dest/'overlay.jpg',quality=93,subsampling=0)
def board(cid,kind):
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',23)
    canvas=Image.new('RGB',(1800,1280),'white');draw=ImageDraw.Draw(canvas)
    for i,n in enumerate(NAMES):
        x=(i%3)*600;y=(i//3)*640
        draw.text((x+8,y+5),f'{cid} {n} | {kind}',fill='black',font=font)
        canvas.paste(Image.open(OUT/cid/'faces'/n/f'{kind}.jpg').resize((600,600)),(x,y+40))
    canvas.save(OUT/cid/f'{kind}_board.jpg',quality=93)
def transfer(model,reference,target,face,work):
    for i,cid in enumerate([reference,target]):shutil.copy2(REPORT/'images'/cid/'faces'/face/'rgb.jpg',work/f'{i:05d}.jpg')
    src=np.asarray(Image.open(REPORT/'images'/reference/'faces'/face/'labels.png'))
    # Class 13 has no visually certified example in the official reference set.
    ids=[int(v) for v in np.unique(src) if v not in (13,255) and (src==v).sum()>=80]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        state=model.init_state(video_path=str(work),offload_video_to_cpu=True)
        for cid in ids:model.add_new_mask(state,frame_idx=0,obj_id=cid,mask=src==cid)
        result=None
        for index,objects,logits in model.propagate_in_video(state,start_frame_idx=0,max_frame_num_to_track=2):
            if index==1:
                values=logits[:,0].float();top,ind=values.topk(min(2,len(objects)),dim=0)
                labels=np.asarray(objects,dtype='uint8')[ind[0].cpu().numpy()]
                valid=(top[0]>0)&((top[0]-top[1])>.5)
                labels[~valid.cpu().numpy()]=255;result=labels
        assert result is not None
        model.reset_state(state);del state
    return result
def summarize(cid,palette):
    faces={n:np.asarray(Image.open(OUT/cid/'faces'/n/'labels.png')) for n in NAMES}
    erp=from_faces(faces)
    render(np.asarray(Image.open(REPORT/'images'/cid/'erp/rgb.png')),erp,OUT/cid/'erp',palette)
    board(cid,'overlay');board(cid,'semantic')
    p=OUT/cid/'annotation.json';record=read(p)
    record.update(coverage=float((erp!=255).mean()),class_pixels={str(int(c)):int((erp==c).sum()) for c in np.unique(erp)})
    write(p,record);print('SUMMARY',cid,record['coverage'],flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--ids',nargs='+',default=TARGETS);p.add_argument('--render-only',action='store_true');a=p.parse_args()
    palette=np.zeros((256,3),dtype='uint8');palette[255]=[35,40,50]
    for c in read(REPORT/'taxonomy/classes.json')['classes']:palette[c['id']]=c['display_color']
    if a.render_only:
        for cid in a.ids:summarize(cid,palette)
        return
    torch.set_num_threads(2)
    model=build_sam2_video_predictor('configs/sam2/sam2_hiera_l.yaml',str(ROOT/'weights/sam2_hiera_large.pt'),device='cuda',apply_postprocessing=False)
    model.eval();model.requires_grad_(False)
    for cid in a.ids:
        meta=read(REPORT/'images'/cid/'metadata.json');assert not meta['has_official_semantics']
        refs=[r['camera_id'] for r in read(REPORT/'images'/cid/'tasks.json')['neighbor_references']]
        record={'camera_id':cid,'image_id':meta['image_id'],'split':meta['split'],'has_official_annotation':False,
          'source':'AI-assisted annotation based on official 16-class references','human_confirmed':False,
          'official_ids_unchanged':True,'reference_cameras':refs,'model':'frozen SAM2 Hiera-L',
          'method':'Two separate RGB-guided propagations from original official face masks; retain positive non-ambiguous pixels only when both reference predictions agree. Further visual review and explicit object prompts follow.',
          'excluded_class_ids':[13],'exclusion_reason':'No visually certified official FIRE-FIGHTING example; keep unknown rather than fabricate.',
          'review_status':'pending_visual_review','faces':[]}
        for face in NAMES:
            dest=OUT/cid/'faces'/face
            if (dest/'transfer.json').exists():record['faces'].append(read(dest/'transfer.json'));continue
            start=time.time();pred=[]
            for reference in refs:
                with tempfile.TemporaryDirectory(prefix='iris_sam_',dir='/tmp') as d:pred.append(transfer(model,reference,cid,face,Path(d)))
                gc.collect()
            labels=pred[0].copy();labels[(pred[0]!=pred[1])|(pred[0]==255)]=255
            render(np.asarray(Image.open(REPORT/'images'/cid/'faces'/face/'rgb.jpg')),labels,dest,palette)
            facts={'face':face,'references':refs,'coverage':float((labels!=255).mean()),'reference_disagreement_fraction':float((pred[0]!=pred[1]).mean()),'seconds':round(time.time()-start,2)}
            write(dest/'transfer.json',facts);record['faces'].append(facts)
            print('FACE',cid,face,facts,flush=True)
        write(OUT/cid/'annotation.json',record);summarize(cid,palette)
if __name__=='__main__':main()
