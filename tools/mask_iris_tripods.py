"""Predict capture-rig masks on down projections, for inspection before compositing."""
from pathlib import Path
import sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import cv2
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spcgs.common import ROOT,read,write,ref
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

RUN=ROOT/'runs/iris_tripod_cleanup_20261025'

def main():
    torch.set_num_threads(2)
    model=build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(ROOT/'weights/sam2.1_hiera_tiny.pt'),device='cuda:2',apply_postprocessing=False)
    predictor=SAM2ImagePredictor(model); manifest=read(RUN/'manifest.json')
    (RUN/'masks').mkdir(exist_ok=True); rows=[]
    with torch.inference_mode():
        for row in manifest['images']:
            cid=row['camera_id'];rgb=np.array(Image.open(RUN/'inputs'/(cid+'.png')).convert('RGB'))
            predictor.set_image(rgb)
            center_gray=rgb[492:532,492:532].mean(-1)
            cy,cx=np.unravel_index(center_gray.argmin(),center_gray.shape);cx+=492;cy+=492
            radius=105 if row['height_band_relative_to_core']=='upper' else 175
            bbox=[512-radius,512-radius,512+radius,512+radius]
            masks,scores,_=predictor.predict(point_coords=np.array([[cx,cy]],dtype='f4'),point_labels=np.array([1]),
                box=np.array(bbox,dtype='f4'),multimask_output=True)
            options=[]
            for k,m in enumerate(masks):
                n,lab,stats,centers=cv2.connectedComponentsWithStats(m.astype('u1'),connectivity=8)
                center=lab[cy,cx]
                if center==0:continue
                component=lab==center; ys,xs=np.nonzero(component)
                area=int(component.sum()); box=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]
                if 400<area<22000 and min(box[:2])>300 and max(box[2:])<720:
                    options.append((float(scores[k]),component,k,area,box))
            if not options:raise ValueError('No plausible center rig mask: '+cid)
            score,mask,k,area,box=max(options,key=lambda x:x[0])
            Image.fromarray(mask.astype('u1')*255).save(RUN/'masks'/(cid+'_sam.png'))
            rows.append({'camera_id':cid,'source':ref(RUN/'inputs'/(cid+'.png')),'score':score,'mask_choice':k,
                         'raw_pixels':area,'raw_bbox':box,'raw_mask':ref(RUN/'masks'/(cid+'_sam.png')),
                         'prompt':{'point':[int(cx),int(cy)],'box':bbox},'visual_inspection_pending':True})
            print(cid,score,area,box,flush=True)
    write(RUN/'mask_predictions.json',{'model':'SAM2.1 Hiera tiny','source':'capture-rig artifact mask, NOT official semantics','images':rows})
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    for start in range(0,len(rows),13):
        chunk=rows[start:start+13];sheet=Image.new('RGB',(5*320,((len(chunk)+4)//5)*346),'#f3f5f7');draw=ImageDraw.Draw(sheet)
        for i,row in enumerate(chunk):
            cid=row['camera_id'];rgb=np.array(Image.open(RUN/'inputs'/(cid+'.png')));mask=np.array(Image.open(row['raw_mask']['path']))>0
            rgb[mask]=(.4*rgb[mask]+.6*np.array([255,0,120])).astype('u1')
            # Zoom central rig for reliable review, preserving all surrounding real structures.
            img=Image.fromarray(rgb).crop((300,300,724,724)).resize((320,320))
            x=i%5*320;y=i//5*346;sheet.paste(img,(x,y+25));draw.text((x+5,y+3),cid,font=font,fill='#173343')
        sheet.save(RUN/f'rig_masks_{start+1}_{start+len(chunk)}.jpg',quality=95)

if __name__=='__main__':main()
