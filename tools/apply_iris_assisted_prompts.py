"""Actual SAM2 masks for visually identified official-class objects."""
import argparse,collections
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from annotate_iris_unlabelled import ROOT,REPORT,OUT,read,write,render,summarize
from iris_assisted_visual_prompts import PROMPTS

def main():
    p=argparse.ArgumentParser();p.add_argument('--ids',nargs='+');a=p.parse_args()
    torch.set_num_threads(2)
    model=build_sam2('configs/sam2/sam2_hiera_l.yaml',str(ROOT/'weights/sam2_hiera_large.pt'),device='cuda',apply_postprocessing=False)
    model.eval();model.requires_grad_(False);predictor=SAM2ImagePredictor(model)
    palette=np.zeros((256,3),dtype='uint8');palette[255]=[35,40,50]
    for c in read(REPORT/'taxonomy/classes.json')['classes']:palette[c['id']]=c['display_color']
    grouped=collections.defaultdict(list)
    for prompt in PROMPTS:
        if not a.ids or prompt['camera'] in a.ids:grouped[prompt['camera'],prompt['face']].append(prompt)
    for (cid,face),prompts in grouped.items():
        dest=OUT/cid/'faces'/face;rgb=np.array(Image.open(REPORT/'images'/cid/'faces'/face/'rgb.jpg'));labels=np.array(Image.open(dest/'labels.png'));records=[]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            predictor.set_image(rgb)
            for q in prompts:
                pts=np.array(q['points']+list(q['negative']),dtype='f4')*1.5
                pl=np.array([1]*len(q['points'])+[0]*len(q['negative']),dtype='i4')
                masks,scores,_=predictor.predict(point_coords=pts,point_labels=pl,box=np.array(q['box'],dtype='f4')*1.5,multimask_output=True)
                k=int(np.argmax(scores));mask=masks[k].astype(bool);labels[mask]=q['category']
                records.append({**q,'pixels':int(mask.sum()),'sam_score':float(scores[k]),'human_confirmed':False})
        render(rgb,labels,dest,palette);write(dest/'visual_prompts.json',records)
        print('PROMPTS',cid,face,len(records),round(float((labels!=255).mean()),4),flush=True)
    for cid in sorted({c for c,f in grouped}):summarize(cid,palette)
if __name__=='__main__':main()
