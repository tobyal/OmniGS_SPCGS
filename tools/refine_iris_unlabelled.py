"""Refine official-reference seeds with target-image SAM2 object masks."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from annotate_iris_unlabelled import REPORT,OUT,NAMES,TARGETS,read,write,render,summarize,ROOT

def main():
    p=argparse.ArgumentParser();p.add_argument('--ids',nargs='+',default=TARGETS);a=p.parse_args()
    torch.set_num_threads(2)
    model=build_sam2('configs/sam2/sam2_hiera_l.yaml',str(ROOT/'weights/sam2_hiera_large.pt'),device='cuda',apply_postprocessing=False)
    model.eval();model.requires_grad_(False)
    generator=SAM2AutomaticMaskGenerator(model,points_per_side=20,points_per_batch=32,pred_iou_thresh=.85,stability_score_thresh=.92,crop_n_layers=0,min_mask_region_area=0)
    palette=np.zeros((256,3),dtype='uint8');palette[255]=[35,40,50]
    for c in read(REPORT/'taxonomy/classes.json')['classes']:palette[c['id']]=c['display_color']
    for cid in a.ids:
        for face in NAMES:
            dest=OUT/cid/'faces'/face
            if (dest/'refinement.json').exists():continue
            original=np.array(Image.open(dest/'labels.png'));labels=original.copy();rgb=np.array(Image.open(REPORT/'images'/cid/'faces'/face/'rgb.jpg'))
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):masks=generator.generate(rgb)
            accepted=[]
            for proposal in sorted(masks,key=lambda x:-x['area']):
                mask=proposal['segmentation'];known=original[mask];known=known[known!=255]
                if len(known)<max(100,proposal['area']*.2):continue
                ids,counts=np.unique(known,return_counts=True);k=int(np.argmax(counts));sem=int(ids[k]);purity=float(counts[k]/len(known))
                if purity<.9:continue
                region=mask&((original==255)|(original==sem));labels[region]=sem
                accepted.append({'category_id':sem,'mask_area':int(proposal['area']),'reference_support_fraction':float(len(known)/proposal['area']),'reference_label_purity':purity,'sam_mask_quality_score':float(proposal['predicted_iou'])})
            render(rgb,labels,dest,palette)
            write(dest/'refinement.json',{'source':'Target RGB SAM2 masks assigned by supported original-class consensus; not official ground truth','proposals':len(masks),'accepted':accepted,'coverage_before':float((original!=255).mean()),'coverage_after':float((labels!=255).mean()),'sam_masks_not_boxes':True})
            print('REFINE',cid,face,len(masks),len(accepted),round(float((labels!=255).mean()),4),flush=True)
        record=read(OUT/cid/'annotation.json');record['target_rgb_boundary_refinement']=True;write(OUT/cid/'annotation.json',record);summarize(cid,palette)
if __name__=='__main__':main()
