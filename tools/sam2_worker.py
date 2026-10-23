"""SAM2 ISP on a real + two online-rendered panoramas. No proxy masks.

Panorama coordinates are cyclically shifted to place the random prompt away
from the horizontal seam; every output is unshifted to original ERP coordinates.
Validity is a basic sanity gate, never an accuracy claim.
"""
import sys,json,os,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch

def main(request_path):
    c=json.loads(Path(request_path).read_text());out=Path(c['output']);root=Path(c['root']);rng=np.random.default_rng(c['seed'])
    torch.set_num_threads(2)
    from sam2.build_sam import build_sam2_video_predictor,build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    start=time.time();imgs=[np.array(Image.open(p).convert('RGB')) for p in c['images']];H,W=imgs[0].shape[:2]
    assert all(im.shape==imgs[0].shape for im in imgs)
    x,y=int(rng.integers(W)),int(rng.integers(H));shift=W//2-x
    frames=out/'sequence';frames.mkdir(exist_ok=True)
    for i,im in enumerate(imgs):Image.fromarray(np.roll(im,shift,axis=1)).save(frames/f'{i:05d}.jpg',quality=95)
    cfg=c.get('sam_config','configs/sam2.1/sam2.1_hiera_t.yaml');ckpt=c.get('sam_checkpoint',str(root/'weights/sam2.1_hiera_tiny.pt'))
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        predictor=build_sam2_video_predictor(cfg,ckpt,device='cuda',apply_postprocessing=False)
        state=predictor.init_state(video_path=str(frames),offload_video_to_cpu=True)
        predictor.add_new_points_or_box(inference_state=state,frame_idx=0,obj_id=1,points=np.array([[W//2,y]],dtype='f4'),labels=np.array([1],dtype='i4'))
        masks=np.zeros((3,H,W),dtype='u1')
        for index,objects,logits in predictor.propagate_in_video(state):masks[index]=np.roll((logits[0,0]>0).cpu().numpy(),-shift,axis=1)
        del predictor,state
        model=build_sam2(cfg,ckpt,device='cuda',apply_postprocessing=False)
        generator=SAM2AutomaticMaskGenerator(model,points_per_side=8,points_per_batch=16,pred_iou_thresh=.7,stability_score_thresh=.85,crop_n_layers=0)
        proposals=generator.generate(np.roll(imgs[0],shift,axis=1))
    regions=np.full((H,W),-1,dtype='i4')
    for i,p in enumerate(sorted(proposals,key=lambda p:-p['area'])):
        regions[np.roll(p['segmentation'],-shift,axis=1)]=i
    hs,ws=c['height'],c['width'];small=[]
    for i,mask in enumerate(masks):
        small.append(np.array(Image.fromarray(mask).resize((ws,hs),Image.Resampling.NEAREST)))
        overlay=imgs[i].copy();overlay[mask>0]=(overlay[mask>0]*.4+np.array([240,80,80])*.6).astype('u1')
        im=Image.fromarray(overlay);d=ImageDraw.Draw(im)
        if i==0:d.ellipse((x-4,y-4,x+4,y+4),fill='yellow')
        im.save(out/f'mask_{i}.png');Image.fromarray(mask*255).save(out/f'mask_{i}_binary.png')
    np.stack(small).astype('u1').tofile(out/'masks.u8')
    reg=np.array(Image.fromarray(regions,mode='I').resize((ws,hs),Image.Resampling.NEAREST),dtype='i4');reg.tofile(out/'regions.i32')
    areas=masks.mean((1,2));ratios=areas[1:]/max(float(areas[0]),1e-10)
    valid=[bool(.001<areas[0]<.95 and .001<a<.95 and .1<r<10 and int(s.sum())>=9) for a,r,s in zip(areas[1:],ratios,small[1:])]
    result={'status':'completed','backend':'SAM2.1 Hiera-tiny','checkpoint':ckpt,'sam_config':cfg,'prompt_xy':[x,y],
      'rolled_prompt_xy':[W//2,y],'horizontal_roll':shift,'seed':c['seed'],'valid':valid,'area_fractions':areas.tolist(),
      'area_ratios':ratios.tolist(),'automatic_masks':len(proposals),'seconds':time.time()-start,
      'validity_note':'Area and nonempty sanity checks only; object correspondence requires visual inspection. Full ERP distortion remains; no accuracy claims.',
      'masks':str(out/'masks.u8'),'regions':str(out/'regions.i32')}
    (out/'outputs.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main(sys.argv[1])
