"""Apply recorded second-pass visual decisions, retaining every prior revision."""
import argparse, sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spcgs.segmentation import *
from prepare_iris_segmentation import preview

def connected_tripod(mask):
    # Sample discrete ERP mask with nearest neighbors into the down face.
    size=1024
    t=((np.arange(size)+.5)/size*2-1)*np.tan(np.deg2rad(50))
    x,y=np.meshgrid(t,t)
    rays=np.stack([x,y,np.ones_like(x)],-1)@dict(FACES)['down'].T
    rays/=np.linalg.norm(rays,axis=-1,keepdims=True)
    u=np.rint((np.arctan2(rays[...,0],rays[...,2])/(2*np.pi)+.5)*2048-.5).astype(int)%2048
    v=np.clip(np.rint((np.arcsin(rays[...,1])/np.pi+.5)*1024-.5).astype(int),0,1023)
    _,components=cv2.connectedComponents(mask[v,u].astype('u1'),connectivity=8)
    xy,_=point_to_face(1024,1022,'down');xx,yy=np.rint(xy).astype(int)
    anchor=components[yy,xx]
    if anchor==0:return np.zeros_like(mask,bool)
    # Intersect with the actual original SAM mask; reprojection cannot add pixels.
    return mask & mask_to_erp(components==anchor,'down')

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default=str(DEFAULT_RUN));a=p.parse_args();s=Store(a.run)
    decision=read(ROOT/'docs/iris_segmentation_final_decisions.json');first=read(ROOT/'docs/iris_segmentation_corrections.json')
    for name,item in s.items.items():
        f=s.folder(name);marker=f/'finalization.json'
        if marker.exists():continue
        before,source,state=s.arrays(name)
        if (source==3).any() or state['review_state']=='approved':raise RuntimeError('Human work present; refusing rebuild')
        i=str(item['index']);refine=read(f/'refinement.json')
        rejected=set(first['rejected_prompt_ids'].get(i,[])+decision['additional_rejected_prompt_ids'].get(i,[]))
        records=[r for r in read(f/'sam_candidates.json')['prompts'] if r['prompt_id'] not in rejected]
        records += [r for j,r in enumerate(refine['extra_prompts']) if j not in decision['rejected_extra_prompt_ids'].get(i,[])]
        official=np.array(Image.open(f/'official_mapped.png'));labels=official.copy();src=(labels!=255).astype('u1')
        proposal=np.full_like(labels,255);conflict=np.zeros_like(labels,bool)
        for r in records:
            mask=np.array(Image.open(f/r['mask']))>0;cl=r['class_id']
            conflict|=mask&(proposal!=255)&(proposal!=cl);proposal[mask]=cl
        proposal[conflict]=255;raw=np.array(Image.open(f/'official_raw.png'))
        fill=(labels==255)&(proposal!=255)&((raw==255)|(item['index']==56))
        labels[fill]=proposal[fill];src[fill]=2
        tripod=np.zeros_like(labels,bool)
        accepted=item['index'] in decision['tripod_box_accepted_indices']
        if accepted:
            tripod=connected_tripod(np.array(Image.open(f/'sam_masks/tripod_box.png'))>0)
            accepted=bool(tripod.any())
            if accepted:Image.fromarray(tripod.astype('u1')*255).save(f/'sam_masks/tripod_applied.png')
            labels[tripod]=255;src[tripod]=4
        meta={'observer':decision['observer'],'rejected_original_prompts':sorted(rejected),
              'rejected_extra_prompts':decision['rejected_extra_prompt_ids'].get(i,[]),
              'tripod_applied':accepted,'tripod_pixels':int(tripod.sum()),
              'changed_pixels':int((labels!=before).sum()),'human_approved':False}
        s.save(name,labels,src,state['revision'],actor='Codex visual audit',action='visual_audit',details=meta)
        st=s.state(name);st['review_state']='issues'
        if i in decision['reasons']:st['issues'].append(decision['reasons'][i])
        st['issues'].append('三脚架掩码已作候选修正，边界仍需人工检查。' if accepted else '三脚架 SAM 框候选未被采纳；下方相机遮挡区域需人工修正。')
        write(f/'state.json',st);write(marker,meta);preview(s.root,name,labels,official)
        print('FINAL',item['index'],'ignore',round(st['ignore_fraction'],4),flush=True)
    rows=list(s.items.items());out=s.root/'qa'
    for start in range(0,len(rows),4):
        im=Image.new('RGB',(1024,2184),'#131a24');d=ImageDraw.Draw(im)
        for j,(name,item) in enumerate(rows[start:start+4]):
            st=s.state(name);d.text((5,j*546+5),f'{item["index"]} {name} ignore {st["ignore_fraction"]:.1%}',fill='white')
            im.paste(Image.open(s.folder(name)/'current_preview.jpg'),(0,j*546+26))
        im.save(out/f'final_{start+1:02d}_{start+4:02d}.jpg',quality=94)
    for start in [0,28]:
        im=Image.new('RGB',(1792,1120),'#131a24');d=ImageDraw.Draw(im)
        for j,(name,item) in enumerate(rows[start:start+28]):
            rgb=np.array(Image.open(item['rgb']).convert('RGB'));f=s.folder(name)/'sam_masks/tripod_applied.png'
            if f.exists():
                mask=np.array(Image.open(f))>0;rgb[mask]=(.3*rgb[mask]+.7*np.array([255,0,255])).astype('u1')
            down=perspective(rgb,dict(FACES)['down'],256,100);x=j%7*256;y=j//7*280
            im.paste(Image.fromarray(down),(x,y+24));d.text((x+5,y+5),str(item['index'])+(' applied' if f.exists() else ' pending'),fill='white')
        im.save(out/f'tripods_final_{start+1}_{start+28}.jpg',quality=95)

if __name__=='__main__':main()
