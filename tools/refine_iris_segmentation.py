"""Apply recorded visual decisions, then generate additional SAM candidates.

Rebuilds only untouched machine candidates. Refuses any human edit/approval.
The official source and all prior revisions remain immutable.
"""
import argparse,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spcgs.segmentation import *
from prepare_iris_segmentation import preview

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default=str(DEFAULT_RUN));a=p.parse_args();s=Store(a.run)
    decisions=read(ROOT/'docs/iris_segmentation_corrections.json');sam=SamSegmenter();out=s.root/'qa';out.mkdir(exist_ok=True)
    for name,item in s.items.items():
        f=s.folder(name);marker=f/'refinement.json'
        if marker.exists():continue
        before,source,state=s.arrays(name)
        if (source==3).any() or state['review_state']=='approved':raise RuntimeError('Human work present; refusing rebuild')
        i=str(item['index']);records=read(f/'sam_candidates.json')['prompts'];official=np.array(Image.open(f/'official_mapped.png'))
        labels=official.copy();src=(labels!=255).astype('u1');proposal=np.full_like(labels,255);conflict=np.zeros_like(labels,bool)
        rejected=decisions['rejected_prompt_ids'].get(i,[])
        for record in records:
            if record['prompt_id'] in rejected:continue
            mask=np.array(Image.open(f/record['mask']))>0;cl=record['class_id']
            conflict|=mask&(proposal!=255)&(proposal!=cl);proposal[mask]=cl
        extra=[]
        for j,(cl,x,y) in enumerate(decisions['extra_prompts'].get(i,[])):
            mask,info=sam.predict(item['rgb'],[[x,y]])
            dest=f'sam_masks/extra_{j:02d}.png';Image.fromarray(mask.astype('u1')*255).save(f/dest)
            conflict|=mask&(proposal!=255)&(proposal!=cl);proposal[mask]=cl;extra.append({'class_id':cl,'mask':dest,**info})
        proposal[conflict]=255
        # Native inspection explicitly identified the missing cabinet in 56;
        # its new masks can label previously unmappable official regions.
        raw=np.array(Image.open(f/'official_raw.png'))
        fill=(labels==255)&(proposal!=255)&((raw==255)|(item['index']==56))
        labels[fill]=proposal[fill];src[fill]=2
        tripod,info=sam.predict(item['rgb'],[[1024,1022]],face='down',box=np.array([320,320,720,720],dtype='f4'))
        Image.fromarray(tripod.astype('u1')*255).save(f/'sam_masks/tripod_box.png')
        # Keep tripod correction as a candidate until the down-face QA mosaic
        # has actually been inspected. A separate finalization applies decisions.
        s.save(name,labels,src,state['revision'],actor='Codex visual refinement',action='candidate_refinement',details={'rejected_prompts':rejected,'extra_prompts':extra})
        st=s.state(name);st['issues']=[decisions['known_residuals']]
        if any(c['has_record'] for c in read(f/'coverage.json')):st['issues'].append('官方 CAD 投影边界及实拍遮挡仍需检查。')
        if i in decisions['reasons']:st['issues'].append('已撤回错误候选：'+decisions['reasons'][i])
        st['review_state']='issues';write(f/'state.json',st)
        write(marker,{'observer':decisions['observer'],'rejected_prompt_ids':rejected,'extra_prompts':extra,'tripod':info,'changed_pixels':int((labels!=before).sum()),'human_approved':False})
        preview(s.root,name,labels,official);print('REFINED',i,'ignore',round(st['ignore_fraction'],3),flush=True)
    rows=list(s.items.items())
    for start in range(0,56,4):
        im=Image.new('RGB',(1024,2184),'#131a24');d=ImageDraw.Draw(im)
        for j,(name,item) in enumerate(rows[start:start+4]):
            st=s.state(name);d.text((5,j*546+5),f'{item["index"]} {name} ignore {st["ignore_fraction"]:.1%}',fill='white');im.paste(Image.open(s.folder(name)/'current_preview.jpg'),(0,j*546+26))
        im.save(out/f'refined_{start+1:02d}_{start+4:02d}.jpg',quality=94)
    for start in [0,28]:
        im=Image.new('RGB',(1792,1120),'#131a24');d=ImageDraw.Draw(im)
        for j,(name,item) in enumerate(rows[start:start+28]):
            rgb=np.array(Image.open(item['rgb']).convert('RGB'));mask=np.array(Image.open(s.folder(name)/'sam_masks/tripod_box.png'))>0
            rgb[mask]=(.3*rgb[mask]+.7*np.array([255,0,255])).astype('u1');down=perspective(rgb,dict(FACES)['down'],256,100)
            x=j%7*256;y=j//7*280;im.paste(Image.fromarray(down),(x,y+24));d.text((x+5,y+5),str(item['index']),fill='white')
        im.save(out/f'tripods_{start+1}_{start+28}.jpg',quality=95)

if __name__=='__main__':main()
