"""Execute native-vision-authored point prompts with real SAM2 masks."""
import argparse,sys,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spcgs.segmentation import *
from prepare_iris_segmentation import preview

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default=str(DEFAULT_RUN));p.add_argument('--indices',type=int,nargs='*');a=p.parse_args()
    store=Store(a.run);plan=read(ROOT/'docs/iris_segmentation_prompts.json');sam=SamSegmenter();start=time.time()
    for name,item in store.items.items():
        i=item['index'];folder=store.folder(name);done=folder/'sam_candidates.json'
        if a.indices and i not in a.indices:continue
        if done.exists():print('Cached',i,flush=True);continue
        labels,src,state=store.arrays(name)
        if state['review_state']=='approved':raise ValueError('Never replace a human-approved image')
        if sha(item['rgb'])!=item['rgb_sha256']:raise ValueError('RGB changed')
        official=np.array(Image.open(folder/'official_mapped.png'));proposal=np.full_like(labels,255);conflict=np.zeros_like(labels,bool)
        prompts=plan['images'][str(i)];records=[];(folder/'sam_masks').mkdir(exist_ok=True)
        # Group by face to reuse image embeddings. IDs retain authored order.
        jobs=sorted(enumerate(prompts),key=lambda row:best_face(row[1][1],row[1][2]))
        for j,(cls,x,y) in jobs:
            mask,info=sam.predict(item['rgb'],[[x,y]])
            Image.fromarray(mask.astype('u1')*255).save(folder/'sam_masks'/f'{j:02d}.png')
            conflict|=mask&(proposal!=255)&(proposal!=cls)
            proposal[mask]=cls
            records.append({'prompt_id':j,'class_id':cls,'mask':f'sam_masks/{j:02d}.png',**info})
        proposal[conflict]=255
        # Missing *officially unlabeled* regions only, not unmappable classes.
        raw=np.array(Image.open(folder/'official_raw.png'))
        fill=(labels==255)&(raw==255)&(proposal!=255)
        labels[fill]=proposal[fill];src[fill]=2
        # Capture tripod is visibly present in all 56 nadirs. Save its real SAM
        # mask for visual checking; corrections are applied only after inspection.
        artifact,info=sam.predict(item['rgb'],[[1024,1022],[1024,850]],[1,0],face='down')
        Image.fromarray(artifact.astype('u1')*255).save(folder/'sam_masks'/'tripod.png')
        s=store.save(name,labels,src,state['revision'],actor='SAM2 + Codex prompts',action='candidate_completion',details={'prompts_sha256':sha(ROOT/'docs/iris_segmentation_prompts.json')})
        write(done,{'observer':plan['observer'],'rgb_sha256':item['rgb_sha256'],'model':'SAM2.1 Hiera tiny','checkpoint_sha256':sha(ROOT/'weights/sam2.1_hiera_tiny.pt'),
            'prompts':sorted(records,key=lambda x:x['prompt_id']),'tripod_candidate':info,'conflict_pixels':int(conflict.sum()),'added_pixels':int(fill.sum()),'human_approved':False})
        preview(store.root,name,labels,official)
        print('DONE',i,name,'added',int(fill.sum()),'ignore',round(s['ignore_fraction'],3),'elapsed',round(time.time()-start,1),flush=True)

if __name__=='__main__':main()
