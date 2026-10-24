"""Fixed SAM2 regions from captured RGB, used for evaluation only.

Masks never enter training. Uniform RNG prompts and nearest 3D cameras are fixed
before final models exist. Basic validity and manual correspondence review differ.
"""
import sys,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from spcgs.common import ROOT,read,write,ref
from tools.sam2_worker import client,SamWorker

def main(run):
    run=Path(run).resolve();cfg=read(run/'config.json');protocol=read(run/'00_protocol/manifest.json')
    folder=run/'06_comparison/evaluation_correspondences';folder.mkdir(parents=True,exist_ok=True)
    training_request=run/'05_spc/c2/request.json';socket=read(training_request).get('sam_server_socket') if training_request.exists() else None
    if socket and Path(socket).exists():cfg['sam_server_socket']=socket;worker=None
    else:worker=SamWorker({**cfg,'root':str(ROOT)})
    heldout=[*[{**v,'split':'unseen_train29'} for v in protocol['unseen_train29']],*protocol['validation'],*protocol['test']]
    entries=[];font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    for index in [1,14,27,48,56]:
        anchor=next(v for v in protocol['train'] if v['original_train_index']==index)
        neighbors=sorted(heldout,key=lambda v:np.linalg.norm(np.array(v['center'])-anchor['center']))[:2]
        for seed in [2,16,42]:
            event=folder/f'view{index:02d}_seed{seed}';event.mkdir(exist_ok=True);cams=[anchor,*neighbors]
            request={**cfg,'root':str(ROOT),'height':cfg['semantic_height'],'width':cfg['semantic_width'],
                'seed':seed,'output':str(event),'cameras':cams,'images':[],
                'scope':'Fixed evaluation correspondences from real RGB only; never used in optimization'}
            for i,cam in enumerate(cams):
                image=event/f'{i}.png'
                if not image.exists():Image.open(cam['image']).convert('RGB').resize((cfg['width'],cfg['height']),Image.Resampling.BOX).save(image)
                request['images'].append(str(image))
            write(event/'request.json',request)
            if not (event/'outputs.json').exists():
                result=client(request) if worker is None else worker.run(request)
            else:result=read(event/'outputs.json')
            canvas=Image.new('RGB',(2304,836),'white');draw=ImageDraw.Draw(canvas)
            for i,cam in enumerate(cams):
                draw.text((i*768+8,4),f'{cam["split"]}: {cam["name"]}',fill='black',font=font)
                canvas.paste(Image.open(request['images'][i]).resize((768,384)),(i*768,34))
                draw.text((i*768+8,422),f'Random prompt {seed}; area={result["area_fractions"][i]:.4f}',fill='black',font=font)
                canvas.paste(Image.open(event/f'mask_{i}.png').resize((768,384)),(i*768,452))
            canvas.save(event/'correspondence.jpg',quality=94)
            entries.append({'event':str(event),'anchor_index':index,'seed':seed,'basic_valid':result['valid'],
                'heldout_ids':[v['name'] for v in neighbors],'heldout_splits':[v['split'] for v in neighbors],
                'distances_3d':[float(np.linalg.norm(np.array(v['center'])-anchor['center'])) for v in neighbors]})
            print(event.name,result['valid'],flush=True)
    write(folder/'manifest.json',{'scope':'Evaluation only, fixed before final models. Actual held-out RGB for SAM2, no human semantic GT.',
        'selection':'5 spatial anchors; two nearest held-out 3D camera centers; fixed uniform random seeds 2,16,42.',
        'sam_checkpoint':ref(cfg['sam_checkpoint']),'events':entries})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();main(a.run)
