"""Novel pose + arbitrary text queries against a saved semantic Gaussian model."""
import argparse,os
from pathlib import Path
from .common import *
from .pipeline import evaluate,base_request

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True);p.add_argument('--classes',required=True);p.add_argument('--cameras',help='JSON list of name, center, rotation, optional image; world-to-camera rotation');p.add_argument('--split',default='test');p.add_argument('--view');p.add_argument('--width',type=int,default=1024);p.add_argument('--semantic-width',type=int,default=256);p.add_argument('--gpu',type=int,default=0);a=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch
    from transformers import CLIPModel,CLIPProcessor
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True);cfg=default_config();cfg.update(width=a.width,height=a.width//2,semantic_width=a.semantic_width,semantic_height=a.semantic_width//2,gpu=a.gpu,classes=[x.strip() for x in a.classes.split(',') if x.strip()])
    modeldir=ROOT/'weights/clip-vit-base-patch16';model=CLIPModel.from_pretrained(modeldir,local_files_only=True).cuda().eval();processor=CLIPProcessor.from_pretrained(modeldir,local_files_only=True)
    with torch.inference_mode():
        x=processor(text=['a photo of a '+v for v in cfg['classes']],return_tensors='pt',padding=True);feat=torch.nn.functional.normalize(model.get_text_features(**{k:v.cuda() for k,v in x.items()}),dim=-1);feat.cpu().numpy().astype('f4').tofile(out/'query_text.f32')
    del model;torch.cuda.empty_cache()
    views=read(a.cameras) if a.cameras else cameras(a.split)
    if a.view:views=[v for v in views if v['name']==a.view]
    assert views,'No requested camera found'
    with Stage(out,{'model':ref(a.model),'views':views,'queries':cfg['classes']},cfg) as stage:
        request={**base_request(cfg),'action':'render','archive':str(Path(a.model).resolve()),'query_text':str(out/'query_text.f32'),'views':views,'output':str(out/'renders')}
        native(request,out);from .visuals import semantic_preview
        from PIL import Image
        import numpy as np
        for v in views:
            name=v['name'];scores=np.fromfile(out/'renders'/(name+'_logits.f16'),dtype='f2').reshape(len(cfg['classes']),cfg['semantic_height'],cfg['semantic_width']);label=scores.argmax(0).astype('u1');ids=np.fromfile(out/'renders'/(name+'_ids.i32'),dtype='i4').reshape(label.shape);label[ids<0]=255
            Image.fromarray(label).save(out/'renders'/(name+'_labels.png'));semantic_preview(out/'renders'/(name+'_rgb.png'),label,cfg['classes'],out/'renders'/(name+'_semantic.png'))
        stage.done({'renders':str(out/'renders'),'queries':cfg['classes'],'unknown_value':255},{'semantic_accuracy':None,'reason':'No semantic GT supplied for novel queries'})
if __name__=='__main__':main()
