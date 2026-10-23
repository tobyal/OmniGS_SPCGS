"""Real pretrained CLIP patch features on six perspective faces, reprojected to ERP.

This is a documented lightweight dense-CLIP adaptation, not LSeg and not labels
from a detector. Text queries share the same pretrained 512-D embedding space.
"""
import os
from pathlib import Path
import numpy as np
from .common import ROOT,read,write,ref,signature,project,Stage

def extract(views,folder,cfg):
    os.environ['CUDA_VISIBLE_DEVICES']=str(cfg['gpu'])
    import torch
    from torch.nn import functional as F
    from transformers import CLIPModel,CLIPProcessor
    from PIL import Image
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    modeldir=ROOT/'weights/clip-vit-base-patch16'
    cachekey=signature({'views':[{**v,'image_sha256':ref(v['image'])['sha256']} for v in views],
        'classes':cfg['classes'],'height':cfg['cache_height'],'width':cfg['cache_width'],'weights':ref(modeldir/'pytorch_model.bin'),'extractor':cfg.get('feature_backend','patch_tokens')+'_v2'})
    if (folder/'manifest.json').exists() and read(folder/'manifest.json')['signature']==cachekey:return read(folder/'manifest.json')
    with Stage(folder,{'views':views,'model':str(modeldir)},cfg) as stage:
        model=CLIPModel.from_pretrained(modeldir,local_files_only=True).eval().cuda()
        processor=CLIPProcessor.from_pretrained(modeldir,local_files_only=True)
        with torch.inference_mode():
            tokens=processor(text=['a photo of a '+x for x in cfg['classes']],return_tensors='pt',padding=True)
            text=F.normalize(model.get_text_features(**{k:v.cuda() for k,v in tokens.items()}),dim=-1)
            text.cpu().numpy().astype('f4').tofile(folder/'text.f32')
            H,W=cfg['cache_height'],cfg['cache_width']
            yy,xx=np.meshgrid(np.arange(H),np.arange(W),indexing='ij');lon=((xx+.5)/W-.5)*2*np.pi;lat=((yy+.5)/H-.5)*np.pi
            erp=np.stack([np.cos(lat)*np.sin(lon),np.sin(lat),np.cos(lat)*np.cos(lon)],-1)
            # columns are right, down, forward in the OmniGS camera convention.
            faces=[np.eye(3),np.array([[0,0,1],[0,1,0],[-1,0,0]]),np.diag([-1,1,-1]),np.array([[0,0,-1],[0,1,0],[1,0,0]]),np.array([[1,0,0],[0,0,-1],[0,1,0]]),np.array([[1,0,0],[0,0,1],[0,-1,0]])]
            t=np.linspace(-1+1/224,1-1/224,224);cy,cx=np.meshgrid(t,t,indexing='ij');rays=np.stack([cx,cy,np.ones_like(cx)],-1)
            mapped=[];sample=[];weight=[]
            for R in faces:
                d=rays@R.T;d=d/np.linalg.norm(d,axis=-1,keepdims=True)
                grid=np.stack([np.arctan2(d[...,0],d[...,2])/np.pi,np.arcsin(d[...,1])*2/np.pi],-1)
                sample.append(torch.tensor(grid,dtype=torch.float32,device='cuda'))
                local=erp@R;z=local[...,2];uv=local[...,:2]/np.maximum(z[...,None],1e-6)
                weight.append(torch.tensor(((np.abs(uv)<1.0).all(-1)&(z>0))*np.maximum(z,0)**4,dtype=torch.float32,device='cuda'))
                mapped.append(torch.tensor(uv,dtype=torch.float32,device='cuda'))
            rows=[];samples=[]
            for v in views:
                arr=np.array(Image.open(v['image']).convert('RGB'),dtype=np.float32)/255
                inp=torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).cuda()
                # grid_sample's ERP seam needs explicit circular border, preserving pixel-center mapping.
                inp=F.pad(inp,(1,1,0,0),mode='circular');grids=torch.stack(sample);grids[...,0]*=arr.shape[1]/(arr.shape[1]+2)
                cube=F.grid_sample(inp.expand(6,-1,-1,-1),grids,align_corners=False,padding_mode='border')
                mean=torch.tensor([.48145466,.4578275,.40821073],device='cuda')[None,:,None,None];std=torch.tensor([.26862954,.26130258,.27577711],device='cuda')[None,:,None,None]
                if cfg.get('feature_backend','patch_tokens')=='local_crops':
                    # Actual CLIP image embeddings (CLS) on overlapping perspective crops.
                    # Unlike untrained patch-token projection, each feature has the same
                    # image-text objective used in CLIP pretraining. Deliberately coarse.
                    pos=np.linspace(0,112,5,dtype=int);crops=[]
                    for face in cube:
                        for y0 in pos:
                            for x0 in pos:crops.append(F.interpolate(face[:,y0:y0+112,x0:x0+112][None],size=(224,224),mode='bicubic',align_corners=False)[0])
                    embeddings=[]
                    for start in range(0,len(crops),24):
                        batch=torch.stack(crops[start:start+24]);embeddings.append(F.normalize(model.get_image_features(pixel_values=(batch-mean)/std),dim=-1))
                    patch=torch.cat(embeddings).reshape(6,5,5,512).permute(0,3,1,2)
                    # Crop centers span [-.5,.5] of the 90-degree face. Border extension
                    # is explicit and weights favor the closest cube face near seams.
                    dense=F.grid_sample(patch,torch.stack(mapped)*2,align_corners=True,padding_mode='border')
                else:
                    o=model.vision_model(pixel_values=(cube-mean)/std);patch=model.visual_projection(model.vision_model.post_layernorm(o.last_hidden_state[:,1:]))
                    patch=F.normalize(patch,dim=-1).transpose(1,2).reshape(6,512,14,14)
                    dense=F.grid_sample(patch,torch.stack(mapped),align_corners=False,padding_mode='border')
                weights=torch.stack(weight)
                assert (weights.sum(0)>0).all()
                feat=F.normalize((dense*weights[:,None]).sum(0)/weights.sum(0).clamp_min(1e-6),dim=0)
                dest=folder/(v['name']+'.f16');feat.cpu().numpy().astype('f2').tofile(dest)
                logits=torch.einsum('kc,chw->khw',text,feat);np.savez_compressed(folder/(v['name']+'_relevancy.npz'),cosine=logits.cpu().numpy().astype('f2'),labels=logits.argmax(0).cpu().numpy().astype('u1'))
                from .visuals import semantic_preview
                semantic_preview(v['image'],logits.argmax(0).cpu().numpy(),cfg['classes'],folder/'preview'/(v['name']+'.png'))
                rows.append({**v,'feature':str(dest),'feature_height':H,'feature_width':W})
                samples.append(feat.flatten(1)[:,::4]);print('CLIP',v['name'],flush=True)
            # Uncentered low-rank basis: preserves arbitrary text query ability via 32 -> 512 head.
            mat=torch.cat(samples,1).float();u,_,_=torch.linalg.svd(mat,full_matrices=False);basis=u[:,:32]
            basis.cpu().numpy().astype('f4').tofile(folder/'decoder.f32')
            result={'signature':cachekey,'views':rows,'text':str(folder/'text.f32'),'decoder':str(folder/'decoder.f32'),
                'classes':cfg['classes'],'model':ref(modeldir/'pytorch_model.bin'),'feature_backend':cfg.get('feature_backend','patch_tokens'),'adaptation':'CLIP ViT-B/16 '+cfg.get('feature_backend','patch_tokens')+' on six 90-degree perspective faces; bilinear sphere reprojection; no independent semantic GT'}
            write(folder/'manifest.json',result);stage.done({'manifest':str(folder/'manifest.json'),'views':len(rows)})
        return result

def init_semantics(xyz,cache,dest):
    """Deterministic initialization from actual training CLIP, no synthetic supervision.

    Nearest camera is used only to initialize attributes, NOT for contributor IDs.
    Occlusion is subsequently resolved by the original alpha compositor.
    """
    from scipy.spatial import cKDTree
    views=cache['views'];basis=np.fromfile(cache['decoder'],dtype='f4').reshape(512,32)
    nearest=cKDTree([v['center'] for v in views]).query(xyz)[1]
    out=np.zeros((len(xyz),32),dtype='f4')
    for i,v in enumerate(views):
        ids=np.flatnonzero(nearest==i);h,w=v['feature_height'],v['feature_width'];u,y,r=project(xyz[ids],v,h,w)
        feat=np.fromfile(v['feature'],dtype='f2').reshape(512,h,w).astype('f4');low=np.einsum('chw,cd->hwd',feat,basis)
        out[ids]=low[np.clip(np.rint(y).astype(int),0,h-1),np.rint(u).astype(int)%w]
    out.tofile(dest);return str(dest)
