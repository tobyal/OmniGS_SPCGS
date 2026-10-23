"""3D-OVS overlapping patch accumulation with an explicit ERP adapter.

Reference: references/3d_ovs (MIT), commit f77e2fc3. The patch schedule,
independently jittered bounds, raw CLIP embedding accumulation and spatial
min/max relevancy follow its extract_clip_feature.py and dataLoader/llff.py.
Panorama projection and fixed equal scale fusion are documented adaptations.
"""
import os, random, time
from pathlib import Path
import numpy as np
from PIL import Image
from .common import ROOT, read, write, ref, signature, Stage

VERSION='ovs_erp_v1'
FACES=[('front',np.eye(3)),('right',np.array([[0,0,1],[0,1,0],[-1,0,0]])),
       ('back',np.diag([-1,1,-1])),('left',np.array([[0,0,-1],[0,1,0],[1,0,0]])),
       ('up',np.array([[1,0,0],[0,0,-1],[0,1,0]])),('down',np.array([[1,0,0],[0,0,1],[0,-1,0]]))]

def unit(x,axis):
    return x/np.maximum(np.linalg.norm(x,axis=axis,keepdims=True),1e-8)

def patch_boxes(h,w,p,rng):
    """Same loop bounds and four independent randint calls as 3D-OVS."""
    stride=p//4
    if stride<1:raise ValueError('Patch grid too small')
    return np.array([(max(x*stride-rng.randint(0,p//4),0),max(y*stride-rng.randint(0,p//4),0),
                      min(x*stride+p+rng.randint(0,p//4),w),min(y*stride+p+rng.randint(0,p//4),h))
                     for y in range((h-p)//stride+1+int((h-p)%stride>0))
                     for x in range((w-p)//stride+1+int((w-p)%stride>0))],dtype='i4')

def accumulate_rectangles(embeddings,boxes,h,w):
    """Rectangle difference arrays, equivalent to adding each vector to its crop.

    Accumulate in float64 to avoid cancellation in the two prefix sums; the
    CLIP vectors and saved averages remain float32/float16.
    """
    d=embeddings.shape[1];delta=np.zeros((h+1,w+1,d),dtype='f8');count=np.zeros((h+1,w+1),dtype='i4')
    l,u,r,b=boxes.T
    for yy,xx,sign in [(u,l,1),(b,r,1),(u,r,-1),(b,l,-1)]:
        np.add.at(delta,(yy,xx),sign*embeddings);np.add.at(count,(yy,xx),sign)
    count=count.cumsum(0).cumsum(1)[:h,:w]
    if (count==0).any():raise ValueError('Uncovered pixel in overlapping patch grid')
    dense=(delta.cumsum(0).cumsum(1)[:h,:w]/count[...,None]).astype('f4').transpose(2,0,1)
    return dense,count.astype('i4')

def perspective(rgb,rotation,size,fov):
    import cv2
    t=((np.arange(size,dtype='f4')+.5)/size*2-1)*np.tan(np.deg2rad(fov/2))
    x,y=np.meshgrid(t,t);rays=np.stack([x,y,np.ones_like(x)],-1)@rotation.T;rays=unit(rays,-1)
    h,w=rgb.shape[:2];u=(np.arctan2(rays[...,0],rays[...,2])/(2*np.pi)+.5)*w-.5
    v=np.clip((np.arcsin(rays[...,1])/np.pi+.5)*h-.5,0,h-1)
    padded=np.concatenate([rgb[:,-1:],rgb,rgb[:,:1]],1)
    return cv2.remap(padded,(u+1).astype('f4'),v.astype('f4'),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)

def erp_rays(h,w):
    y,x=np.meshgrid(np.arange(h),np.arange(w),indexing='ij')
    lon=((x+.5)/w-.5)*2*np.pi;lat=((y+.5)/h-.5)*np.pi
    return np.stack([np.cos(lat)*np.sin(lon),np.sin(lat),np.cos(lat)*np.cos(lon)],-1)

def inverse_maps(h,w,fov):
    rays=erp_rays(h,w);t=np.tan(np.deg2rad(fov/2));grids=[];weights=[]
    for _,R in FACES:
        local=rays@R;z=local[...,2];uv=local[...,:2]/np.maximum(z[...,None],1e-8)/t
        grids.append(uv.astype('f4'));weights.append((((np.abs(uv)<=1).all(-1))&(z>0))*np.maximum(z,0)**4)
    weights=np.array(weights,dtype='f4');assert (weights.sum(0)>0).all()
    return np.array(grids),weights

def spatial_relevancy(cosine):
    lo=cosine.min(axis=(-2,-1),keepdims=True);hi=cosine.max(axis=(-2,-1),keepdims=True)
    return (cosine-lo)/np.maximum(hi-lo,1e-8)

class Encoder:
    def __init__(self,cfg):
        import torch
        from transformers import CLIPModel,CLIPProcessor
        self.torch=torch;self.cfg=cfg
        self.model=CLIPModel.from_pretrained(ROOT/'weights/clip-vit-base-patch16',local_files_only=True).eval().cuda()
        self.processor=CLIPProcessor.from_pretrained(ROOT/'weights/clip-vit-base-patch16',local_files_only=True)
    def images(self,im,boxes):
        torch=self.torch;out=[];bs=self.cfg.get('clip_batch_size',64)
        with torch.inference_mode():
            for i in range(0,len(boxes),bs):
                pixels=self.processor(images=[im.crop(tuple(b)) for b in boxes[i:i+bs]],return_tensors='pt')['pixel_values'].cuda()
                with torch.autocast('cuda',dtype=torch.float16,enabled=self.cfg.get('clip_fp16',True)):
                    vectors=self.model.get_image_features(pixel_values=pixels)
                out.append(vectors.float().cpu().numpy())
        return np.concatenate(out)
    def texts(self,prompts):
        with self.torch.inference_mode():
            tokens=self.processor(text=prompts,return_tensors='pt',padding=True,truncation=False)
            return self.model.get_text_features(**{k:v.cuda() for k,v in tokens.items()}).float().cpu().numpy()

def visual_cache(view,cache_root,cfg,weights,encoder):
    import torch
    import torch.nn.functional as fn
    source=ref(view['image']);h,w=cfg['cache_height'],cfg['cache_width']
    recipe={'version':VERSION,'image_sha256':source['sha256'],'model_sha256':weights['sha256'],
            'face_size':cfg.get('clip_face_size',1024),'down_sample':8,'fov':cfg.get('clip_fov',100),'divisors':[5,7,10],
            'seed':cfg.get('seed',0),'erp_shape':[h,w],'fp16_encoding':cfg.get('clip_fp16',True),
            'preprocess':'HF CLIPProcessor local ViT-B/16; bicubic shortest edge resize and center crop',
            'projection':'pixel centers; +Y down; periodic horizontal ERP padding; inverse bilinear face sampling; z^4 overlap weighting'}
    key=signature(recipe);folder=cache_root/'F'/key;manifest=folder/'manifest.json'
    if manifest.exists():
        m=read(manifest)
        if all(Path(m[k]).exists() for k in ['feature','scales','coverage','patch_boxes']):return m
    folder.mkdir(parents=True,exist_ok=True);write(folder/'inputs.json',{'rgb':source,'recipe':recipe,'status':'running'})
    rgb=np.asarray(Image.open(view['image']).convert('RGB'));grids,blend=inverse_maps(h,w,recipe['fov'])
    grid=torch.from_numpy(grids).cuda();blend=torch.from_numpy(blend).cuda()
    total=torch.zeros((3,512,h,w),device='cuda');counts=[];allboxes={};timing=[]
    rng=random.Random(recipe['seed']);size=recipe['face_size']//8
    for face_index,(name,R) in enumerate(FACES):
        start=time.time()
        face=Image.fromarray(perspective(rgb,R,recipe['face_size'],recipe['fov']))
        face.save(folder/(name+'.jpg'),quality=92)
        # PIL default resampling matches image.resize in the reference extractor.
        small=face.resize((size,size));facecounts=[]
        for s,div in enumerate(recipe['divisors']):
            boxes=patch_boxes(size,size,size//div,rng);embeddings=encoder().images(small,boxes)
            dense,count=accumulate_rectangles(embeddings,boxes,size,size)
            mapped=fn.grid_sample(torch.from_numpy(dense)[None].cuda(),grid[face_index:face_index+1],mode='bilinear',padding_mode='border',align_corners=False)[0]
            total[s]+=mapped*blend[face_index][None];facecounts.append(count);allboxes[f'{name}_scale{s}']=boxes
        counts.append(facecounts);timing.append(time.time()-start)
        print('DENSE',view['name'],name,'seconds',round(timing[-1],1),flush=True)
    raw=(total/blend.sum(0)[None,None]).cpu().numpy();scales=unit(raw,1)
    final=unit(scales.mean(0),0)
    np.save(folder/'raw_multiscale.npy',raw.astype('f2'));final.astype('f2').tofile(folder/'F.f16')
    np.savez_compressed(folder/'coverage.npz',counts=np.array(counts),face_weights=blend.cpu().numpy(),inverse_grids=grids)
    np.savez_compressed(folder/'patch_boxes.npz',**allboxes)
    m={'key':key,'feature':str(folder/'F.f16'),'feature_height':h,'feature_width':w,'scales':str(folder/'raw_multiscale.npy'),
       'scale_shape':[3,512,h,w],'scale_storage':'raw coverage-normalized embeddings after ERP fusion; normalize channel dimension before relevancy',
       'scale_fusion':'uniform mean of unit scale vectors, then unit normalize','coverage':str(folder/'coverage.npz'),'patch_boxes':str(folder/'patch_boxes.npz'),
       'patch_count':sum(len(b) for b in allboxes.values()),'recipe':recipe,'source':source,'seconds':sum(timing),'status':'completed'}
    write(manifest,m);return m

def text_cache(config,cache_root,weights,encoder):
    classes=config['classes'];prompts=[p for c in classes for p in c['prompts']]
    key=signature({'version':VERSION,'prompts':[c['prompts'] for c in classes],'ids':[c['id'] for c in classes],'model_sha256':weights['sha256']})
    folder=cache_root/'T'/key
    if (folder/'manifest.json').exists():
        m=read(folder/'manifest.json')
        if Path(m['text']).exists() and Path(m['prompt_features']).exists():return m
    folder.mkdir(parents=True,exist_ok=True);raw=encoder().texts(prompts);normed=unit(raw,1);text=[];rows=[];start=0
    for i,c in enumerate(classes):
        end=start+len(c['prompts']);text.append(unit(normed[start:end].mean(0),0));rows.append({'row':i,'class_id':c['id'],'key':c['key'],'name_en':c['name_en'],'prompts':c['prompts'],'prompt_rows':list(range(start,end))});start=end
    np.savez_compressed(folder/'prompt_features.npz',raw=raw,normalized=normed);np.array(text,dtype='f4').tofile(folder/'T.f32')
    m={'key':key,'text':str(folder/'T.f32'),'shape':[len(classes),512],'prompt_features':str(folder/'prompt_features.npz'),'rows':rows,'aggregation':'unit-normalize each prompt, arithmetic class mean, unit-normalize class mean','status':'completed'}
    write(folder/'manifest.json',m);return m

def relevancy_cache(f,t,cache_root,view,classes):
    key=signature({'version':VERSION,'F':f['key'],'T':t['key'],'scale_fusion':'equal','target':'softmax(mean(spatial_minmax(T*unit(F_scale))))'})
    folder=cache_root/'S'/key
    if (folder/'manifest.json').exists():
        m=read(folder/'manifest.json')
        if Path(m['relevancy']).exists() and Path(m['maps']).exists():return m
    folder.mkdir(parents=True,exist_ok=True)
    scales=unit(np.load(f['scales']).astype('f4'),1);text=np.fromfile(t['text'],dtype='f4').reshape(-1,512)
    raw=np.einsum('kd,sdhw->skhw',text,scales,optimize=True);rel=spatial_relevancy(raw);s=rel.mean(0)
    feat=np.fromfile(f['feature'],dtype='f2').astype('f4').reshape(512,f['feature_height'],f['feature_width'])
    aggregate_cosine=np.einsum('kd,dhw->khw',text,unit(feat,0),optimize=True)
    s.astype('f2').tofile(folder/'S.f16')
    probs=np.exp(s-s.max(0));probs/=probs.sum(0);entropy=-(probs*np.log(probs+1e-8)).sum(0);sorted_s=np.sort(s,axis=0);margin=sorted_s[-1]-sorted_s[-2]
    np.savez_compressed(folder/'maps.npz',cosine_scales=raw.astype('f2'),relevancy_scales=rel.astype('f2'),cosine=aggregate_cosine.astype('f2'),S=s.astype('f2'),probabilities=probs.astype('f2'),labels=s.argmax(0).astype('u1'),entropy=entropy.astype('f2'),margin=margin.astype('f2'))
    from .visuals import semantic_preview
    semantic_preview(view['image'],s.argmax(0),classes,folder/'semantic.png')
    semantic_preview(view['image'],aggregate_cosine.argmax(0),classes,folder/'raw_cosine_semantic.png')
    save_responses(view['image'],s,classes,folder/'responses.jpg')
    m={'key':key,'relevancy':str(folder/'S.f16'),'relevancy_height':f['feature_height'],'relevancy_width':f['feature_width'],'relevancy_classes':len(classes),
       'maps':str(folder/'maps.npz'),'F_key':f['key'],'T_key':t['key'],'target_conversion':'softmax over classes, temperature 1, after bilinear resize; explicit approximation because SPC-GS target conversion is not public',
       'uncertainty':{'mean_entropy':float(entropy.mean()),'normalized_entropy':float(entropy.mean()/np.log(len(classes))),'margin_quantiles':np.quantile(margin,[0,.1,.5,.9,1]).tolist(),'label_fraction':dict(zip(classes,np.bincount(s.argmax(0).ravel(),minlength=len(classes)).astype(float)/s.shape[1]/s.shape[2]))},'status':'completed'}
    write(folder/'manifest.json',m);return m

def save_responses(rgb,s,classes,out):
    import cv2
    from PIL import ImageDraw
    w,h=384,192;cols=4;rows=(len(classes)+1+cols-1)//cols
    canvas=Image.new('RGB',(cols*w,rows*(h+24)),'white');draw=ImageDraw.Draw(canvas)
    canvas.paste(Image.open(rgb).convert('RGB').resize((w,h)),(0,24));draw.text((5,4),'RGB',fill='black')
    for i,c in enumerate(classes,1):
        x=i%cols*w;y=i//cols*(h+24);heat=cv2.applyColorMap((np.clip(s[i-1],0,1)*255).astype('u1'),cv2.COLORMAP_TURBO)
        canvas.paste(Image.fromarray(heat[:,:,::-1]).resize((w,h)),(x,y+24));draw.text((x+4,y+4),c,fill='black')
    canvas.save(out,quality=92)

def extract(views,folder,cfg):
    os.environ['CUDA_VISIBLE_DEVICES']=str(cfg.get('gpu',0));import torch
    torch.set_num_threads(cfg.get('clip_cpu_threads',4))
    folder=Path(folder);cache_root=Path(cfg.get('clip_cache_root',ROOT/'cache/industrial_clip'));cache_root.mkdir(parents=True,exist_ok=True)
    config=read(ROOT/cfg.get('class_config','configs/industrial_classes.json'));classes=[c['name_en'] for c in config['classes']]
    if classes!=cfg['classes']:raise ValueError('Class configuration and training class row order differ')
    weights=ref(ROOT/'weights/clip-vit-base-patch16/pytorch_model.bin');enc=[]
    def encoder():
        if not enc:enc.append(Encoder(cfg))
        return enc[0]
    with Stage(folder,{'views':views,'class_config':ref(ROOT/cfg.get('class_config','configs/industrial_classes.json')),'model':weights},cfg) as stage:
        t=text_cache(config,cache_root,weights,encoder);rows=[];cov=torch.zeros((512,512),device='cuda')
        for view in views:
            f=visual_cache(view,cache_root,cfg,weights,encoder);s=relevancy_cache(f,t,cache_root,view,classes)
            rows.append({**view,**{k:f[k] for k in ['feature','feature_height','feature_width']},**{k:s[k] for k in ['relevancy','relevancy_height','relevancy_width','relevancy_classes']},'F_manifest':str(Path(f['feature']).parent/'manifest.json'),'S_manifest':str(Path(s['relevancy']).parent/'manifest.json')})
            feat=np.fromfile(f['feature'],dtype='f2').reshape(512,-1)[:,::16].astype('f4');v=torch.from_numpy(feat).cuda();cov+=v@v.T
            print('FST',view['name'],'patches',f['patch_count'],'entropy',s['uncertainty']['normalized_entropy'],flush=True)
        _,u=torch.linalg.eigh(cov);u[:,-32:].flip(1).cpu().numpy().astype('f4').tofile(folder/'decoder.f32')
        # Keep the standard run-local manifest beside its text file for train_stage.
        import shutil
        shutil.copy2(t['text'],folder/'text.f32')
        result={'signature':signature({'views':rows,'T':t['key'],'version':VERSION}),'views':rows,'text':str(folder/'text.f32'),'T_manifest':str(Path(t['text']).parent/'manifest.json'),
                'decoder':str(folder/'decoder.f32'),'classes':classes,'feature_backend':'ovs_multiscale','class_config':ref(ROOT/cfg.get('class_config','configs/industrial_classes.json')),'model':weights,'cache_root':str(cache_root),
                'adaptation':f"six overlapping {cfg.get('clip_fov',100)}-degree views; 3D-OVS patch accumulation; uniform scale fusion (not its learned scale selector); cached continuous S used by soft CE"}
        write(folder/'manifest.json',result);stage.done({'manifest':str(folder/'manifest.json'),'views':len(rows)})
    return result
