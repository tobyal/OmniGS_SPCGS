"""Frozen CLIP class-prototype supervision from versioned official labels.

This is not image-CLIP distillation. Invalid pixels never supervise cosine/CE.
"""
from pathlib import Path
import os
import numpy as np
from PIL import Image
from .common import ROOT,read,write,ref,signature,project

PROMPTS={
 'piping':['industrial pipes and valves','a pipe with elbows, flanges and valves','insulated industrial piping'],
 'equipment':['industrial machinery and electrical cabinets','pumps, motors and equipment housings','industrial tanks, junction boxes and fixed light fixtures'],
 'building':['the floor, walls and ceiling of an industrial building','building columns, doors and concrete foundations','civil building structure and concrete steps'],
 'structural_frame':['metal beams and structural supports','steel frames supporting industrial equipment','metal brackets and pipe supports'],
 'access_structure':['industrial access platforms, guardrails and ladders','metal grating walkways and handrails','maintenance access structures'],
 'cable_tray':['industrial cable trays','metal cable baskets and cable tray channels','overhead trays carrying electrical cables'],
 'hvac':['ventilation ducts and air vents','rectangular metal air ducts','round corrugated ventilation ducts'],
}

def targets(labels,text,smoothing=.05):
    labels=np.asarray(labels);valid=labels!=255;K=len(text)
    assert set(np.unique(labels))<=set(range(K))|{255}
    ids=np.where(valid,labels,0)
    feature=text[ids].transpose(2,0,1).copy();feature[:,~valid]=0
    probabilities=np.full((K,*labels.shape),smoothing/K,dtype='f4')
    for k in range(K):probabilities[k]+=(ids==k)*(1-smoothing)
    # Native consumes softmax(S), so store log probabilities, not probabilities.
    scores=np.log(probabilities);scores[:,~valid]=-np.log(K)
    return feature.astype('f2'),scores.astype('f2'),valid.astype('u1')

def extract(run,cfg):
    import torch
    from transformers import CLIPModel,CLIPTokenizer
    run=Path(run).resolve();protocol=read(run/'00_protocol/manifest.json');folder=run/'03_semantic/cache';folder.mkdir(parents=True,exist_ok=True)
    classes=protocol['classes']['classes'];assert [c['id'] for c in classes]==list(range(7))
    names=[c['name'] for c in classes];assert names==list(PROMPTS)
    H,W=128,256;modeldir=ROOT/'weights/clip-vit-base-patch16'
    inputs={'protocol':ref(run/'00_protocol/manifest.json'),'weights':ref(modeldir/'pytorch_model.bin'),'prompts':PROMPTS,'size':[H,W],'smoothing':.05,'schema':'official_prototypes_v1'}
    key=signature(inputs)
    if (folder/'manifest.json').exists():
        result=read(folder/'manifest.json');assert result['signature']==key;return result
    os.environ['CUDA_VISIBLE_DEVICES']=str(cfg['gpu']);torch.set_num_threads(4)
    model=CLIPModel.from_pretrained(modeldir,local_files_only=True).eval().cuda();model.requires_grad_(False)
    tokenizer=CLIPTokenizer.from_pretrained(modeldir,local_files_only=True);rows=[]
    with torch.inference_mode():
        for name in names:
            tokens=tokenizer(PROMPTS[name],padding=True,return_tensors='pt');feat=model.get_text_features(**{k:v.cuda() for k,v in tokens.items()})
            feat=torch.nn.functional.normalize(feat,dim=-1);rows.append(torch.nn.functional.normalize(feat.mean(0),dim=0).cpu().numpy())
    text=np.stack(rows).astype('f4');assert text.shape==(7,512)
    # All seven prototypes are represented exactly in the initial 32D subspace.
    basis=np.linalg.qr(np.concatenate([text.T,np.eye(512,dtype='f4')],1))[0][:,:32].astype('f4')
    assert np.max(np.abs(text@basis@basis.T-text))<1e-5
    text.tofile(folder/'text.f32');basis.tofile(folder/'decoder.f32');write(folder/'prompts.json',PROMPTS)
    label_rows={r['id']:r for r in protocol['labels']};views=[];counts=np.zeros(7,dtype=np.int64);palette=np.array([c['color'] for c in classes],dtype='u1')
    for v in protocol['train']:
        label_info=label_rows[v['name']];assert ref(label_info['label'])['sha256']==label_info['label_sha256']
        labels=np.array(Image.open(label_info['label']).resize((W,H),Image.Resampling.NEAREST));F,S,mask=targets(labels,text)
        base=folder/v['name'];F.tofile(str(base)+'_F.f16');S.tofile(str(base)+'_S.f16');mask.tofile(str(base)+'_W.u8');Image.fromarray(labels).save(str(base)+'_labels.png')
        valid=mask.astype(bool);counts+=np.bincount(labels[valid],minlength=7)
        assert np.array_equal(S.argmax(0)[valid],labels[valid]);assert np.allclose(F[:,valid].T,text[labels[valid]],atol=3e-4)
        colors=np.zeros((H,W,3),dtype='u1');colors[valid]=palette[labels[valid]]
        rgb=Image.open(v['image']).convert('RGB').resize((1024,512));target=Image.fromarray(colors).resize((1024,512),Image.Resampling.NEAREST)
        canvas=Image.new('RGB',(1024,1024));canvas.paste(rgb,(0,0));canvas.paste(target,(0,512));(folder/'preview').mkdir(exist_ok=True);canvas.save(folder/'preview'/(v['name']+'.png'))
        views.append({**v,'feature':str(base)+'_F.f16','feature_height':H,'feature_width':W,
          'relevancy':str(base)+'_S.f16','relevancy_height':H,'relevancy_width':W,'relevancy_classes':7,'semantic_target':'softmax_relevancy',
          'semantic_mask':str(base)+'_W.u8','mask_height':H,'mask_width':W,'valid_fraction':float(valid.mean()),'labels':str(base)+'_labels.png','label_provenance':label_info})
    result={'signature':key,'inputs':inputs,'views':views,'generated_views':[],'text':str(folder/'text.f32'),'decoder':str(folder/'decoder.f32'),
      'classes':names,'class_config':protocol['classes'],'feature_backend':'official_prototypes','adaptation':'Frozen CLIP text class-prototype supervision; not full image-CLIP distillation',
      'semantic_valid_counts':counts.tolist(),'mean_valid_fraction':float(np.mean([v['valid_fraction'] for v in views])),'smoothing':.05,'scores_format':'log class probabilities; native applies softmax','frozen_text':True,'prompt_count':{k:len(v) for k,v in PROMPTS.items()}}
    write(folder/'manifest.json',result);print('OFFICIAL TEACHER',result['mean_valid_fraction'],counts.tolist(),flush=True);return result

def initialize(xyz,cache,dest):
    """Nearest valid projection initializes attributes; it is not a visibility test."""
    from scipy.spatial import cKDTree
    views=cache['views'];basis=np.fromfile(cache['decoder'],dtype='f4').reshape(512,32)
    text=np.fromfile(cache['text'],dtype='f4').reshape(7,512);low=text@basis
    nearest=cKDTree([v['center'] for v in views]).query(xyz,k=len(views))[1]
    out=np.zeros((len(xyz),32),dtype='f4');pending=np.ones(len(xyz),bool);used=np.full(len(xyz),-1,dtype='i4')
    for rank in range(len(views)):
        for i,v in enumerate(views):
            ids=np.flatnonzero(pending&(nearest[:,rank]==i))
            if not len(ids):continue
            H,W=v['feature_height'],v['feature_width'];u,y,_=project(xyz[ids],v,H,W)
            labs=np.array(Image.open(v['labels']))[np.clip(np.rint(y).astype(int),0,H-1),np.rint(u).astype(int)%W]
            valid=labs!=255;out[ids[valid]]=low[labs[valid]];pending[ids[valid]]=False;used[ids[valid]]=i
        if not pending.any():break
    out.tofile(dest);write(str(dest)+'.json',{'points':len(xyz),'valid_initializations':int((~pending).sum()),'unobserved_zero':int(pending.sum()),'policy':'nearest camera with valid projected official label; heuristic initialization, no claimed occlusion visibility','view_counts':{v['name']:int((used==i).sum()) for i,v in enumerate(views)}})
    return str(dest)
