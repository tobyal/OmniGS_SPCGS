"""ERP label geometry and immutable review revisions. No CLIP or training."""
from pathlib import Path
from functools import lru_cache
import io, json, hashlib, time, uuid, zipfile, threading
import cv2
import numpy as np
from PIL import Image
from .common import ROOT, read, write, sha
from .clip_dense import FACES, perspective, erp_rays

DEFAULT_RUN = ROOT/'runs/iris_semantics_20261024'
CONFIG = ROOT/'configs/iris_segmentation_classes.json'
FACE_CODES = dict(zip('FRBLUD', FACES))
LOCK = threading.RLock()

def png_bytes(array):
    out=io.BytesIO(); Image.fromarray(array).save(out,format='PNG'); return out.getvalue()

def validate(labels, config):
    if labels.shape != (1024,2048) or labels.dtype != np.uint8:
        raise ValueError('Labels must be uint8, 2048 x 1024')
    if not set(np.unique(labels)).issubset({255, *(c['id'] for c in config['classes'])}):
        raise ValueError('Unknown class IDs')

@lru_cache(maxsize=12)
def inverse_face(face, size, fov=90):
    rays=erp_rays(1024,2048); rotation=dict(FACES)[face]
    local=rays@rotation
    z=local[...,2]; scale=np.tan(np.deg2rad(fov/2))
    x=local[...,0]/np.maximum(z,1e-8)/scale
    y=local[...,1]/np.maximum(z,1e-8)/scale
    valid=(z>0)&(np.abs(x)<=1)&(np.abs(y)<=1)
    xx=np.clip(np.rint((x+1)*size/2-.5),0,size-1).astype('i4')
    yy=np.clip(np.rint((y+1)*size/2-.5),0,size-1).astype('i4')
    return yy,xx,valid,z

def from_faces(faces, size=1500):
    # Each pixel belongs to the face with greatest forward depth. Exact nearest
    # sampling of IDs; face boundaries have deterministic tie ownership.
    rays=erp_rays(1024,2048)
    owner=np.stack([(rays@r)[...,2] for _,r in FACES]).argmax(0)
    out=np.full((1024,2048),255,np.uint8)
    for i,(name,_) in enumerate(FACES):
        yy,xx,valid,_=inverse_face(name,size,90)
        pick=(owner==i)&valid
        out[pick]=faces[name][yy[pick],xx[pick]]
    return out

def point_to_face(x,y,face,size=1024,fov=100):
    lon=((x+.5)/2048-.5)*2*np.pi;lat=((y+.5)/1024-.5)*np.pi
    ray=np.array([np.sin(lon)*np.cos(lat),np.sin(lat),np.cos(lon)*np.cos(lat)])
    q=ray@dict(FACES)[face]
    xy=(q[:2]/q[2]/np.tan(np.deg2rad(fov/2))+1)*size/2-.5
    return xy,q[2]

def best_face(x,y):
    return max((n for n,_ in FACES),key=lambda n:point_to_face(x,y,n)[1])

def mask_to_erp(mask,face,fov=100):
    yy,xx,valid,_=inverse_face(face,mask.shape[0],fov)
    return mask[yy,xx].astype(bool)&valid

def decode_rle(seg):
    counts=np.asarray(seg['counts'],dtype='i8')
    if (counts<0).any() or counts.sum()!=np.prod(seg['size']):raise ValueError('Invalid RLE')
    return np.repeat(np.arange(len(counts))%2,counts).reshape(seg['size'],order='F').astype(bool)

def palette(config):
    p=np.zeros((256,3),np.uint8);p[255]=[32,38,48]
    for c in config['classes']:p[c['id']]=c['color']
    return p

def statistics(labels,source,config):
    return {'ignore_fraction':float((labels==255).mean()),
      'class_pixels':{str(c['id']):int((labels==c['id']).sum()) for c in config['classes']},
      'source_pixels':{str(i):int((source==i).sum()) for i in range(5)}}

class Store:
    def __init__(self,root=DEFAULT_RUN):
        self.root=Path(root).resolve();self.config=read(self.root/'classes.json')
        self.manifest=read(self.root/'manifest.json');self.items={r['id']:r for r in self.manifest['images']}
    def folder(self,name):
        if name not in self.items:raise KeyError('Unknown image')
        return self.root/'images'/name
    def state(self,name):return read(self.folder(name)/'state.json')
    def arrays(self,name):
        s=self.state(name);p=self.folder(name)/'revisions'/s['revision']
        return np.array(Image.open(p/'labels.png')),np.array(Image.open(p/'source.png')),s
    def save(self,name,labels,source,base,actor='human',action='edit',details=None):
        validate(labels,self.config)
        if source.shape!=labels.shape or source.dtype!=np.uint8 or not set(np.unique(source)).issubset(range(5)):raise ValueError('Invalid provenance')
        with LOCK:
            previous=self.state(name) if (self.folder(name)/'state.json').exists() else None
            if previous and previous['revision']!=base:raise RuntimeError('Revision changed; reload before saving')
            revision=uuid.uuid4().hex[:16];p=self.folder(name)/'revisions'/revision;p.mkdir(parents=True)
            (p/'labels.png').write_bytes(png_bytes(labels));(p/'source.png').write_bytes(png_bytes(source))
            metadata={'revision':revision,'parent':base,'created_at':time.time(),'actor':actor,'action':action,'details':details,
                      'labels_sha256':sha(p/'labels.png'),'rgb_sha256':self.items[name]['rgb_sha256'],
                      **statistics(labels,source,self.config)}
            write(p/'metadata.json',metadata)
            state={'revision':revision,'review_state':'issues' if previous and previous['review_state']=='issues' else 'candidate',
                   'approved_revision':None,'issues':previous.get('issues',[]) if previous else [],**statistics(labels,source,self.config)}
            write(self.folder(name)/'state.json',state)
            return state
    def review(self,name,revision,approve,reviewer,note='',issues=None):
        with LOCK:
            s=self.state(name)
            if s['revision']!=revision:raise RuntimeError('Revision changed; reload')
            if approve and not reviewer.strip():raise ValueError('Human reviewer name required')
            s.update(review_state='approved' if approve else ('issues' if issues else 'candidate'),
                     approved_revision=revision if approve else None,reviewer=reviewer,reviewed_at=time.time(),review_note=note)
            if issues is not None:s['issues']=issues
            history=self.folder(name)/'review_events.jsonl'
            with history.open('a') as f:f.write(json.dumps(s,ensure_ascii=False)+'\n')
            write(self.folder(name)/'state.json',s);return s
    def export(self,approved_only=True):
        with LOCK:
            rows=[];buff=io.BytesIO()
            with zipfile.ZipFile(buff,'w',zipfile.ZIP_DEFLATED) as z:
                z.writestr('classes.json',json.dumps(self.config,ensure_ascii=False,indent=2))
                for name,item in self.items.items():
                    labels,source,state=self.arrays(name)
                    approved=state['review_state']=='approved' and state['approved_revision']==state['revision']
                    if approved_only and not approved:continue
                    validate(labels,self.config)
                    if source.shape!=labels.shape or source.dtype!=np.uint8 or not set(np.unique(source)).issubset(range(5)):raise ValueError('Invalid provenance')
                    if sha(item['rgb'])!=item['rgb_sha256']:raise ValueError('RGB changed: '+name)
                    folder=self.folder(name)
                    if sha(folder/'revisions'/state['revision']/'labels.png')!=read(folder/'revisions'/state['revision']/'metadata.json')['labels_sha256']:raise ValueError('Saved label hash changed: '+name)
                    z.writestr('labels/'+name+'.png',png_bytes(labels));z.writestr('source/'+name+'.png',png_bytes(source))
                    folder=self.folder(name)
                    z.writestr('metadata/'+name+'.json',(folder/'revisions'/state['revision']/'metadata.json').read_bytes())
                    for audit in ['coverage.json','sam_candidates.json','refinement.json','finalization.json']:
                        if (folder/audit).exists():z.writestr('audit/'+name+'/'+audit,(folder/audit).read_bytes())
                    rows.append({**item,**state,'human_approved':approved,'label':'labels/'+name+'.png',
                                 'label_sha256':hashlib.sha256(png_bytes(labels)).hexdigest()})
                z.writestr('manifest.json',json.dumps({'schema':'iris_erp_semantic_v1','width':2048,'height':1024,'ignore_id':255,
                    'approved_only':approved_only,'unreviewed_candidates_included':any(not r['human_approved'] for r in rows),
                    'images':rows},ensure_ascii=False,indent=2))
            return buff.getvalue(),len(rows)

class SamSegmenter:
    """One loaded model, one cached perspective embedding; prompts are explicit."""
    def __init__(self,device='cuda'):
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        torch.set_num_threads(2);self.torch=torch;self.device=device
        self.predictor=SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(ROOT/'weights/sam2.1_hiera_tiny.pt'),device=device,apply_postprocessing=False))
        self.key=None;self.lock=threading.Lock()
    def predict(self,path,points,labels=None,face=None,box=None):
        labels=labels or [1]*len(points)
        if not points:raise ValueError('At least one positive point is required')
        if len(labels)!=len(points) or not set(labels).issubset({0,1}) or 1 not in labels:raise ValueError('Invalid SAM point labels')
        face=face or best_face(*points[labels.index(1)]);key=(str(path),face)
        projected=[point_to_face(x,y,face) for x,y in points]
        xy=np.array([p[0] for p in projected],dtype='f4')
        if any(p[1]<=0 for p in projected) or (xy<0).any() or (xy>1023).any():raise ValueError('Prompts must fit within the same 100 degree perspective')
        with self.lock,self.torch.inference_mode():
            if key!=self.key:
                rgb=np.array(Image.open(path).convert('RGB'))
                self.predictor.set_image(perspective(rgb,dict(FACES)[face],1024,100));self.key=key
            masks,scores,_=self.predictor.predict(point_coords=xy,point_labels=np.array(labels),box=box,multimask_output=True)
            k=int(np.argmax(scores));mask=mask_to_erp(masks[k],face)
            return mask,{'face':face,'model_score':float(scores[k]),'points':points,'point_labels':labels,'face_points':xy.tolist(),'pixels':int(mask.sum())}
