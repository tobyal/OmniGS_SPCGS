"""Frozen SAM2 correspondence with optional model reuse, never a fallback model.

The client needs only stdlib. All masks return to original ERP coordinates.
"""
import sys,json,time,hashlib,socket,traceback
from pathlib import Path

class SamWorker:
    def __init__(self,c):
        import torch
        from sam2.build_sam import build_sam2_video_predictor
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        import sam2
        torch.set_num_threads(2)
        self.cfg=c.get('sam_config','configs/sam2/sam2_hiera_l.yaml')
        self.ckpt=str(Path(c.get('sam_checkpoint',str(Path(c['root'])/'weights/sam2_hiera_large.pt'))).resolve())
        if c.get('require_sam2_hiera_large',False):
            assert self.cfg=='configs/sam2/sam2_hiera_l.yaml' and Path(self.ckpt).name=='sam2_hiera_large.pt'
        self.model=build_sam2_video_predictor(self.cfg,self.ckpt,device='cuda',apply_postprocessing=False)
        # Video predictor inherits SAM2Base; AMG can share its frozen backbone.
        self.generator=SAM2AutomaticMaskGenerator(self.model,points_per_side=8,points_per_batch=16,
            pred_iou_thresh=.7,stability_score_thresh=.85,crop_n_layers=0)
        self.model.eval();self.model.requires_grad_(False)
        config_path=Path(sam2.__file__).parent/self.cfg
        self.provenance={'backend':Path(self.ckpt).stem,'checkpoint':self.ckpt,'sam_config':self.cfg,
            'worker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'checkpoint_sha256':hashlib.sha256(Path(self.ckpt).read_bytes()).hexdigest(),
            'configuration_path':str(config_path),'configuration_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
            'predictor_class':type(self.model).__name__,'parameters':sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters':sum(p.numel() for p in self.model.parameters() if p.requires_grad),'torch':torch.__version__}
        self.requests=0

    def run(self,c):
        import numpy as np
        import torch
        from PIL import Image,ImageDraw
        assert c.get('sam_config',self.cfg)==self.cfg and str(Path(c.get('sam_checkpoint',self.ckpt)).resolve())==self.ckpt
        start=time.time();out=Path(c['output']);out.mkdir(parents=True,exist_ok=True)
        rng=np.random.default_rng(c['seed']);imgs=[np.array(Image.open(p).convert('RGB')) for p in c['images']]
        H,W=imgs[0].shape[:2];assert len(imgs)==3 and all(im.shape==imgs[0].shape for im in imgs)
        x,y=int(rng.integers(W)),int(rng.integers(H));shift=W//2-x
        frames=out/'sequence';frames.mkdir(exist_ok=True)
        for i,im in enumerate(imgs):Image.fromarray(np.roll(im,shift,axis=1)).save(frames/f'{i:05d}.jpg',quality=95)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            state=self.model.init_state(video_path=str(frames),offload_video_to_cpu=True)
            self.model.add_new_points_or_box(inference_state=state,frame_idx=0,obj_id=1,
                points=np.array([[W//2,y]],dtype='f4'),labels=np.array([1],dtype='i4'))
            masks=np.zeros((3,H,W),dtype='u1')
            for index,objects,logits in self.model.propagate_in_video(state):
                masks[index]=np.roll((logits[0,0]>0).cpu().numpy(),-shift,axis=1)
            del state
            proposals=self.generator.generate(np.roll(imgs[0],shift,axis=1))
        regions=np.full((H,W),-1,dtype='i4');proposal_masks=[];proposal_metadata=[]
        for i,p in enumerate(sorted(proposals,key=lambda p:-p['area'])):
            original=np.roll(p['segmentation'],-shift,axis=1);regions[original]=i;proposal_masks.append(original)
            ys,xs=np.nonzero(original)
            proposal_metadata.append({'region_id':i,'area':int(p['area']),'predicted_iou':float(p['predicted_iou']),'stability_score':float(p['stability_score']),
                'original_ERP_bbox_xyxy':[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)] if len(xs) else None,
                'point_coords_original_ERP':[[float((x-shift)%W),float(y)] for x,y in p['point_coords']],
                'mask_coordinates':'original unrolled ERP; mask may cross longitude seam'})
        packed=np.packbits(np.asarray(proposal_masks,dtype='u1').reshape(-1,H,W),axis=-1)
        np.savez_compressed(out/'automatic_masks.npz',packed=packed,shape=np.array([len(proposal_masks),H,W],dtype='i4'))
        (out/'automatic_masks.json').write_text(json.dumps(proposal_metadata,indent=2))
        hs,ws=c['height'],c['width'];small=[]
        for i,mask in enumerate(masks):
            small.append(np.array(Image.fromarray(mask).resize((ws,hs),Image.Resampling.NEAREST)))
            overlay=imgs[i].copy();overlay[mask>0]=(overlay[mask>0]*.4+np.array([240,80,80])*.6).astype('u1')
            im=Image.fromarray(overlay);d=ImageDraw.Draw(im)
            if i==0:d.ellipse((x-5,y-5,x+5,y+5),fill='yellow')
            im.save(out/f'mask_{i}.png');Image.fromarray(mask*255).save(out/f'mask_{i}_binary.png')
        np.stack(small).astype('u1').tofile(out/'masks.u8')
        reg=np.array(Image.fromarray(regions).resize((ws,hs),Image.Resampling.NEAREST),dtype='i4');reg.tofile(out/'regions.i32')
        areas=masks.mean((1,2));ratios=areas[1:]/max(float(areas[0]),1e-10);valid=[];reasons=[]
        for a,r,s in zip(areas[1:],ratios,small[1:]):
            reasons.append([why for ok,why in [(.001<areas[0]<.95,'real_mask_area'),(.001<a<.95,'pseudo_mask_area'),(.1<r<10,'area_ratio'),(int(s.sum())>=9,'semantic_grid_area')] if not ok])
            valid.append(not reasons[-1])
        self.requests+=1
        result={**self.provenance,'status':'completed','prompt_xy':[x,y],'rolled_prompt_xy':[W//2,y],
            'semantic_prompt_xy':[(x+.5)*ws/W-.5,(y+.5)*hs/H-.5],
            'horizontal_roll':shift,'seed':c['seed'],'valid':valid,'invalid_reasons':reasons,
            'area_fractions':areas.tolist(),'area_ratios':ratios.tolist(),'automatic_masks':len(proposals),
            'seconds':time.time()-start,'worker_request_number':self.requests,'rgb_size':[W,H],'semantic_size':[ws,hs],
            'validity_note':'Area/nonempty gates only, not object correspondence accuracy. Original ERP coordinates restored before resizing.',
            'masks':str(out/'masks.u8'),'regions':str(out/'regions.i32'),'all_automatic_masks':str(out/'automatic_masks.npz'),
            'all_automatic_mask_metadata':str(out/'automatic_masks.json')}
        (out/'outputs.json').write_text(json.dumps(result,indent=2));return result

def client(c):
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as s:
        s.settimeout(900);s.connect(c['sam_server_socket']);s.sendall((json.dumps(c)+'\n').encode())
        with s.makefile('r') as f:result=json.loads(f.readline())
    if result.get('status')!='completed':raise RuntimeError(result)
    return result

def serve(config):
    worker=SamWorker(config);path=Path(config['sam_server_socket'])
    assert not path.exists(),f'Socket already exists: {path}'
    try:
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as s:
            s.bind(str(path));s.listen(2)
            print(json.dumps({'ready':True,**worker.provenance}),flush=True)
            while True:
                conn,_=s.accept()
                with conn,conn.makefile('r') as f:
                    try:result=worker.run(json.loads(f.readline()))
                    except Exception:result={'status':'failed','error':traceback.format_exc()}
                    conn.sendall((json.dumps(result)+'\n').encode())
    finally:path.unlink(missing_ok=True)

if __name__=='__main__':
    if sys.argv[1]=='--serve':serve(json.loads(Path(sys.argv[2]).read_text()))
    else:
        c=json.loads(Path(sys.argv[1]).read_text())
        print(json.dumps(client(c) if c.get('sam_server_socket') else SamWorker(c).run(c)))
