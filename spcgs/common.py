from pathlib import Path
import hashlib, json, os, subprocess, time, traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/dataset'
STAGES = {'baseline':'00_baseline', 'm1':'01_sgi_sfm', 'sgi':'02_sgi_gaussians',
          'semantic':'03_semantic', 'isp':'04_isp', 'spc':'05_spc', 'compare':'06_comparison', 'report':'07_reports'}

def read(p): return json.loads(Path(p).read_text())
def write(p, data):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False));tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def signature(obj): return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()
def ref(p):
    p=Path(p).resolve();return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def run_command(args,log,env=None):
    with Path(log).open('a') as f:
        f.write('\nCOMMAND '+json.dumps([str(x) for x in args])+'\n');f.flush()
        subprocess.run([str(x) for x in args],stdout=f,stderr=subprocess.STDOUT,check=True,env=env,cwd=ROOT)
def native(request, folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);write(folder/'request.json',request)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(request.get('gpu',0)),PF_MODE='0',SGS_MODE='0',OMP_NUM_THREADS='2')
    env.pop('PF_FIELD',None)
    write(folder/'execution.json',{'binary':ref(ROOT/'code/bin/spcgs_native'),'started':time.time(),'cuda_visible_devices':env['CUDA_VISIBLE_DEVICES'],'PF_MODE':0,'SGS_MODE':0})
    run_command([ROOT/'code/bin/spcgs_native',folder/'request.json'],folder/'native.log',env)
class Stage:
    def __init__(self,folder,inputs,config):self.folder=Path(folder);self.inputs=inputs;self.config=config
    def __enter__(self):
        self.folder.mkdir(parents=True,exist_ok=True);(self.folder/'preview').mkdir(exist_ok=True)
        write(self.folder/'inputs.json',self.inputs);write(self.folder/'config.json',self.config)
        self.start=time.time();write(self.folder/'outputs.json',{'status':'running','started':self.start});return self
    def done(self,outputs,metrics=None,status='completed',reason=None):
        write(self.folder/'outputs.json',{'status':status,'reason':reason,'seconds':time.time()-self.start,**outputs})
        write(self.folder/'metrics.json',metrics or {})
    def __exit__(self,typ,value,tb):
        if typ:
            (self.folder/'error.log').write_text(''.join(traceback.format_exception(typ,value,tb)))
            self.done({},status='failed',reason=str(value))
def cameras(split):
    j=read(DATA/'openMVG'/f'data_openmvg_{split}.json');extr={x['key']:x['value'] for x in j['extrinsics']}
    result=[]
    for v in j['views']:
        d=v['value']['ptr_wrapper']['data'];p=extr[d['id_pose']];path=Path(j['root_path'])/d['filename']
        if not path.exists():path=DATA/'images'/d['filename']
        result.append({'name':Path(d['filename']).stem,'image':str(path.resolve()),'rotation':p['rotation'],'center':p['center'],'split':split,'view_id':v['key']})
    return sorted(result,key=lambda x:x['name'])
def ply(p):
    from plyfile import PlyData
    v=PlyData.read(p)['vertex'].data
    xyz=np.stack([v[k] for k in ['x','y','z']],1).astype('f4')
    if 'red' in v.dtype.names:rgb=np.stack([v[k] for k in ['red','green','blue']],1)/255.
    else:rgb=np.stack([v[f'f_dc_{i}'] for i in range(3)],1)*0.28209479177387814+0.5
    return xyz,np.clip(rgb,0,1).astype('f4'),v
def project(xyz,cam,H,W):
    v=(xyz-np.array(cam['center']))@np.array(cam['rotation']).T
    r=np.linalg.norm(v,axis=1);u=(np.arctan2(v[:,0],v[:,2])/(2*np.pi)+.5)*W-.5
    y=(np.arcsin(np.clip(v[:,1]/np.maximum(r,1e-10),-1,1))/np.pi+.5)*H-.5
    return u,y,r
def pcd_write(p,xyz,rgb):
    from plyfile import PlyData,PlyElement
    a=np.empty(len(xyz),dtype=[(k,'f4') for k in ['x','y','z']]+[(k,'u1') for k in ['red','green','blue']])
    for i,k in enumerate(['x','y','z']):a[k]=xyz[:,i]
    for i,k in enumerate(['red','green','blue']):a[k]=np.clip(rgb[:,i]*255,0,255).astype('u1')
    PlyData([PlyElement.describe(a,'vertex')],text=False).write(str(p))
def default_config(mode='quick'):
    full=mode=='full'
    return {'mode':mode,'seed':0,'gpu':0,'width':2048 if full else 512,'height':1024 if full else 256,
      'semantic_width':256 if full else 128,'semantic_height':128 if full else 64,
      'cache_width':128 if full else 64,'cache_height':64 if full else 32,
      'train_views':56 if full else 8,'eval_views':16 if full else 3,
      'sgi_iterations':10000 if full else 100,'semantic_iterations':10000 if full else 100,'spc_iterations':10000 if full else 100,
      'densify_from':500 if full else 20,'densify_every':100 if full else 20,'densify_until':9000 if full else 80,
      'densify_threshold':.0002,'max_gaussians':1000000 if full else 60000,
      'ogr_every':3000 if full else 50,'ogr_radius':1.,'ogr_min_neighbors':5,
      'isp_every':100 if full else 25,'spc_ramp':4000 if full else 80,'local_samples':128 if full else 32,
      'local_weight':1.,'contrast_samples':512 if full else 256,'save_every':1000 if full else 50,
      'opacity_reset_every':3000 if full else 0,'pose_noise_fraction':.02,
      'classes':['wall','floor','pipe','valve','pump','motor','cabinet','machine','ceiling','door','window','metal frame'],
      'm1_mode':'skip','generated_manifest':None,'generation_command':None,'sam_config':'configs/sam2.1/sam2.1_hiera_t.yaml',
      'sam_checkpoint':str(ROOT/'weights/sam2.1_hiera_tiny.pt')}
