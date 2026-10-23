"""Optional learned perceptual metric plus explicit initialization comparison."""
import argparse,os
from pathlib import Path
import numpy as np
from .common import *
from .pipeline import base_request

def lpips_eval(run,preload=False):
    import torch,lpips,cv2
    os.environ.setdefault('CUDA_VISIBLE_DEVICES','0');model=lpips.LPIPS(net='alex').eval().cuda()
    if preload:return
    run=Path(run);cfg=read(run/'config.json');lookup={v['name']:v for split in ['train','test'] for v in cameras(split)}
    for mp in run.rglob('renders/metrics.json'):
        met=read(mp)
        if not met.get('views'):continue
        values=[]
        with torch.inference_mode():
            for row in met['views']:
                name=row['view'];pred=np.fromfile(mp.parent/(name+'_rgb.f32'),dtype='f4').reshape(3,cfg['height'],cfg['width']);gt=cv2.cvtColor(cv2.imread(lookup[name]['image']),cv2.COLOR_BGR2RGB);gt=cv2.resize(gt,(cfg['width'],cfg['height']),interpolation=cv2.INTER_AREA).astype('f4').transpose(2,0,1)/255
                value=float(model(torch.from_numpy(pred.copy())[None].cuda()*2-1,torch.from_numpy(gt.copy())[None].cuda()*2-1).item());row['lpips']=value
                if row['split']=='test':values.append(value)
        met['lpips']=float(np.mean(values));met['lpips_reason']=None;met['lpips_model']='lpips AlexNet pretrained; float RGB in [-1,1]';write(mp,met)
        parent=mp.parent.parent/'metrics.json'
        if parent.exists():other=read(parent);other.update({k:met[k] for k in ['views','lpips','lpips_reason','lpips_model']});write(parent,other)
    write(run/'07_reports/lpips_status.json',{'status':'completed','backbone':'AlexNet','weights':'Official torchvision AlexNet and lpips v0.1 package linear layers'})

def initialization(run):
    from PIL import Image,ImageDraw
    from .visuals import projection_preview
    run=Path(run);cfg=read(run/'config.json');out=run/'06_comparison/initialization';out.mkdir(parents=True,exist_ok=True);manifest=read(run/'data_manifest.json');views=[manifest['train'][0],manifest['test'][0]]
    sgi=run/'02_sgi_gaussians';req={**base_request(cfg),'action':'render','archive':str(sgi/'initial/model.pt'),'output':str(out/'g0_renders'),'views':views};native(req,out/'g0_job')
    p0,c0,_=ply(run/'01_sgi_sfm/original.ply');p2,c2,_=ply(sgi/'final/point_cloud.ply')
    for v in views:
        name=v['name'];projection_preview(p0,c0,v,out/(name+'_p0.png'));projection_preview(p2,c2,v,out/(name+'_g2.png'))
        paths=[v['image'],out/(name+'_p0.png'),out/'g0_renders'/(name+'_rgb.png'),run/'00_baseline/renders'/(name+'_rgb.png'),out/(name+'_g2.png'),sgi/'renders'/(name+'_rgb.png')]
        labels=['Reference RGB','P0: original SfM points','G0: native PCD initialization','Original trained baseline','G2: SGI Gaussian centers','G2: SGI RGB rendering']
        w,h=cfg['width'],cfg['height'];canvas=Image.new('RGB',(3*w,2*(h+28)),'white');draw=ImageDraw.Draw(canvas)
        for i,(p,label) in enumerate(zip(paths,labels)):
            x=(i%3)*w;y=(i//3)*(h+28);canvas.paste(Image.open(p).convert('RGB').resize((w,h)),(x,y+28));draw.text((x+8,y+7),label,fill='black')
        canvas.save(out/(name+'_initialization.png'))
    write(out/'outputs.json',{'status':'completed','g0':str(sgi/'initial/model.pt'),'g2':str(sgi/'final/model.pt'),'g1':None,'g1_reason':'M1 generated view augmentation was skipped; no enhanced P1/G1 is claimed.'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sources={'G0 / native initialization':sgi/'initial/point_cloud.ply','G2 / SGI':sgi/'final/point_cloud.ply','Original baseline':ROOT/'baseline/checkpoint.ply'}
    fig,axes=plt.subplots(3,3,figsize=(15,11));statistics={}
    for col,(label,path) in enumerate(sources.items()):
        xyz,rgb,v=ply(path);sigma=np.exp(np.stack([v[f'scale_{i}'] for i in range(3)],1));opacity=1/(1+np.exp(-v['opacity']));stride=max(1,len(xyz)//20000)
        axes[0,col].scatter(xyz[::stride,0],xyz[::stride,1],s=.25,c=rgb[::stride]);axes[0,col].set_title(f'{label}: {len(xyz):,}');axes[0,col].set_aspect('equal');axes[0,col].set_xlabel('World X');axes[0,col].set_ylabel('World Y')
        ax=axes[1,col];ax.hist(np.log10(np.maximum(sigma.mean(1),1e-9)),bins=70,density=True,alpha=.65,label='log10 mean scale');ax.set_xlabel('log10 mean Gaussian scale');ax.set_ylabel('Density');axes[2,col].hist(opacity,bins=50,density=True,alpha=.65,color='orange');axes[2,col].set_xlabel('Opacity');axes[2,col].set_ylabel('Density')
        statistics[label]={'model':str(path),'count':len(xyz),'scale_quantiles':np.quantile(sigma,[.01,.1,.5,.9,.99]).tolist(),'opacity_quantiles':np.quantile(opacity,[.01,.1,.5,.9,.99]).tolist(),'display_stride':stride}
    fig.suptitle('Same world frame; display sampling only, complete models are saved');fig.tight_layout();fig.savefig(out/'gaussian_distribution.png',dpi=150);plt.close(fig);write(out/'distribution_metrics.json',statistics)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',nargs='?');p.add_argument('--preload',action='store_true');p.add_argument('--initialization',action='store_true');a=p.parse_args()
    if a.initialization:initialization(a.run)
    else:lpips_eval(a.run,a.preload)
