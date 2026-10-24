"""Saved-render comparisons for the sparse official-label SGI experiment."""
from pathlib import Path
import json,html
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .common import ROOT,read,write,native,ply,cameras
from .pipeline import base_request

def representatives(protocol):
    return [v for v in protocol['train'] if v['original_train_index'] in [1,14,27,48,56]]

def colored(labels,palette):
    out=np.zeros((*labels.shape,3),dtype='u1');valid=labels<len(palette);out[valid]=palette[labels[valid]];return Image.fromarray(out)

def confusion_metrics(matrix):
    m=np.asarray(matrix,dtype=np.int64);tp=np.diag(m);support=m.sum(1);union=support+m.sum(0)-tp
    iou=np.divide(tp,union,out=np.zeros(len(tp),float),where=union>0);recall=np.divide(tp,support,out=np.zeros(len(tp),float),where=support>0)
    return {'pixel_accuracy':float(tp.sum()/max(1,m.sum())),'miou':float(iou[union>0].mean()),'mean_class_accuracy':float(recall[support>0].mean()),'iou':iou.tolist(),'recall':recall.tolist(),'support':support.tolist(),'confusion':m.tolist()}

def evaluate_renders(folder,views,cache,rgb_size=(2048,1024)):
    import cv2
    from skimage.metrics import structural_similarity
    folder=Path(folder);labels={v['name']:v for v in cache['views']};palette=np.array([c['color'] for c in cache['class_config']['classes']],dtype='u1');rows=[];total=np.zeros((7,7),np.int64)
    for v in views:
        rgbpath=folder/(v['name']+'_rgb.f32')
        if not rgbpath.exists():continue
        pred=np.fromfile(rgbpath,dtype='f4').reshape(3,rgb_size[1],rgb_size[0]).transpose(1,2,0)
        gt=cv2.cvtColor(cv2.imread(v['image']),cv2.COLOR_BGR2RGB);gt=cv2.resize(gt,rgb_size,interpolation=cv2.INTER_AREA).astype('f4')/255
        mse=np.mean((pred-gt)**2);ssim=structural_similarity(gt,pred,data_range=1,channel_axis=2,gaussian_weights=True,sigma=1.5,use_sample_covariance=False)
        row={'id':v['name'],'split':v['split'],'psnr':float(-10*np.log10(max(mse,1e-12))),'ssim':float(ssim)}
        if (folder/(v['name']+'_logits.f16')).exists():
            logits=np.fromfile(folder/(v['name']+'_logits.f16'),dtype='f2').astype('f4').reshape(7,128,256);prediction=logits.argmax(0).astype('u1')
            ids=np.fromfile(folder/(v['name']+'_ids.i32'),dtype='i4').reshape(128,256)
            Image.fromarray(prediction).save(folder/(v['name']+'_labels.png'));display=prediction.copy();display[ids<0]=255
            colored(display,palette).resize((1024,512),Image.Resampling.NEAREST).save(folder/(v['name']+'_semantic.png'))
            probs=np.exp(logits-logits.max(0));probs/=probs.sum(0);row.update(predicted_fraction=(np.bincount(prediction.ravel(),minlength=7)/prediction.size).tolist(),mean_confidence=float(probs.max(0).mean()),render_coverage=float((ids>=0).mean()))
            if v['name'] in labels:
                target=np.array(Image.open(labels[v['name']]['labels']));valid=target!=255
                matrix=np.bincount((target[valid]*7+prediction[valid]).astype(int),minlength=49).reshape(7,7);total+=matrix
                row['official_training_fit']=confusion_metrics(matrix)
        rows.append(row)
    groups={split:{'views':sum(r['split']==split for r in rows),'psnr':float(np.mean([r['psnr'] for r in rows if r['split']==split])),'ssim':float(np.mean([r['ssim'] for r in rows if r['split']==split]))} for split in sorted({r['split'] for r in rows})}
    metrics={'views':rows,'groups':groups,'rgb_resolution':list(rgb_size),'semantic_resolution':[256,128],
      'semantic_evaluation_note':'Official masked training-label fit only; no semantic generalization accuracy is claimed for unseen, validation or test images.',
      'official_training_fit':confusion_metrics(total) if total.sum() else None}
    write(folder/'metrics.json',metrics);return metrics

def render_stage(run,kind,initial=False,short=False,gpu=0):
    run=Path(run).resolve();protocol=read(run/'00_protocol/manifest.json');cache=read(run/'03_semantic/cache/manifest.json')
    stage=run/{'short':'02_sgi_short','sgi':'02_sgi_gaussians','rgb':'02_rgb_p0_reference'}[kind]
    archive=stage/('initial' if initial else 'final')/'model.pt';assert archive.exists(),archive
    cfg=read(stage/'request.json');folder=stage/('renders_initial' if initial else 'renders')
    views=representatives(protocol) if initial or short else [*protocol['train'],*[{**v,'split':'unseen_train29'} for v in protocol['unseen_train29']],*protocol['validation'],*protocol['test']]
    # Only camera metadata and image targets for evaluation are introduced here,
    # after optimization. No held-out image enters any training request.
    request={**base_request(cfg),'action':'render','archive':str(archive),'gpu':gpu,'output':str(folder),'views':views,'render_semantics':kind!='rgb'}
    if not all((folder/(v['name']+'_rgb.f32')).exists() for v in views):native(request,stage/('render_initial_job' if initial else 'render_job'))
    metrics=evaluate_renders(folder,views,cache);print('EVALUATION',kind,'initial' if initial else 'final',metrics['groups'],metrics['official_training_fit'],flush=True);return metrics

def panels(run):
    run=Path(run);protocol=read(run/'00_protocol/manifest.json');cache=read(run/'03_semantic/cache/manifest.json');out=run/'07_report';out.mkdir(exist_ok=True)
    palette=np.array([c['color'] for c in cache['class_config']['classes']],dtype='u1');by_label={r['id']:r for r in protocol['labels']}
    fontfile='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';font=ImageFont.truetype(fontfile,20)
    for cam in representatives(protocol):
        name=cam['name'];label=np.array(Image.open(by_label[name]['label']));valid=label!=255
        items=[('Real RGB',Image.open(cam['image']).convert('RGB')),('Official teacher / black = ignore',colored(label,palette)),
          ('Valid semantic supervision W',Image.fromarray((valid*255).astype('u1')).convert('RGB'))]
        for title,folder,kind in [('P0 initial RGB','02_rgb_p0_reference/renders_initial','rgb'),('P0 RGB reference / 10k','02_rgb_p0_reference/renders','rgb'),
          ('P1 initial RGB','02_sgi_gaussians/renders_initial','rgb'),('P1 initial semantic','02_sgi_gaussians/renders_initial','semantic'),
          ('SGI 10k RGB','02_sgi_gaussians/renders','rgb'),('SGI 10k semantic','02_sgi_gaussians/renders','semantic')]:
            path=run/folder/(name+'_'+kind+'.png')
            if path.exists():items.append((title,Image.open(path).convert('RGB')))
        columns=3;tw,th=768,384;canvas=Image.new('RGB',(tw*columns,(th+34)*((len(items)+columns-1)//columns)+84),'white');draw=ImageDraw.Draw(canvas)
        for i,(title,im) in enumerate(items):
            x,y=(i%columns)*tw,(i//columns)*(th+34);draw.text((x+8,y+4),title,fill='black',font=font);canvas.paste(im.resize((tw,th),Image.Resampling.NEAREST if 'semantic' in title or 'teacher' in title or title.endswith('W') else Image.Resampling.LANCZOS),(x,y+34))
        top=canvas.height-76
        for i,c in enumerate(cache['class_config']['classes']):
            x=(i%4)*(canvas.width//4);y=top+(i//4)*32;draw.rectangle((x+8,y+5,x+26,y+23),fill=tuple(c['color']));draw.text((x+34,y),f"{i}: {c['name']}",font=font,fill='black')
        canvas.save(out/(name+'_comparison.jpg'),quality=92)
    index={v['name']:i+1 for i,v in enumerate(cameras('train'))}
    for cam in protocol['unseen_train29']:
        name=cam['name'];number=index[name]
        if number not in [3,22,39,55]:continue
        items=[('Unseen real RGB / no semantic GT',Image.open(cam['image']).convert('RGB'))]
        for title,folder,kind in [('P0 RGB reference / 10k','02_rgb_p0_reference/renders','rgb'),('SGI 10k RGB','02_sgi_gaussians/renders','rgb'),('SGI 10k semantic / inspection only','02_sgi_gaussians/renders','semantic')]:
            path=run/folder/(name+'_'+kind+'.png')
            if path.exists():items.append((title,Image.open(path).convert('RGB')))
        canvas=Image.new('RGB',(1536,2*418),'white');draw=ImageDraw.Draw(canvas)
        for i,(title,im) in enumerate(items):
            x,y=(i%2)*768,(i//2)*418;draw.text((x+8,y+4),title,fill='black',font=font);canvas.paste(im.resize((768,384),Image.Resampling.NEAREST if 'semantic' in title else Image.Resampling.LANCZOS),(x,y+34))
        canvas.save(out/(f'unseen_{number:02d}_'+name+'.jpg'),quality=92)
    return out

def diagnostics(run):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree
    from collections import Counter
    run=Path(run);out=run/'07_report';out.mkdir(exist_ok=True);protocol=read(run/'00_protocol/manifest.json');p0=read(run/'01_p0/metrics.json');m1=read(run/'01_sgi_sfm/metrics.json')
    contributions=read(run/'01_sgi_sfm/contributions.json');spatial={r['id']:r for r in protocol['spatial']};obs=read(run/'01_sgi_sfm/observations.json')
    points0=np.fromfile(run/'01_p0/points.f32',dtype='f4').reshape(-1,6)[:,:3];points1=np.fromfile(run/'01_sgi_sfm/enhanced_points.f32',dtype='f4').reshape(-1,6)[:,:3];added=points1[len(points0):]
    centers=np.array([r['center'] for r in protocol['spatial']]);nearest=cKDTree(centers).query(added)[1]
    region_counts=Counter(protocol['spatial'][i]['region'] for i in nearest)
    region_summary=[]
    for region in sorted({r['region'] for r in protocol['spatial']}):
        cams=[r for r in protocol['spatial'] if r['region']==region];selected=[r for r in contributions if r['region']==region]
        region_summary.append({'region':region,'training_indices':[r['index'] for r in cams],'height_layers':dict(Counter(r['height_layer'] for r in cams)),
          'selected_frames':len(selected),'points_by_source':sum(r['added_points'] for r in selected),'new_points_nearest_region':region_counts[region]})
    geometry={'p0':p0,'p1':m1,'regions':region_summary,'contributions':contributions,'historical_P2':read(ROOT/'runs/industrial_spcgs/m1_wan360_56_batch_20261023/01_sgi_sfm/metrics.json')}
    occupancy=[]
    for step in [.1,.2,.3]:
        a=set(map(tuple,np.floor(points0/step).astype('i8')));b=set(map(tuple,np.floor(points1/step).astype('i8')))
        occupancy.append({'voxel_size_world_units':step,'P0_occupied':len(a),'P1_occupied':len(b),'additional_occupied':len(b-a),'relative_increase':(len(b)-len(a))/len(a)})
    geometry['voxel_occupancy']=occupancy
    write(out/'coverage_analysis.json',{'purpose':'Post-hoc spatial sampling analysis, not a filter or ground-truth surface completeness measure. Axis-aligned grid at world origin.',
      'voxel_occupancy':occupancy,'new_point_nearest_P0':m1['new_point_distance_to_original']})
    write(out/'geometry.json',geometry)
    write(out/'camera56_connectivity.json',{'camera':protocol['train'][-1],'shared_track_edges':[e for e in read(run/'01_p0/connectivity.json')['edges'] if protocol['train'][-1]['name'] in (e['a'],e['b'])]})
    # Descriptive labels only: these votes never affect geometry acceptance.
    maps={r['id']:np.array(Image.open(r['label'])) for r in protocol['labels']};class_counts=Counter();class_points=[]
    for track in obs:
        votes=[]
        for o in track['real_observations']:
            u,v=o['uv'];label=int(maps[o['view']][int(np.clip(round(v),0,1023)),int(round(u))%2048])
            if label!=255:votes.append(label)
        counts=Counter(votes);best=counts.most_common(1)
        label=best[0][0] if best and best[0][1]>=2 and best[0][1]>len(votes)/2 else 255
        class_counts[label]+=1;class_points.append({'generated_view':track['generated_view'],'xyz':track['xyz'],'official_label_vote':label,'votes':votes})
    write(out/'new_point_semantic_votes.json',{'note':'Post-hoc analysis only; at least two agreeing official observations and a strict majority. No new filtering or geometric truth claim.',
      'counts':{str(k):v for k,v in class_counts.items()},'points':class_points})
    from .m1_wan360 import point_previews
    projection_dir=out/'point_projections';projection_dir.mkdir(exist_ok=True);point_previews(projection_dir,points0,added,representatives(protocol))
    fig,axes=plt.subplots(2,3,figsize=(16,9))
    for ax,(a,b,title) in zip(axes[0],[(0,1,'XY'),(0,2,'XZ'),(1,2,'YZ')]):
        ax.scatter(points0[:,a],points0[:,b],s=1,c='#4089bb',label=f'P0 {len(points0)}');ax.scatter(added[:,a],added[:,b],s=3,c='#e25146',label=f'new {len(added)}')
        ax.scatter(centers[:,a],centers[:,b],s=16,marker='^',c='black',label='27 cameras');ax.set_title(title);ax.set_aspect('equal');ax.legend(fontsize=8)
    axes[1,0].hist([r['max_error_px'] for r in obs],bins=30,color='#e25146');axes[1,0].set(xlabel='Maximum track reprojection error (2k pixels)',ylabel='New points')
    axes[1,1].bar([str(r['region']) for r in region_summary],[r['points_by_source'] for r in region_summary],color='#4089bb');axes[1,1].set(xlabel='Source region',ylabel='New points')
    d=m1['contribution_by_direction'];axes[1,2].bar(list(d),list(d.values()),color='#e25146');axes[1,2].set_ylabel('New points');fig.tight_layout();fig.savefig(out/'geometry.png',dpi=180);plt.close(fig)
    train_metrics={}
    for kind,folder in [('sgi','02_sgi_gaussians'),('rgb','02_rgb_p0_reference'),('short','02_sgi_short')]:
        stage=run/folder
        if not (stage/'losses.jsonl').exists():continue
        loss=[json.loads(line) for line in (stage/'losses.jsonl').read_text().splitlines()];model=read(stage/'final/model.json') if (stage/'final/model.json').exists() else None
        train_metrics[kind]={'first':loss[0],'last':loss[-1],'iterations':len(loss),'model':model,'camera_counts':dict(Counter(r['view'] for r in loss)),
          'ogr_removed':sum(r.get('ogr_removed',0) for r in loss),'ogr_events':[{'step':r['step'],'removed':r['ogr_removed']} for r in loss if 'ogr_removed' in r],
          'evaluation':read(stage/'renders/metrics.json') if (stage/'renders/metrics.json').exists() else None}
        if kind=='sgi':
            fig,axes=plt.subplots(2,3,figsize=(16,8))
            for ax,key in zip(axes.flat,['rgb','semantic_cosine','semantic_ce','local3d','total','gaussians']):
                y=np.array([r[key] for r in loss]);n=50;mean=np.convolve(y,np.ones(n)/n,mode='valid') if len(y)>=n else y
                ax.plot(np.arange(len(y)-len(mean)+1,len(y)+1),mean);ax.set_title(key+' / 50-step mean');ax.set_xlabel('Step')
                for r in loss:
                    if 'ogr_removed' in r:ax.axvline(r['step'],color='#e25146',alpha=.4)
            fig.tight_layout();fig.savefig(out/'losses.png',dpi=160);plt.close(fig)
            events=list((stage/'ogr').glob('*'));fig,axes=plt.subplots(1,max(1,len(events)),figsize=(6*max(1,len(events)),5),squeeze=False)
            for ax,event in zip(axes[0],sorted(events,key=lambda p:int(p.name))):
                xyz=np.fromfile(event/'before_xyz.f32',dtype='f4').reshape(-1,3);mask=np.fromfile(event/'removed_mask.u8',dtype='u1').astype(bool)
                ax.scatter(xyz[~mask,0],xyz[~mask,1],s=.15,c='#4089bb');ax.scatter(xyz[mask,0],xyz[mask,1],s=2,c='#e25146');ax.set_title(f'OGR step {event.name}: removed {mask.sum()}');ax.set_aspect('equal')
            fig.tight_layout();fig.savefig(out/'ogr.png',dpi=160);plt.close(fig)
    # Same world axes for the Gaussian distributions, including all outliers.
    clouds=[('P0 initial',points0),('P1 initial',points1)]
    for label,folder in [('P0 RGB / 10k','02_rgb_p0_reference'),('SGI / 10k','02_sgi_gaussians')]:
        if (run/folder/'final/point_cloud.ply').exists():clouds.append((label,ply(run/folder/'final/point_cloud.ply')[0]))
    fig,axes=plt.subplots(len(clouds),3,figsize=(16,4*len(clouds)))
    for row,(label,xyz) in enumerate(clouds):
        for ax,(a,b,title) in zip(axes[row],[(0,1,'XY'),(0,2,'XZ'),(1,2,'YZ')]):
            ax.scatter(xyz[:,a],xyz[:,b],s=.25,c='#4089bb');ax.scatter(centers[:,a],centers[:,b],s=15,c='#e25146',marker='^');ax.set_title(f'{label} / {len(xyz):,} / {title}');ax.set_aspect('equal')
    fig.tight_layout();fig.savefig(out/'gaussian_distribution.png',dpi=150);plt.close(fig)
    write(out/'training.json',train_metrics);return geometry,train_metrics

def report(run):
    import shutil
    run=Path(run);out=panels(run);geometry,training=diagnostics(run);protocol=read(run/'00_protocol/manifest.json');cache=read(run/'03_semantic/cache/manifest.json')
    previous_distribution=ROOT/'runs/iris_semantics_20261024/qa/camera_annotation_distribution.png'
    if previous_distribution.exists():shutil.copy2(previous_distribution,out/'camera_all56_annotation.png')
    summary={'run':str(run),'training_indices':[v['original_train_index'] for v in protocol['train']],'training_ids':[v['name'] for v in protocol['train']],
      'p0_points':geometry['p0']['points'],'new_points':geometry['p1']['added'],'p1_points':geometry['p1']['enhanced_count'],'regions':geometry['regions'],
      'teacher':{k:cache[k] for k in ['classes','mean_valid_fraction','semantic_valid_counts','smoothing','adaptation']},'generated_semantic_loss':False,'training':training}
    write(out/'summary.json',summary)
    fit=(training.get('sgi',{}).get('evaluation') or {}).get('official_training_fit')
    if fit:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        matrix=np.asarray(fit['confusion']);normalized=matrix/np.maximum(matrix.sum(1,keepdims=True),1);fig,axes=plt.subplots(1,2,figsize=(15,6))
        im=axes[0].imshow(normalized,vmin=0,vmax=1,cmap='Blues');axes[0].set_xticks(range(7),cache['classes'],rotation=55,ha='right');axes[0].set_yticks(range(7),cache['classes']);axes[0].set(xlabel='Prediction',ylabel='Official training label');fig.colorbar(im,ax=axes[0])
        axes[1].barh(cache['classes'],fit['iou'],color='#4089bb');axes[1].set(xlim=(0,1),xlabel='IoU on valid training pixels');fig.suptitle('Training-label fit only / not held-out semantic accuracy');fig.tight_layout();fig.savefig(out/'semantic_fit.png',dpi=180);plt.close(fig)
    body=['<!DOCTYPE html><html lang="zh"><meta charset="utf-8"><title>27 视角 M1 + SGI 实验报告</title><style>body{font:16px/1.65 system-ui;max-width:1320px;margin:32px auto;padding:0 24px;color:#182635}img{width:100%;height:auto;margin:14px 0}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}code,pre{background:#eef2f5;white-space:pre-wrap;word-break:break-all}h2{margin-top:40px}a{color:#146da8}</style>',
      '<h1>27 张官方标注视角：稀疏 M1 + M2 / SGI</h1>',
      '<p>本轮输出是后续 SPC 训练的 SGI 初始化模型；未运行 M4/M5。全部新产物保存在独立 run，历史 56 视角结果仅作参考。</p>',
      ('<p>本报告使用 opacity_lr_fixed 修正版：修复原构造函数的不透明度学习率自初始化错误，恢复声明的默认值 0.05，SGI 与 RGB 参考均重新从 P1/P0 训练。几何、视频和七类教师缓存完全复用。父目录中不透明度未正常学习的首轮结果保留作诊断，不作为推荐初始化。</p>' if (run/'reused_inputs.json').exists() else ''),
      '<h2>训练协议</h2><p>实际官方六面记录选出 27 张训练图像，其余 29 张、6 张验证和 16 张测试图像均未进入训练或几何观测。沿用提供的 RMX 位姿，历史特征轨迹仅作为关联提案；P0 的全部坐标都由所选观测重新求解。</p>',
      '<p>原训练序号：'+', '.join(map(str,summary['training_indices']))+'</p><img src="../00_protocol/camera_distribution.png">',
      ('<p>此前 56 张训练视角的官方标注覆盖分布，供对应本轮所选与未选视角。</p><img src="camera_all56_annotation.png">' if previous_distribution.exists() else ''),
      '<h2>M1：真实观测与缓存生成视角</h2>',
      f"<p>P0 {summary['p0_points']:,} 点 → 新增 {summary['new_points']:,} 点 → P1 {summary['p1_points']:,} 点。复用 108 个视频、432 个早期候选帧；没有重新生成视频。四个方向均只按实际几何质量处理。</p>",
      '<p>每个源、每个方向最多选一帧，优先选择能通过原几何检查的新轨迹较多的候选。源按三维区域与高度层轮换处理。空间组点数不强行均衡；没有有效几何支持的区域不会补造点。</p><img src="geometry.png">',
      '<pre>'+html.escape(json.dumps(geometry['regions'],ensure_ascii=False,indent=2))+'</pre>',
      f"<p>以原世界坐标原点建立边长 0.2 的体素网格，占据格数由 {geometry['voxel_occupancy'][1]['P0_occupied']} 增至 {geometry['voxel_occupancy'][1]['P1_occupied']}；这只描述空间采样覆盖，不是有真值的表面完整率，也不参与点的筛选。</p>",
      '<p>P0 允许至少两张所选真实视角支持，并执行原重投影、视差、去重与密度检查。历史 3,979 个基础点采用不同的轨迹保留条件，因此点数不能单独解释为重建精度提高。新增点的误差是内部几何一致性指标，不等同于外部测量精度；点投影也不等同于可见性判断。</p>',
      '<h2>M2：七类官方原型监督</h2>',
      f"<p>冻结 CLIP ViT-B/16，每类三个英文提示取归一化均值构造 T（7×512）；F=T[label]，S=log(0.95·onehot+0.05/7)，W 由可靠标签有效性决定。语义有效率 {cache['mean_valid_fraction']:.2%}。这是类别原型监督，不是完整 CLIP 图像特征蒸馏。</p>",
      '<p>W 同时控制余弦损失和交叉熵，不影响 RGB。未确认的 SAM 补全不参与监督；生成帧只用于 M1，第一轮关闭 L_GS。保留 32 维语义、32→512 映射、局部三维正则、增密和 OGR。原生掩码检查验证了无效像素不产生语义梯度、全零掩码损失为零及与独立 PyTorch 公式一致。</p>',
      '<h2>训练与评估</h2><p>RGB 指标为各图 PSNR / SSIM 的算术平均。语义指标只衡量 27 张训练图上的官方掩码拟合，不能作为未见视角语义准确率。普通参考从 P0 开始，仅使用同样 27 张 RGB、同分辨率和 10,000 步；SGI 从 P1 开始，加入语义与 OGR。因此这一对比衡量组合流程，不单独归因于 Wan360。</p><p>原有透明度重置在 3000、6000 步执行，刚保存的对应检查点会暂时变暗；损失随后恢复。OGR 的移除数量与分布单独记录，最终比较使用恢复训练后的 10000 步模型。</p>',
      '<table><tr><th>模型</th><th>步数</th><th>高斯数</th><th>训练 / 未见 29 / 验证 / 测试 PSNR</th><th>训练语义 mIoU</th></tr>']
    for name,data in training.items():
        evaluation=data['evaluation'];groups=evaluation['groups'] if evaluation else {};fit=evaluation['official_training_fit'] if evaluation else None
        vals=' / '.join(f"{groups[s]['psnr']:.3f}" if s in groups else '-' for s in ['train','unseen_train29','val','test'])
        miou=f"{fit['miou']:.2%}" if fit else '-'
        label=name+' (5 representative training views)' if name=='short' else name
        body.append(f"<tr><td>{label}</td><td>{data['iterations']}</td><td>{data['model']['gaussians'] if data['model'] else '-'}</td><td>{vals}</td><td>{miou}</td></tr>")
    body+=['</table><img src="semantic_fit.png"><img src="losses.png"><img src="ogr.png"><img src="gaussian_distribution.png"><h2>固定视角对比</h2><p>所有语义图使用同一七类配色。黑色教师像素为 ignore；预测图黑色表示无高斯贡献。图像从原 RGB 分辨率缩放展示；语义训练/渲染分辨率为 256×128。</p>']
    for cam in representatives(protocol):body.append(f"<h3>{cam['original_train_index']}: {cam['name']}</h3><img src=\"{cam['name']}_comparison.jpg\">")
    body+=['<h2>未见训练视角检查</h2><p>下列图像没有参与几何观测、RGB 或语义训练。22、39 号位于主要标注相机覆盖之外；没有官方语义真值，不计算语义准确率。</p>']
    for path in sorted(out.glob('unseen_*.jpg')):body.append(f'<h3>{path.stem}</h3><img src="{path.name}">')
    body+=['<h2>逐项数据与模型</h2><ul>']
    for title,path in [('完整统计','summary.json'),('几何统计与逐源贡献','geometry.json'),('训练、RGB 和逐类语义统计','training.json'),('训练与标签快照','../00_protocol/manifest.json'),('P0 基础点云','../01_p0/points.ply'),('P1 增强点云','../01_sgi_sfm/enhanced.ply'),('SGI 完整模型','../02_sgi_gaussians/final/model.pt'),('SGI 高斯 PLY','../02_sgi_gaussians/final/point_cloud.ply'),('七类教师缓存','../03_semantic/cache/manifest.json'),('原生掩码检查','../03_semantic/mask_validation/results.json')]:body.append(f'<li><a href="{path}">{title}</a></li>')
    body+=['</ul><p>model.pt 保存高斯属性、32 维语义、映射/分类头、Adam 和随机状态。已用原生加载器重新加载并渲染；后续 SPC 应引用相同的七类 teacher 与本 run 的 27 视角 manifest，禁止默认切回全 56 视角或旧 20 类缓存。实际 SPC 优化尚未执行。</p></html>']
    (out/'report.html').write_text('\n'.join(body));print('REPORT',out/'report.html',flush=True);return summary
