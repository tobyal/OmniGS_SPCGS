"""Evaluation and delivery for the sparse27 SGI -> SPC experiment.

Reads completed models and held-out images only after optimization. No labels or
correspondences produced here are fed back into training.
"""
import argparse,json,html
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .common import read,write,ref,native,sha,cameras
from .pipeline import base_request

MODELS=['c0','c1','c2']
SPLITS=['train','unseen_train29','val','test']

def all_views(protocol):
    return [*protocol['train'],*[{**v,'split':'unseen_train29'} for v in protocol['unseen_train29']],*protocol['validation'],*protocol['test']]

def lpips_saved(run,model,gpu):
    import torch,lpips,cv2
    run=Path(run);cfg=read(run/'config.json');folder=run/'05_spc'/model/'renders_all'
    metrics=read(folder/'metrics.json');lookup={v['name']:v for v in all_views(read(run/'00_protocol/manifest.json'))}
    device=torch.device(f'cuda:{gpu}');network=lpips.LPIPS(net='alex',version='0.1').eval().to(device)
    with torch.inference_mode():
        for row in metrics['views']:
            if row.get('lpips') is not None:continue
            pred=np.fromfile(folder/(row['id']+'_rgb.f32'),dtype='f4').reshape(3,cfg['height'],cfg['width'])
            gt=cv2.cvtColor(cv2.imread(lookup[row['id']]['image']),cv2.COLOR_BGR2RGB)
            gt=cv2.resize(gt,(cfg['width'],cfg['height']),interpolation=cv2.INTER_AREA).astype('f4').transpose(2,0,1)/255
            row['lpips']=float(network(torch.from_numpy(pred.copy())[None].to(device)*2-1,torch.from_numpy(gt.copy())[None].to(device)*2-1).item())
            assert np.isfinite(row['lpips'])
    for split,group in metrics['groups'].items():group['lpips']=float(np.mean([r['lpips'] for r in metrics['views'] if r['split']==split]))
    metrics['lpips_protocol']='LPIPS v0.1 AlexNet; official pretrained backbone and learned linear weights; saved float RGB, full 2048x1024 ERP, [-1,1], no crop.'
    write(folder/'metrics.json',metrics);print(model,metrics['groups'],flush=True)

def query_saved(run,model,gpu):
    run=Path(run);cfg=read(run/'config.json');cache=read(run/'03_semantic/cache/manifest.json');protocol=read(run/'00_protocol/manifest.json')
    view=protocol['unseen_train29'][0];folder=run/'05_spc'/model;dest=folder/'text_query'
    # Reuse the frozen seven text vectors; query mode bypasses the learned
    # class calibration head and uses normalized mapped features versus text.
    req={**base_request(cfg),'gpu':gpu,'action':'render','archive':str(folder/'final/model.pt'),
        'query_text':cache['text'],'views':[view],'output':str(dest)}
    native(req,folder/'text_query_job')
    scores=np.fromfile(dest/(view['name']+'_logits.f16'),dtype='f2').astype('f4').reshape(7,128,256)
    assert np.isfinite(scores).all() and np.ptp(scores)>0
    valid=np.fromfile(dest/(view['name']+'_ids.i32'),dtype='i4').reshape(128,256)>=0;assert valid.any()
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,2,figsize=(16,15));axes[0,0].imshow(Image.open(view['image']));axes[0,0].set_title('Unseen captured RGB')
    palette=plt.get_cmap('viridis').copy();palette.set_bad('#eeeeee')
    for i,(ax,name) in enumerate(zip(axes.flat[1:],cfg['classes'])):
        im=ax.imshow(np.ma.array(scores[i],mask=~valid),vmin=float(scores[:,valid].min()),vmax=float(scores[:,valid].max()),cmap=palette);ax.set_title(name)
    for ax in axes.flat:ax.axis('off')
    fig.suptitle('Frozen seven-class text queries / gray = no contributing Gaussian')
    fig.colorbar(im,ax=list(axes.flat),shrink=.4,label='Text correlation logit / shared scale')
    fig.savefig(dest/'seven_text_queries.png',dpi=130,bbox_inches='tight');plt.close(fig)
    write(dest/'validation.json',{'status':'passed','archive':ref(folder/'final/model.pt'),'query_text':ref(cache['text']),
        'view':view['name'],'classes':cfg['classes'],'finite':True,'render_coverage':float(valid.mean()),'logit_range':[float(scores.min()),float(scores.max())],
        'note':'Independent native load; frozen seven text prototypes reused without running CLIP or constructing a new teacher.'})

def contact_sheet(event,target,title):
    event=Path(event);status=read(event/'outputs.json');font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    canvas=Image.new('RGB',(2304,836),'white');draw=ImageDraw.Draw(canvas)
    for i in range(3):
        draw.text((i*768+8,4),('Captured training RGB' if i==0 else f'Online pseudo {i}')+' / '+title,fill='black',font=font)
        canvas.paste(Image.open(event/f'{i}.png').resize((768,384)),(i*768,34))
        draw.text((i*768+8,422),f'Mask area={status["area_fractions"][i]:.4f} / '+('source' if i==0 else str(status['valid'][i-1])),fill='black',font=font)
        canvas.paste(Image.open(event/f'mask_{i}.png').resize((768,384)),(i*768,452))
    canvas.save(target,quality=94)

def saved_isp_check(event,logged_inter):
    """Verify saved coordinates and recompute inter from self-rendered tensors.

    These are diagnostic observations; they do not add a mask acceptance rule.
    """
    event=Path(event);status=read(event/'outputs.json');h,w=128,256
    masks=np.fromfile(event/'masks.u8',dtype='u1').reshape(3,h,w)>0
    same=[];source=None
    for i in range(3):
        im=Image.open(event/f'mask_{i}_binary.png')
        same.append(bool(np.array_equal(masks[i],np.array(im.resize((w,h),Image.Resampling.NEAREST))>0)))
        if i==0:source=np.array(im)>0
    assert all(same)
    px,py=status['prompt_xy'];sx,sy=status['rolled_prompt_xy']
    assert (px+status['horizontal_roll'])%source.shape[1]==sx and py==sy
    result={'mask_grid_exact_match':same,'prompt_in_source_mask':bool(source[py,px])}
    if source.any():
        yy,xx=np.nonzero(source);dx=np.minimum(abs(xx-px),source.shape[1]-abs(xx-px))
        result['prompt_distance_to_source_mask_px']=float(np.sqrt((dx**2+(yy-py)**2).min()))
    features=np.fromfile(event/'real_features.f16',dtype='f2').astype('f4').reshape(32,h,w)
    logits=np.fromfile(event/'real_logits.f16',dtype='f2').astype('f4').reshape(7,h,w);value=0.
    if masks[0].any():cls=np.bincount(logits[:,masks[0]].argmax(0),minlength=7).argmax()
    for i in range(2):
        if not status['valid'][i] or not masks[0].any():continue
        target=np.fromfile(event/f'pseudo_{i}_features.f16',dtype='f2').astype('f4').reshape(32,h,w)
        logit=np.fromfile(event/f'pseudo_{i}_logits.f16',dtype='f2').astype('f4').reshape(7,h,w)[:,masks[i+1]]
        maximum=logit.max(0);ce=np.mean(np.log(np.exp(logit-maximum).sum(0))+maximum-logit[cls])
        value+=float(np.linalg.norm(target[:,masks[i+1]].mean(1)-features[:,masks[0]].mean(1))+ce)
    result.update(inter_from_saved_float16=value,inter_logged=logged_inter,absolute_difference=abs(value-logged_inter))
    return result

def contributor_plot(event,target):
    """Visualize the saved alpha*T maxima without mapping old IDs to new models."""
    import cv2
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    event=Path(event);target=Path(target);masks=np.fromfile(event/'masks.u8',dtype='u1').reshape(3,128,256)>0
    fig,axes=plt.subplots(3,3,figsize=(18,10));sets=[];rows=[]
    for i,name in enumerate(['real','pseudo_0','pseudo_1']):
        ids=np.fromfile(event/(name+'_ids.i32'),dtype='i4').reshape(128,256)
        weights=np.fromfile(event/(name+'_weights.f32'),dtype='f4').reshape(128,256)
        padded=np.pad(np.pad(masks[i].astype('u1'),((0,0),(2,2)),mode='wrap'),((2,2),(0,0)),mode='constant')
        mask=cv2.erode(padded,np.ones((5,5),'u1'),borderType=cv2.BORDER_CONSTANT,borderValue=0)[2:-2,2:-2]>0
        mask &= ids>=0;chosen=set(map(int,np.unique(ids[mask])));sets.append(chosen)
        # Identical ID -> identical display color across all three frames.
        color_ids=(ids.astype('i8')*2654435761)&0xffffff
        codes=np.stack([(color_ids>>shift)&255 for shift in [0,8,16]],-1).astype('u1');codes[~mask]=0
        axes[0,i].imshow(Image.open(event/(name+'_rgb.png')));axes[0,i].set_title(name+' / model-rendered RGB')
        axes[1,i].imshow(codes,interpolation='nearest');axes[1,i].set_title(f'Eroded region: {mask.sum()} pixels / {len(chosen)} IDs')
        im=axes[2,i].imshow(weights,vmin=0,vmax=1,cmap='magma');axes[2,i].set_title('Maximum alpha*T per pixel')
        rows.append({'frame':name,'eroded_pixels':int(mask.sum()),'unique_ids':len(chosen),'shared_with_source':len(chosen & sets[0])})
    for ax in axes.flat:ax.axis('off')
    fig.suptitle('Actual renderer contributors / ID colors are identifiers, not semantic classes')
    fig.colorbar(im,ax=list(axes.flat),shrink=.4,label='alpha*T');fig.savefig(target,dpi=150,bbox_inches='tight');plt.close(fig)
    write(target.with_suffix('.json'),{'event':str(event),'frames':rows,
        'note':'IDs belong to this pre-update ISP snapshot. No old index is interpreted using a later densified/pruned checkpoint.'})

def audit(run):
    run=Path(run);cfg=read(run/'config.json');protocol=read(run/'00_protocol/manifest.json');allowed={v['name'] for v in protocol['train']}
    cache=read(run/'03_semantic/cache/manifest.json');assert len(allowed)==27 and len(cache['classes'])==7
    init=read(run/'initialization_validation.json');assert init['status']=='passed' and len(init['models'])==4
    upstream=read(run/'upstream.json')
    assert sha(cfg['sgi_archive'])==upstream['sgi']['sha256']
    assert sha(run/'03_semantic/cache/manifest.json')==upstream['teacher']['sha256']
    assert sha(run/'00_protocol/manifest.json')==upstream['protocol']['sha256']
    configs=[{k:v for k,v in read(run/'05_spc'/m/'config.json').items() if k!='gpu'} for m in MODELS]
    assert configs[0]==configs[1]==configs[2]
    optimizers=[read(run/'05_spc'/m/'optimizer_parameters.json') for m in MODELS]
    assert optimizers[0]==optimizers[1]==optimizers[2] and abs(optimizers[0]['opacity_lr']-0.05)<1e-8
    results={};orders=[]
    for model in MODELS:
        folder=run/'05_spc'/model;rows=[json.loads(x) for x in (folder/'losses.jsonl').read_text().splitlines()]
        request=read(folder/'request.json');assert [r['step'] for r in rows]==list(range(1,10001))
        execution=read(folder/'execution.json')
        assert execution['binary']['sha256']==read(run/'code_provenance.json')['native']['sha256']
        assert request['stage_reset'] and request['archive']==cfg['sgi_archive'] and request['iterations']==10000
        assert request['spc2d']==(model!='c0') and request['spc3d']==(model=='c2')
        assert {v['name'] for v in request['views']}==allowed and request['classes']==cache['classes']
        assert request['views']==cache['views'] and request['text']==cache['text'] and request['decoder']==cache['decoder']
        assert read(folder/'outputs.json')['status']=='completed'
        orders.append([r['view'] for r in rows]);events=[r for r in rows if r.get('isp_event')]
        assert len(events)==(0 if model=='c0' else 100)
        assert all(np.isfinite(v) for r in rows for v in r.values() if isinstance(v,(int,float)))
        assert all(r['generated_semantic']==0 for r in rows)
        error=max(abs(r['total']-(r['rgb']+r['clip']+r['generated_semantic']+cfg['local_weight']*r['local3d']+r['lambda']*(r['inter']+r['intra']+r['spc3d']))) for r in rows)
        assert error<1e-4
        for key in ['inter','intra']+(['spc3d'] if model=='c2' else []):
            if model!='c0':assert max(r[key] for r in events)>0 and max(r[key+'_semantic_grad_norm'] for r in events)>0
        if model!='c2':assert all(r['spc3d']==0 for r in rows)
        if model=='c0':assert all(r['inter']==r['intra']==0 for r in rows)
        details=[]
        for row in events:
            event=folder/'isp'/str(row['step']);s=read(event/'outputs.json');req=read(event/'request.json')
            assert s['backend']=='sam2_hiera_large' and s['trainable_parameters']==0
            assert s['checkpoint_sha256']==read(run/'sam2_provenance.json')['checkpoint']['sha256']
            assert req['cameras'][0]['name'] in allowed and s['valid']==[bool(x) for x in s['valid']]
            details.append({'step':row['step'],'source':row['view'],'basic_valid':s['valid'],'invalid_reasons':s['invalid_reasons'],
                'prompt_xy':s['prompt_xy'],'areas':s['area_fractions'],'gaussian_regions':read(event/'gaussian_regions.json')})
            if row['step'] in [100,1000,1500,4000,7000,10000]:details[-1]['saved_coordinate_and_inter_check']=saved_isp_check(event,row['inter'])
        modelmeta=read(folder/'final/model.json');assert modelmeta['step']==10000 and modelmeta['semantic_dim']==32
        assert np.array_equal(np.fromfile(folder/'final/text.f32',dtype='f4'),np.fromfile(cache['text'],dtype='f4'))
        modelmeta.update(classes=cfg['classes'],semantic_resolution=[cfg['semantic_width'],cfg['semantic_height']],
            training_ids=cfg['train_names'],source_sgi_sha256=upstream['sgi']['sha256'],stage_reset=True,
            spc2d=request['spc2d'],spc3d=request['spc3d'],generated_semantic_loss=False,
            semantic_teacher='Reused sparse27 official-label F/S/T/W and frozen seven-class CLIP text prototypes')
        write(folder/'final/model.json',modelmeta)
        evaluation=read(folder/'renders_all/metrics.json');assert len(evaluation['views'])==78
        results[model]={'steps':len(rows),'gaussians':modelmeta['gaussians'],'camera_counts':dict(Counter(orders[-1])),
            'archive':ref(folder/'final/model.pt'),'isp_events':len(events),'basic_valid_pairs':sum(r['isp_valid_pairs'] for r in rows),
            'native_execution':execution,
            'attempted_pairs':2*len(events),'nonzero_spc_steps':sum((r['inter']+r['intra']+r['spc3d'])>0 for r in rows),
            'nonzero_steps_by_term':{key:sum(r[key]>0 for r in rows) for key in ['inter','intra','spc3d']},
            'spc3d_nonzero_steps':sum(r['spc3d']>0 for r in rows),'total_formula_max_error':error,
            'loss_maxima':{key:max(r[key] for r in rows) for key in ['inter','intra','spc3d','local3d']},
            'gradient_maxima':{key:max((r.get(key+'_semantic_grad_norm',0) for r in rows)) for key in ['inter','intra','spc3d']},
            'sam_seconds':sum(r.get('sam_seconds',0) for r in rows),'events':details,
            'evaluation':evaluation,'text_query':read(folder/'text_query/validation.json')}
    assert orders[0]==orders[1]==orders[2]
    result={'status':'completed','initialization':init,'same_training_camera_sequence':True,'same_base_configuration':True,'same_optimizer_parameters':True,'source_sgi_unchanged':True,'upstream_manifests_unchanged':True,
        'models':results,'run':str(run.resolve()),'semantic_gt_note':'Training-label fit only; no held-out mIoU exists.',
        'reproducibility_note':'One scene and one seed. Initial RNG states and camera order match, but native CUDA optimization is not bitwise deterministic; observed differences do not establish statistical significance.',
        'sampling_note':'Basic mask validity is not correspondence accuracy; all failed events retained.'}
    write(run/'completion_audit.json',result);return result

def panels_and_curves(run):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run=Path(run);out=run/'07_report';out.mkdir(exist_ok=True);protocol=read(run/'00_protocol/manifest.json');cache=read(run/'03_semantic/cache/manifest.json')
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    # Fixed spatial examples selected without looking at model scores.
    original_index={v['name']:i+1 for i,v in enumerate(cameras('train'))}
    views=[{**v,'split':'unseen_train29','original_train_index':original_index[v['name']]} for v in protocol['unseen_train29'] if original_index[v['name']] in [3,22,39,55]]
    views += [protocol['validation'][i] for i in [0,3,5]]+[protocol['test'][i] for i in [0,7,15]]
    for cam in views:
        items=[('Captured RGB / '+cam['name'],Image.open(cam['image']).convert('RGB'))]
        items +=[(model.upper()+' semantic',Image.open(run/'05_spc'/model/'renders_all'/(cam['name']+'_semantic.png'))) for model in MODELS]
        items +=[(model.upper()+' RGB',Image.open(run/'05_spc'/model/'renders_all'/(cam['name']+'_rgb.png'))) for model in MODELS]
        canvas=Image.new('RGB',(1536,4*418+84),'white');draw=ImageDraw.Draw(canvas)
        for i,(label,img) in enumerate(items):
            x,y=(i%2)*768,(i//2)*418;draw.text((x+8,y+4),label,fill='black',font=font)
            canvas.paste(img.resize((768,384),Image.Resampling.NEAREST if 'semantic' in label else Image.Resampling.LANCZOS),(x,y+34))
        for i,c in enumerate(cache['class_config']['classes']):
            x=(i%4)*384;y=4*418+(i//4)*32;draw.rectangle((x+8,y+5,x+26,y+23),fill=tuple(c['color']));draw.text((x+34,y),c['name'],fill='black',font=font)
        canvas.save(out/(cam['name']+'_comparison.jpg'),quality=94)
    for model in MODELS:
        rows=[json.loads(x) for x in (run/'05_spc'/model/'losses.jsonl').read_text().splitlines()]
        fig,axes=plt.subplots(2,3,figsize=(16,8));ev=[r for r in rows if r.get('isp_event')]
        for ax,keys in zip(axes.flat,[['rgb','clip','local3d'],['inter','intra','spc3d'],['gaussians'],['inter_semantic_grad_norm','intra_semantic_grad_norm','spc3d_semantic_grad_norm'],['semantic_grad_norm','xyz_grad_norm'],['isp_valid_pairs']]):
            selected=ev if keys[0] in ['inter','inter_semantic_grad_norm','isp_valid_pairs'] else rows
            for key in keys:
                if not selected:continue
                y=np.array([r[key] for r in selected]);x=np.array([r['step'] for r in selected])
                if selected is rows and key!='gaussians':y=np.convolve(y,np.ones(50)/50,mode='valid');x=x[49:]
                ax.plot(x,y,label=key,linewidth=1)
            if selected:ax.legend(fontsize=8)
            ax.set_xlabel('Step');ax.grid(alpha=.2)
        fig.suptitle(model.upper()+' / SPC terms shown at ISP events; other losses: 50-step means');fig.tight_layout();fig.savefig(out/(model+'_curves.png'),dpi=150);plt.close(fig)
    for model in ['c1','c2']:
        for step in [100,1000,1500,4000,7000,10000]:
            event=run/'05_spc'/model/'isp'/str(step)
            if (event/'outputs.json').exists():contact_sheet(event,out/f'online_{model}_{step}.jpg',f'{model.upper()} step {step}')
    contributor_plot(run/'05_spc/c2/isp/7000',out/'gaussian_contributors_c2_7000.png')
    write(out/'representative_views.json',views)

def region_groups(regions):
    # Descriptive groups fixed by inspecting captured RGB before final metrics.
    # No pair is removed from the main evaluation because of this review.
    floor={'view01_seed16','view01_seed42','view14_seed42','view27_seed42','view48_seed42','view56_seed42'}
    common={tuple(x) for x in regions['common_pairs']};common3d={tuple(x) for x in regions['common_3d_pairs']};groups={}
    clear_parts={('view56_seed16',1),('view56_seed16',2),('view48_seed2',1),('view27_seed2',1),('view27_seed2',2)}
    selections={'floor_surface':{p for p in common if p[0] in floor},
        'other_regions_including_uncertain_correspondences':{p for p in common if p[0] not in floor},
        'visually_clear_parts_descriptive_only':common & clear_parts}
    for name,selection in selections.items():
        groups[name]={}
        for model in MODELS:
            rows=[r for r in regions['measurements'] if r['model']==model and (r['event'],r['pair']) in selection]
            gauss=[r['gaussian_semantic_KL'] for r in rows if (r['event'],r['pair']) in common3d]
            groups[name][model]={'pairs':len(rows),'pairs_3d':len(gauss),'gaussian_semantic_KL':float(np.mean(gauss)) if gauss else None,
                **{key:float(np.mean([r[key] for r in rows])) if rows else None for key in ['feature_cosine','region_class_distribution_KL','dominant_class_agrees']}}
    return groups

def region_panels(run):
    """Show the frozen object correspondences and every model on the same views."""
    run=Path(run);out=run/'07_report';font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    for sample in ['view56_seed16','view48_seed2','view27_seed2','view14_seed2']:
        event=run/'06_comparison/evaluation_correspondences'/sample
        cams=read(event/'request.json')['cameras'];status=read(event/'outputs.json')
        canvas=Image.new('RGB',(2304,1672),'white');draw=ImageDraw.Draw(canvas)
        for col,cam in enumerate(cams):
            mask=np.array(Image.open(event/f'mask_{col}_binary.png').resize((768,384),Image.Resampling.NEAREST))>0
            for row,model in enumerate([None,*MODELS]):
                path=Path(cam['image']) if model is None else run/'05_spc'/model/'renders_all'/(cam['name']+'_semantic.png')
                im=np.array(Image.open(path).convert('RGB').resize((768,384),Image.Resampling.LANCZOS if model is None else Image.Resampling.NEAREST))
                im[~mask]=(im[~mask].astype('f4')*.25).astype('u1')
                label=('Captured RGB' if model is None else model.upper()+' semantic')+' / '+('anchor' if col==0 else f'held-out {col}')
                if row==0:label+=' / '+('source' if col==0 else 'basic '+str(status['valid'][col-1]))
                x,y=col*768,row*418;draw.text((x+6,y+3),label,fill='black',font=font);canvas.paste(Image.fromarray(im),(x,y+34))
        canvas.save(out/(sample+'_region_comparison.jpg'),quality=95)

def report(run):
    run=Path(run);summary=audit(run);panels_and_curves(run);region_panels(run);out=run/'07_report';regions=read(run/'06_comparison/fixed_region_consistency/metrics.json')
    summary['region_consistency']=regions['common_pair_summary'];summary['descriptive_region_groups']=region_groups(regions)
    if (out/'findings.json').exists():summary['findings']=read(out/'findings.json')
    write(out/'summary.json',summary)
    body=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>SPC27 第二阶段结果</title>',
        '<style>body{font:17px/1.65 system-ui;max-width:1450px;margin:36px auto;padding:0 24px;color:#182635}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{padding:9px;border-bottom:1px solid #d5dce1;text-align:left}code{overflow-wrap:anywhere}a{color:#146da8}.note{background:#eef3f7;padding:16px}summary{cursor:pointer}</style>',
        '<h1>SGI → SPC：27 张工业全景的第二阶段训练</h1>',
        '<nav><a href="#gallery">直接看 78 视角对比</a> · <a href="#objects">同对象跨视角对比</a> · <a href="#artifacts">模型与日志</a></nav>',
        '<p>C0、C1、C2 各独立训练 10,000 步，均从修复 opacity 学习率后的 287,787 个 SGI 高斯开始。没有重跑 M1/M2、Wan360 或 CLIP 教师。</p>',
        '<p>三组初始位置、DC 颜色、32 维语义及映射参数逐张量完全一致；尺度重新初始化，旋转为单位四元数，透明度为 0.1，高阶 SH 与优化器状态重置。初始随机状态与 27 张训练相机的实际抽样顺序也完全一致。原生 CUDA 优化并非逐位确定，本轮是单场景、单种子的观察性比较，微小差值不能视作统计显著提升。</p>',
        '<p>正式 M4 使用原始 SAM2 Hiera-L，224,430,130 个冻结参数。真实 RGB 与两个当前模型在线渲染伪视角组成序列，循环平移后分割并映回 ERP。面积检查只判断掩码是否可用，不能作为同一物体对应准确率。</p>',
        '<p class="note">下述跨视角指标是固定 SAM 区域上的一致性代理；共同预测错误的类别也可能获得较好一致性。没有未见视角的独立七类语义真值，不计算其 mIoU。训练标签拟合率仅作诊断。</p>',
        '<h2>训练与实际 SPC 执行</h2><table><tr><th>模型</th><th>步数 / 高斯数</th><th>ISP / 基本有效对应</th><th>非零 3D SPC 步数</th><th>最大语义梯度 inter / intra / 3D</th></tr>']
    findings=out/'findings.json'
    if findings.exists():
        answers=['<h2>三个验收问题的结论</h2>']
        for item in read(findings)['answers']:answers.append('<p><b>'+html.escape(item['question'])+'</b><br>'+html.escape(item['answer'])+'</p>')
        body[-1:-1]=answers
    for model,d in summary['models'].items():
        body.append(f'<tr><td>{model.upper()}</td><td>{d["steps"]} / {d["gaussians"]:,}</td><td>{d["isp_events"]} / {d["basic_valid_pairs"]} of {d["attempted_pairs"]}</td><td>{d["spc3d_nonzero_steps"]}</td><td>'+ ' / '.join(f'{v:.5g}' for v in d['gradient_maxima'].values())+'</td></tr>')
    body += ['</table><p>L_GS=0；局部 3D 正则与区域 3D SPC 分别记录。SPC 权重在前 4,000 步从 0 增至 1；每 100 步实际计算一次 ISP/SPC。本实现保留采样式近似：128×128 语义窗口最多 512 对比样本、128 个局部 3D 锚点，区域 KL 使用归一化平均。它是本工程适配版本，不宣称论文数值复现。</p>',
        '<h2>固定区域跨视角一致性</h2><p>5 个空间锚点、每点 3 个固定随机提示、各 2 个邻近未见真实视角，共 30 对候选。掩码在最终模型产生前固定，独立于训练。三组仅在共同可用的相同对应对上比较；腐蚀后的 3D 指标另取共同集合。</p>',
        '<table><tr><th>模型</th><th>2D / 3D 对数</th><th>特征余弦 ↑</th><th>类别分布 KL ↓</th><th>主导类别一致率 ↑</th><th>高斯语义 KL ↓</th></tr>']
    for model,d in regions['common_pair_summary'].items():
        values=[d['feature_cosine'],d['region_class_distribution_KL'],d['dominant_class_agrees'],d['gaussian_semantic_KL']]
        body.append(f'<tr><td>{model.upper()}</td><td>{d["common_pairs"]} / {d["common_3d_pairs"]}</td>'+''.join(f'<td>{v:.6f}</td>' if v is not None else '<td>无可用值</td>' for v in values)+'</tr>')
    body.append('</table><p>共同区域的贡献覆盖率与平均类别分布熵（源 / 目标；熵不是准确率）：</p><ul>')
    for model,d in regions['common_pair_summary'].items():body.append(f'<li>{model.upper()}：覆盖率 {d["real_coverage"]:.2%} / {d["target_coverage"]:.2%}；熵 {d["real_entropy"]:.4f} / {d["target_entropy"]:.4f}。</li>')
    body+=['</ul><h3>地面与其余区域分开观察</h3><p>按最终模型评估前的 RGB 图像检查划分描述组，未据此过滤主统计。其余区域包括有身份歧义或对象漂移的掩码。另列设备柜、爬梯、竖直管体的 5 对目视较清楚对应，作为其余区域中的重叠子集；这也不是人工标注真值或对象准确率。</p><table><tr><th>区域组</th><th>模型</th><th>2D / 3D 对数</th><th>余弦 ↑</th><th>类别 KL ↓</th><th>主导类别一致 ↑</th><th>高斯 KL ↓</th></tr>']
    for group,models in summary['descriptive_region_groups'].items():
        for model,d in models.items():
            values=[d['feature_cosine'],d['region_class_distribution_KL'],d['dominant_class_agrees'],d['gaussian_semantic_KL']]
            label={'floor_surface':'地面','other_regions_including_uncertain_correspondences':'其余区域','visually_clear_parts_descriptive_only':'目视较清楚的设备/梯架'}[group]
            body.append(f'<tr><td>{label}</td><td>{model.upper()}</td><td>{d["pairs"]} / {d["pairs_3d"]}</td>'+''.join(f'<td>{v:.6f}</td>' if v is not None else '<td>无可用值</td>' for v in values)+'</tr>')
    body+=['</table><h2>RGB 与训练标签拟合</h2><table><tr><th>模型</th><th>视角组 / 数量</th><th>PSNR ↑</th><th>SSIM ↑</th><th>LPIPS ↓</th></tr>']
    for model,d in summary['models'].items():
        for split in SPLITS:
            g=d['evaluation']['groups'][split];body.append(f'<tr><td>{model.upper()}</td><td>{split} / {g["views"]}</td><td>{g["psnr"]:.4f}</td><td>{g["ssim"]:.5f}</td><td>{g["lpips"]:.5f}</td></tr>')
    body+=['</table><p>RGB 指标为每幅 2048×1024 图像指标的算术平均；LPIPS v0.1 AlexNet，完整 ERP。语义网格为 256×128，细管线及栏杆的边界分辨率受此限制。</p>']
    for model,d in summary['models'].items():body.append(f'<p>{model.upper()}：27 张训练图有效官方像素 mIoU {d["evaluation"]["official_training_fit"]["miou"]:.4%}（仅训练拟合）。</p>')
    notes=out/'visual_review.json'
    if notes.exists():
        body.append('<h2>图像检查记录</h2><p>下列为助手逐图检查的定性观察，不是人工标注真值，也未用于训练或选择定量样本。</p>')
        for entry in read(notes)['observations']:
            body.append('<p><b>'+html.escape(entry['sample'])+'</b>：'+html.escape(entry['observation'])+'</p>')
    for path in sorted(out.glob('eval_anchor_*.jpg')):body.append(f'<details><summary>固定评估样例：{path.stem}</summary><a href="{path.name}"><img loading="lazy" src="{path.name}"></a></details>')
    body.append('<h2 id="objects">相同对象区域的跨视角预测</h2><p>每列为同一张真实拍摄图像，每行分别显示 RGB、C0、C1、C2。掩码外仅作显示变暗；没有据此更改预测或评价。设备柜、竖管和梯架是预先选定的描述样例；view14_seed2 保留灯具传播到保温管的失败案例。</p>')
    for path in sorted(out.glob('*_region_comparison.jpg')):body.append(f'<details><summary>{path.stem}</summary><a href="{path.name}"><img loading="lazy" src="{path.name}"></a></details>')
    body.append('<h2>实际在线 M4：成功与失败样例</h2>')
    for path in sorted(out.glob('online_*.jpg')):body.append(f'<details><summary>{path.stem}</summary><a href="{path.name}"><img loading="lazy" src="{path.name}"></a></details>')
    body.append('<h2 id="gallery">全部真实视角浏览</h2><p>可切换全部 78 张真实视角。训练 27 张、未见 29 张、验证 6 张、测试 16 张分别标明；以下 RGB 为实际拍摄图像。</p>')
    gallery_views=all_views(read(run/'00_protocol/manifest.json'))
    gallery_views=[v for v in gallery_views if v['split']!='train']+[v for v in gallery_views if v['split']=='train']
    gallery=out/'captured_rgb';gallery.mkdir(exist_ok=True)
    for cam in gallery_views:
        Image.open(cam['image']).convert('RGB').resize((1024,512),Image.Resampling.LANCZOS).save(gallery/(cam['name']+'.jpg'),quality=92)
    body.append('<select id="camera" style="font:inherit;padding:8px">'+''.join('<option value="'+html.escape(v['name'])+'">'+html.escape(v['split']+' / '+v['name'])+'</option>' for v in gallery_views)+'</select>')
    classes=read(run/'03_semantic/cache/manifest.json')['class_config']['classes']
    body.append('<p>'+''.join('<span style="display:inline-block;margin:4px 14px 4px 0"><i style="display:inline-block;width:16px;height:16px;vertical-align:middle;background:rgb('+','.join(map(str,c['color']))+')"></i> '+html.escape(c.get('zh',c['name']))+'</span>' for c in classes)+'</p>')
    body.append('<div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px"><div>拍摄 RGB<img id="captured"></div>'+''.join(f'<div>{m.upper()} 语义<img id="{m}_semantic"></div>' for m in MODELS)+'</div>')
    body.append('<script>function showCamera(){const n=document.getElementById("camera").value;document.getElementById("captured").src="captured_rgb/"+n+".jpg";for(const m of ["c0","c1","c2"])document.getElementById(m+"_semantic").src="../05_spc/"+m+"/renders_all/"+n+"_semantic.png";}document.getElementById("camera").addEventListener("change",showCamera);showCamera();</script>')
    body.append('<h2>相同未见视角的语义和 RGB</h2><p>黑色语义像素表示无高斯贡献；三组配色一致。</p>')
    for cam in read(out/'representative_views.json'):
        name=cam['name']+'_comparison.jpg';body.append(f'<details><summary>{cam["split"]}: {cam["name"]}</summary><a href="{name}"><img loading="lazy" src="{name}"></a></details>')
    body.append('<h2>损失与梯度</h2>')
    body.append('<p>下图为同一次 ISP 保存的最大 alpha×T 贡献索引与权重。彩色仅编码高斯 ID，非语义类别；没有用后续增密后的模型解释旧索引。</p><img loading="lazy" src="gaussian_contributors_c2_7000.png">')
    for model in MODELS:body.append(f'<img loading="lazy" src="{model}_curves.png">')
    body.append('<h2 id="artifacts">模型与可复核记录</h2><ul>')
    for label,path in [('完整统计','summary.json'),('初始化逐张量核验','../initialization_validation.json'),('第二阶段初始完整模型','../03_stage_initialization/initial/model.pt'),('初始 RGB 与语义渲染','../03_stage_initialization/renders_initial/'),('完整训练核验','../completion_audit.json'),('原生数值与梯度测试','../native_checks/native_gradient_checks.json'),('固定区域逐对测量','../06_comparison/fixed_region_consistency/metrics.json'),('评估掩码清单','../06_comparison/evaluation_correspondences/manifest.json'),('配置','../config.json'),('SAM2 来源','../sam2_provenance.json')]:body.append(f'<li><a href="{path}">{label}</a></li>')
    for model in MODELS:
        for label,path in [('完整模型','final/model.pt'),('模型元数据','final/model.json'),('PLY 与语义侧文件','final/'),('运行请求与配置','request.json'),('损失逐步记录','losses.jsonl'),('逐视角指标','renders_all/metrics.json'),('独立文本查询','text_query/seven_text_queries.png')]:body.append(f'<li><a href="../05_spc/{model}/{path}">{model.upper()} {label}</a></li>')
        if model!='c0':body.append(f'<li><a href="../05_spc/{model}/isp/">{model.upper()} 全部在线对应、相机与贡献高斯</a></li>')
    if (out/'evaluation_source_supervision.json').exists():body.append('<li><a href="evaluation_source_supervision.json">源区域官方监督覆盖率：评估后的只读诊断</a></li>')
    body+=['</ul><p>模型同时包含高斯、语义、映射头、优化器与随机状态。每组均已用独立原生进程重新加载并渲染、查询七类冻结文本向量。</p>', '<p>方法参考：<a href="https://arxiv.org/abs/2503.12535">SPC-GS</a>；<a href="https://github.com/facebookresearch/sam2">Meta SAM2</a>。</p></html>']
    (out/'report.html').write_text('\n'.join(body),encoding='utf-8');print(out/'report.html',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--action',choices=['lpips','query','report','audit','panels'],required=True);p.add_argument('--model',choices=MODELS,default='c2');p.add_argument('--gpu',type=int,default=0);a=p.parse_args()
    if a.action=='lpips':lpips_saved(a.run,a.model,a.gpu)
    elif a.action=='query':query_saved(a.run,a.model,a.gpu)
    elif a.action=='audit':audit(a.run)
    elif a.action=='panels':panels_and_curves(a.run)
    else:report(a.run)
