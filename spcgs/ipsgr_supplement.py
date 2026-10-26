"""Measured supplementary artifacts and acceptance evidence for IPSGR20/6."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .common import read, write, ref, native
from .ipsgr_protocol import protocol, MODEL_NAMES
from .pipeline import base_request


def resource_figure(run):
    """Actual video allocation and training intervals, with no inferred video timestamps."""
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .ipsgr_report import register,table
    from .ipsgr_training import model_folder
    generation=read(run/'01_geometry/Wan360/fresh_generation_audit.json')
    wan=read(run/'01_geometry/Wan360/parallel_completed.json');rows=[]
    for name in ['B-RGB','B-SEM','G-AUG','ISLB','smoke-IPSGR-Full','L-BOOT','CV-2D','IPSGR-Full']:
        folder=model_folder(run,name);out=read(folder/'outputs.json');req=read(folder/'fixed_request.json')
        assert out['status']=='completed' and not out['resumed']
        rows.append({'model':name,'GPU':req['gpu'],'started_unix':out['started'],'finished_unix':out['finished'],'seconds':out['seconds_this_invocation'],'steps':out['iterations']})
    table(run,'S_training_intervals',rows,{'note':'Measured per-stage intervals; shared GPUs; Wan360 per-video absolute start times were not logged and are not inferred.'})
    fig,axes=plt.subplots(1,2,figsize=(16,5),gridspec_kw={'width_ratios':[1,2]})
    counts=[generation['GPU_jobs'].get(str(i),0) for i in range(3)]
    axes[0].bar(['GPU0','GPU1','GPU2'],counts,color=['#4c83b7','#7eac8f','#d6a04b'])
    for i,n in enumerate(counts):axes[0].text(i,n+.5,str(n),ha='center')
    axes[0].set(ylabel='Completed fresh videos',ylim=(0,max(counts)+7),title=f'Wan360: 80/80 complete\nWall time {wan["seconds_since_first_single_gpu_launch"]/60:.1f} min')
    axes[0].text(.5,-.19,'One generation worker per GPU; GPU3 ran baselines.',ha='center',transform=axes[0].transAxes,fontsize=9)
    origin=min(r['started_unix'] for r in rows)
    for i,r in enumerate(rows):
        start=(r['started_unix']-origin)/60;length=(r['finished_unix']-r['started_unix'])/60
        axes[1].barh(i,length,left=start,color=plt.get_cmap('tab10')(r['GPU']),height=.65)
        axes[1].text(start+length/2,i,f'GPU{r["GPU"]}',ha='center',va='center',fontsize=8,color='white')
    axes[1].set_yticks(range(len(rows)),[r['model'] for r in rows]);axes[1].invert_yaxis();axes[1].set(xlabel='Minutes since first model training started',title='Independent model training / actual intervals');axes[1].grid(axis='x',alpha=.2)
    fig.tight_layout();paths=[]
    for ext in ['png','pdf']:
        p=run/'08_paper_figures'/f'S_resource_execution.{ext}';fig.savefig(p,dpi=220,bbox_inches='tight');paths.append(p)
    plt.close(fig)
    register(run,'S_resource_execution',paths,[run/'01_geometry/Wan360/fresh_generation_audit.json',run/'09_paper_tables/S_training_intervals.json'],'Operational resource records on shared GPUs; not a controlled hardware speedup benchmark.')
    return paths


def measured_geometry_projection(run):
    from .ipsgr_report import panel
    m=protocol(run);p0=read(run/'01_geometry/P0/observations.json');new=read(run/'01_geometry/P1/observations.json');items=[];counts=[]
    for cid in ['1-24','1-62','1-70']:
        v=next(v for v in m['train'] if v['camera_id']==cid);im=Image.open(v['image']).convert('RGB');a=im.copy();b=im.copy();da=ImageDraw.Draw(a);db=ImageDraw.Draw(b);numbers=[]
        for points,key,color,drawers in [(p0,'observations',(40,130,255),[da,db]),(new,'real_observations',(255,55,35),[db])]:
            num=0
            for point in points:
                for o in point[key]:
                    if o['view']!=v['name']:continue
                    x,y=o['uv'];num+=1
                    for d in drawers:d.ellipse((x-3,y-3,x+3,y+3),fill=color)
            numbers.append(num)
        items.extend([(cid+' / captured training RGB',im,False),(f'P0 / {numbers[0]} measured observations',a,False),(f'P1 / {numbers[0]} original + {numbers[1]} new',b,False)]);counts.append({'camera_id':cid,'P0_observations':numbers[0],'new_observations':numbers[1]})
    path=run/'08_paper_figures/F06_measured_training_observations.png';panel(path,items,3,tw=800,th=400)
    write(path.with_suffix('.json'),{'counts':counts,'source_P0':ref(run/'01_geometry/P0/observations.json'),'source_new':ref(run/'01_geometry/P1/observations.json'),'script':ref(__file__),'note':'Only feature observations actually used in each train camera are marked: blue=P0, red=new. These are sparse geometric support, not dense visibility or surface ground truth.'});return path


def layout_previews(run, gpu=3):
    from .ipsgr_evaluate import colors
    from .ipsgr_report import panel,semantic_legend
    m=protocol(run);cfg=read(run/'experiment_config.json');cfg['gpu']=gpu
    folder=run/'03_layout_bootstrap';assert read(folder/'outputs.json')['status']=='completed'
    ids=['1-24','1-62','1-70']
    views=[next(v for v in m['train'] if v['camera_id']==x) for x in ids]
    palette=np.array([c['color'] for c in m['classes']['classes']],dtype='u1');items=[];evidence={}
    for stage in ['initial','final']:
        dest=folder/'representative_train_renders'/stage
        if not all((dest/(v['name']+'_logits.f16')).exists() for v in views):
            native({**base_request(cfg),'action':'render','archive':str(folder/stage/'model.pt'),'views':views,'output':str(dest),'render_semantics':True},dest/'job')
        evidence[stage]={'checkpoint':ref(folder/stage/'model.pt'),'render_directory':str(dest)}
    for v in views:
        items.append((v['camera_id']+' / captured training RGB',v['image'],False))
        for stage in ['initial','final']:
            dest=folder/'representative_train_renders'/stage
            items.append(('ISLB '+stage+' / RGB',dest/(v['name']+'_rgb.png'),False))
        items.append((v['camera_id']+' / fixed label',colors(np.array(Image.open(v['label'])),palette),True))
        for stage in ['initial','final']:
            dest=folder/'representative_train_renders'/stage
            logits=np.fromfile(dest/(v['name']+'_logits.f16'),dtype='f2').reshape(7,128,256)
            indices=np.fromfile(dest/(v['name']+'_ids.i32'),dtype='i4').reshape(128,256)
            pred=logits.argmax(0).astype('u1');pred[indices<0]=7
            colored=colors(pred,palette);colored.save(dest/(v['name']+'_semantic.png'))
            items.append(('ISLB '+stage+' / semantics',colored,True))
    target=run/'08_paper_figures/F06_ISLB_training_previews.png';panel(target,items,3);semantic_legend(target,m)
    write(folder/'representative_train_renders/evidence.json',{'camera_ids':ids,'train_only':True,'stages':evidence,'figure':str(target),'script':ref(__file__),'note':'Training-fit visualization only; not held-out performance.'})
    return target


def isp_evidence(run):
    summary={};m=protocol(run)
    labels={v['name']:np.array(Image.open(v['label']).resize((256,128),Image.Resampling.NEAREST)) for v in m['train']}
    for name in ['CV-2D','IPSGR-Full']:
        folder=run/'05_models'/name
        events=[json.loads(x) for x in (folder/'losses.jsonl').read_text().splitlines() if json.loads(x).get('isp_event')]
        assert len(events)==100 and [r['step'] for r in events]==list(range(100,10001,100))
        rows=[]
        for r in events:
            event=folder/'isp'/str(r['step']);s=read(event/'outputs.json')
            assert s['status']=='completed' and s['backend']=='sam2_hiera_large' and s['parameters']==224430130 and s['trainable_parameters']==0
            required=['0.png','1.png','2.png','masks.u8','regions.i32','automatic_masks.npz','automatic_masks.json']
            required += [f'{v}_{f}' for v in ['real','pseudo_0','pseudo_1'] for f in ['features.f16','logits.f16','ids.i32','weights.f32']]
            assert all((event/f).exists() for f in required),(name,r['step'])
            with np.load(event/'automatic_masks.npz') as z:assert int(z['shape'][0])==s['automatic_masks']
            source=read(event/'request.json')['cameras'][0]['name'];mask=np.fromfile(event/'masks.u8',dtype='u1').reshape(3,128,256)[0]>0
            hist=np.bincount(labels[source][mask],minlength=7);area=int(mask.sum())
            logits=np.fromfile(event/'real_logits.f16',dtype='f2').reshape(7,128,256)
            predicted=int(np.bincount(logits.argmax(0)[mask],minlength=7).argmax()) if area else None
            regions=read(event/'gaussian_regions.json');eroded_pair=regions['real']['eroded_contributing_pixels']>0 and any(s['valid'][i] and regions['pseudo'][i]['eroded_contributing_pixels']>0 for i in range(2))
            rows.append({'step':r['step'],'valid_pairs':sum(s['valid']),'invalid_reasons':s['invalid_reasons'],'automatic_masks':s['automatic_masks'],'sam_seconds':s['seconds'],'checkpoint_sha256':s['checkpoint_sha256'],'worker_sha256':s['worker_sha256'],'nonzero_gradients':{k:r.get(k,0)>0 for k in ['inter_semantic_grad_norm','intra_semantic_grad_norm','spc3d_semantic_grad_norm']},'source_training_view':source,'source_mask_pixels':area,'source_GT_dominant_class':int(hist.argmax()) if area else None,'source_GT_class_purity':float(hist.max()/area) if area else None,'source_prediction_dominant_class':predicted,'source_eroded_contributing_pixels':regions['real']['eroded_contributing_pixels'],'any_valid_eroded_pair':bool(eroded_pair),'directory':str(event)})
        assert len(set(r['checkpoint_sha256'] for r in rows))==1
        summary[name]={'events':len(rows),'valid_pairs':sum(r['valid_pairs'] for r in rows),'possible_pairs':200,'rejected_pairs':sum(2-r['valid_pairs'] for r in rows),'automatic_masks':sum(r['automatic_masks'] for r in rows),'sam_seconds':sum(r['sam_seconds'] for r in rows),'gradient_events':{k:sum(r['nonzero_gradients'][k] for r in rows) for k in rows[0]['nonzero_gradients']},'empty_source_masks':sum(r['source_mask_pixels']==0 for r in rows),'source_GT_dominant_counts':[sum(r['source_GT_dominant_class']==k for r in rows) for k in range(7)],'source_GT_purity_median':float(np.median([r['source_GT_class_purity'] for r in rows if r['source_GT_class_purity'] is not None])),'valid_event_without_eroded_pair':sum(r['valid_pairs']>0 and not r['any_valid_eroded_pair'] for r in rows),'evidence':rows}
    write(run/'04_cross_view/all_event_audit.json',summary);return summary


def report_details(run,results,p1,roi,cost):
    from .ipsgr_report import table
    m=protocol(run);classes=[c['name'] for c in m['classes']['classes']]
    fmt=lambda x:'—' if x is None else f'{x:.4f}'
    lines=['','## 几何与训练的逐项核验',f'320个候选帧中，{p1["registered_candidates"]}个通过注册和帧质量检查；每个视频至多选择一帧，最终选择{p1["selected_views"]}个。P1新增点的最大观测重投影误差中位数为{p1["reprojection_px"]["median"]:.4f}px，P90为{p1["reprojection_px"]["p90"]:.4f}px。该指标与P0的逐观测误差定义不同，不直接作为优劣比较。','', '|生成方向|最终新增点数|','|---|---:|']
    for direction in ['forward','backward','left','right']:lines.append(f'|{direction}|{p1["by_direction"].get(direction,0)}|')
    source_rows=[{'camera_id':v['camera_id'],'image_id':v['name'],'added_points':p1['by_source'].get(v['name'],0)} for v in m['train']]
    table(run,'S_geometry_sources',source_rows)
    support=read(run/'07_industrial_analysis/geometry_semantic_support.json')['data']
    table(run,'S_geometry_class_support',[{'set':name,**dict(zip(classes,data['class_counts'])),'any_synthetic_RGB_observation_center':data['any_synthetic_RGB_observation'],'all_synthetic_RGB_observation_centers':data['all_synthetic_RGB_observations']} for name,data in support.items()])
    lines += ['', '在真实训练观测坐标上采样固定标签并取多数票，新增点的类别支持数为：'+', '.join(f'{c}={n}' for c,n in zip(classes,support['P1_added']['class_counts']))+'。这是稀疏观测支持统计，不是三维类别真值，也不参与几何过滤。',f'P0中有{support["P0"]["any_synthetic_RGB_observation"]}个点的部分观测中心落入RGB修复区域；新增点中为{support["P1_added"]["any_synthetic_RGB_observation"]}。两者完全由修复区域观测中心支持的点数均为0。该检查针对特征中心，不保证整个特征描述子窗口都位于未修复区域。']
    islb=read(run/'03_layout_bootstrap/outputs.json');loss=[json.loads(x) for x in (run/'03_layout_bootstrap/losses.jsonl').read_text().splitlines()]
    ogr=[{'step':r['step'],'removed':r['ogr_removed']} for r in loss if 'ogr_removed' in r]
    lines += ['',f'ISLB完成{islb["iterations"]}步，最终{islb["gaussians"]:,}个高斯；OGR记录：'+json.dumps(ogr,ensure_ascii=False)+'。初始/最终三张代表性训练视角图只用于检查拟合，不作为测试指标。','']
    evidence=isp_evidence(run)
    cfg=read(run/'experiment_config.json')
    lines += [f'实际工程预算固定为：RGB {cfg["width"]}×{cfg["height"]}、语义{cfg["semantic_width"]}×{cfg["semantic_height"]}；局部3D正则每步采样{cfg["local_samples"]}个高斯并查找5个近邻；Intra在128×128循环补丁内最多取{cfg["contrast_samples"]}个有效像素，温度0.2。跨视角3D KL使用32维高斯语义经Softmax得到的分布。Wan360帧只提供几何，没有额外生成图像语义损失；这些是本轮代码实际配置，不等同于原论文所有计算预算。']
    lines += ['三份正式第二阶段模型采用相同种子、初始张量、初始随机数状态和训练相机序列。GPU计算并非逐位确定性；首个ISP之前的损失差异另存pre_isp_comparability.json。100步短训只检查执行与梯度，其位置学习率衰减周期为100步，不能视为正式10k训练的前100步消融结果。']
    for name,data in evidence.items():
        g=data['gradient_events'];lines.append(f'- {name}：实际调用Hiera-L {data["events"]}次；{data["valid_pairs"]}/200个对应通过既定面积检查，拒绝{data["rejected_pairs"]}个；Inter/Intra/3D非零语义梯度事件分别为{g["inter_semantic_grad_norm"]}/{g["intra_semantic_grad_norm"]}/{g["spc3d_semantic_grad_norm"]}。这是掩码有效性记录，不是对应准确率。')
        lines.append(f'该模型源掩码在训练GT上的主导类别次数为：'+', '.join(f'{c}={n}' for c,n in zip(classes,data['source_GT_dominant_counts']))+f'；空源掩码{data["empty_source_masks"]}次，掩码主导类别纯度中位数{data["source_GT_purity_median"]:.3f}。已有面积有效对应但5×5腐蚀后没有可用区域对的事件有{data["valid_event_without_eroded_pair"]}次。这些统计只分析已保存掩码，不改变监督或筛选。')
    table(run,'S_ISP_source_regions',[{'model':name,'class':c,'source_GT_dominant_events':data['source_GT_dominant_counts'][k]} for name,data in evidence.items() for k,c in enumerate(classes)])
    lines += ['', '## 七类测试 IoU', '', '|模型|'+'|'.join(classes)+'|','|---|'+'---:|'*7]
    class_rows=[]
    for name in MODEL_NAMES[1:]:
        vals=results[name]['semantic_pooled']['iou'];lines.append('|'+name+'|'+'|'.join(fmt(x) for x in vals)+'|');class_rows.append({'model':name,**dict(zip(classes,vals))})
    table(run,'S_per_class_IoU',class_rows)
    best_sem=max(MODEL_NAMES[1:],key=lambda n:results[n]['semantic_pooled']['miou'])
    best_rgb=max(MODEL_NAMES,key=lambda n:results[n]['rgb_mean']['psnr'])
    best_lpips=min(MODEL_NAMES,key=lambda n:results[n]['rgb_mean']['lpips'])
    full=results['IPSGR-Full'];baseline=results['B-SEM']
    lines += ['',f'本轮主语义mIoU最高的是{best_sem}（{100*results[best_sem]["semantic_pooled"]["miou"]:.2f}%）；平均PSNR最高的是{best_rgb}（{results[best_rgb]["rgb_mean"]["psnr"]:.3f}dB）；LPIPS最低的是{best_lpips}（{results[best_lpips]["rgb_mean"]["lpips"]:.4f}）。不同指标分别评价，不能将某一项最高等同于所有目标最优。',f'完整IPSGR相对B-SEM的主语义mIoU变化为{100*(full["semantic_pooled"]["miou"]-baseline["semantic_pooled"]["miou"]):+.2f}个百分点；piping、equipment、structural_frame的IoU变化依次为'+', '.join(f'{100*(full["semantic_pooled"]["iou"][k]-baseline["semantic_pooled"]["iou"][k]):+.2f}个百分点' for k in [0,1,3])+'。这些是单场景单种子实测差值，未进行显著性检验。']
    lines += ['', '## IPSGR-Full 六张测试图的逐图指标','','|相机|PSNR|SSIM|LPIPS|mIoU|mAcc|预测覆盖率|','|---|---:|---:|---:|---:|---:|---:|']
    view_rows=[]
    for name,result in results.items():
        for v in result['views']:
            sem=v.get('semantic',{});row={'model':name,'camera_id':v['camera_id'],**{k:v[k] for k in ['psnr','ssim','lpips']},**{k:sem.get(k) for k in ['miou','macc','coverage']}}
            if sem:row.update({c+'_IoU':sem['iou'][k] for k,c in enumerate(classes)})
            view_rows.append(row)
            if name=='IPSGR-Full':lines.append('|'+v['camera_id']+'|'+'|'.join(fmt(row[k]) for k in ['psnr','ssim','lpips','miou','macc','coverage'])+'|')
    table(run,'S_all_view_metrics',view_rows)
    lines += ['', '其余五组逐图 RGB、语义和七类 IoU 见 S_all_view_metrics.csv；所有逐图召回率和7×8混淆矩阵见06_evaluation各模型metrics.json。','', '## 工业局部的实测结果', '', '|ROI类别|'+'|'.join(MODEL_NAMES[1:])+'|','|---|'+'---:|'*5]
    categories=list(dict.fromkeys(r['category'] for r in roi))
    for cat in categories:lines.append('|'+cat+'|'+'|'.join(fmt(float(np.mean([r['mIoU'] for r in roi if r['model']==name and r['category']==cat]))) for name in MODEL_NAMES[1:])+'|')
    lines += ['', 'ROI表是同类区域的等权均值，T04保留18个ROI的独立结果。小目标、边界错位和遮挡仍需结合F07逐项判断，局部数值不代表独立阀门类别。','', '## 语义分辨率和运行开销']
    higher=read(run/'06_evaluation/IPSGR-Full/semantic_512x256/metrics.json')
    primary=results['IPSGR-Full']['semantic_pooled']['miou'];lines += [f'IPSGR-Full主评价256×128 mIoU={primary:.4f}；独立原生512×256渲染 mIoU={higher["pooled"]["miou"]:.4f}。两种分辨率分别采样GT，后者不替换主消融排名。']
    diagnostic=read(run/'07_industrial_analysis/resolution_audit.json')['rows']
    retention={}
    for k in range(7):
        rows=[r for r in diagnostic if r['width']==256 and r['class_id']==k];retention[classes[k]]=sum(r['GT_class_pixels_after_5x5_erosion'] for r in rows)/max(sum(r['sampled_pixels'] for r in rows),1)
    lines += ['256×128下一个语义像素对应原图8×8像素，5×5腐蚀窗口对应40×40像素。GT类别掩码的腐蚀保留率为：'+', '.join(f'{k}={100*v:.1f}%' for k,v in retention.items())+'。这只诊断细结构尺度，不能替代实际SAM掩码分析。','', '|模型|最终高斯数|最终阶段训练分钟|额外ISLB分钟|RGB渲染毫秒|语义特征渲染毫秒|','|---|---:|---:|---:|---:|---:|']
    for r in cost:lines.append(f'|{r["model"]}|{r["gaussians"]:,}|{r["training_seconds_last_invocation"]/60:.2f}|{r["ISLB_extra_seconds"]/60:.2f}|{r["mean_RGB_render_seconds"]*1000:.2f}|{r["mean_semantic_feature_render_seconds"]*1000:.2f}|')
    lines += ['', '上述耗时为共享GPU的本轮实际运行记录，未进行隔离性能基准。额外ISLB为三个第二阶段模型共享的初始化成本，不能把它隐藏在“相同10k预算”描述中。']
    wan=read(run/'01_geometry/Wan360/parallel_completed.json');generation=read(run/'01_geometry/Wan360/fresh_generation_audit.json')
    lines += [f'80个Wan360视频实际由GPU0/1/2分别生成{generation["GPU_jobs"].get("0",0)}/{generation["GPU_jobs"].get("1",0)}/{generation["GPU_jobs"].get("2",0)}个；从最初单卡启动到全部完成耗时{wan["seconds_since_first_single_gpu_launch"]/60:.2f}分钟，各任务生成耗时合计{wan["summed_video_generation_seconds"]/3600:.2f}小时。GPU3同期运行基线；几何特征匹配在CPU上提前计算。']
    return lines
