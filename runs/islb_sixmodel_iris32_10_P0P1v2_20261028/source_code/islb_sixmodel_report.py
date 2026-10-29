"""Publication figures and a portable Chinese report for the frozen six-model run."""
from pathlib import Path
import argparse, collections, html, json, os, shutil, time
import numpy as np
from PIL import Image
from .common import ROOT, read, write, ref, sha, signature, native
from .islb_sixmodel_data import RUN, MODELS
from .islb_sixmodel_evaluate import colorize, crop_periodic, csvsave

FINAL = ['B-RGB', 'B-SEM', 'ISLB-Only']
LAYOUT = ['L0', 'L1', 'L2']
DISPLAY = {n: ('+ISLB' if n == 'ISLB-Only' else n) for n in MODELS}
CN = ['管道', '设备', '建筑', '支架', '通行结构', '电缆桥架', '通风空调', '阀门']


def qualitative(name, gpu):
    """Render only the three preselected test views, never select new ROIs."""
    assert name != 'B-RGB'
    model = RUN / '05_models' / name
    assert read(model/'checkpoint_validation.json')['status'] == 'passed'
    assert read(model/'final/model.json')['step'] == 10000
    m = read(RUN/'00_protocol/dataset_manifest.json')
    rois = read(RUN/'07_figures/fixed_rois.json')
    names = {r['view'] for r in rois['ROIs']}
    views = [v for v in m['test'] if v['name'] in names]
    folder = RUN/'06_evaluation'/name/'qualitative_1024'
    folder.mkdir(parents=True, exist_ok=True)
    inputs = {'archive': ref(model/'final/model.pt'), 'ROIs': ref(RUN/'07_figures/fixed_rois.json'),
              'native': ref(read(RUN/'00_protocol/common_config.json')['native_binary']), 'resolution': [1024,512]}
    key = signature(inputs)
    if (folder/'complete.json').exists():
        assert read(folder/'complete.json')['signature'] == key
        return
    write(folder/'inputs.json', inputs)
    cfg = read(RUN/'00_protocol/common_config.json')
    req = {**cfg, 'action':'render', 'archive':str(model/'final/model.pt'), 'views':views,
           'gpu':gpu, 'output':str(folder/'native'), 'render_semantics':True,
           'semantic_width':1024, 'semantic_height':512, 'resume':False, 'stage_reset':False}
    files = ['logits.f16','ids.i32','features.f16','rgb.png','rgb.f32']
    if not all((folder/'native'/(v['name']+'_'+s)).exists() for v in views for s in files):
        native(req, folder/f'render_job_{time.time_ns()}')
    palette = [c['color'] for c in m['classes']['classes']]
    for v in views:
        p = folder/'native'/v['name']
        logits = np.fromfile(str(p)+'_logits.f16','f2').reshape(8,512,1024)
        ids = np.fromfile(str(p)+'_ids.i32','i4').reshape(512,1024)
        assert np.isfinite(logits).all()
        labels = logits.argmax(0).astype('u1'); labels[ids<0] = 8
        Image.fromarray(labels).save(folder/(v['name']+'_labels.png'))
        colorize(labels,palette).save(folder/(v['name']+'_semantic.png'))
    write(folder/'complete.json', {'status':'completed','signature':key,'views':list(names),
          'native_semantic_resolution':[1024,512], 'display_ERP_resolution':[2048,1024],
          'resampling':'Nearest-neighbor class labels only; RGB remains native 2K', 'inputs':inputs})
    print('QUALITATIVE_READY',name,flush=True)


def panel_sources():
    m = read(RUN/'00_protocol/dataset_manifest.json')
    rois = read(RUN/'07_figures/fixed_rois.json')['ROIs']
    palette = [c['color'] for c in m['classes']['classes']]
    output = RUN/'07_figures/panels'; output.mkdir(exist_ok=True)
    result = {}
    for r in rois:
        v = next(v for v in m['test'] if v['name'] == r['view'])
        box = r['box_ERP_2k']; prefix = r['roi_id']; result[prefix] = {}
        gt = np.array(Image.open(v['label'])); mask = np.array(Image.open(v['supervision_mask'])) > 0
        gt[~mask] = 8
        full = {'GT_RGB':Image.open(v['image']).convert('RGB'), 'GT_Semantic':colorize(gt,palette)}
        for n in MODELS:
            p = RUN/'06_evaluation'/n
            full[n+'_RGB'] = Image.open(p/'native'/(v['name']+'_rgb.png')).convert('RGB')
            if n != 'B-RGB':
                assert (p/'qualitative_1024/complete.json').exists()
                full[n+'_Semantic'] = Image.open(p/'qualitative_1024'/(v['name']+'_semantic.png')).convert('RGB').resize((2048,1024),Image.Resampling.NEAREST)
        for key, im in full.items():
            dest = output/(prefix+'_'+key+'.png')
            crop_periodic(im,box).save(dest)
            result[prefix][key] = dest
    write(RUN/'07_figures/panel_manifest.json', {'ROI_file':ref(RUN/'07_figures/fixed_rois.json'),
          'panels':{k:{a:str(b.relative_to(RUN/'07_figures')) for a,b in v.items()} for k,v in result.items()},
          'no_result_based_ROI_selection':True, 'original_RGB_pixels_per_panel':[640,360],
          'native_semantic_pixels_per_panel':[320,180]})
    return result


def figure_grid(stem, title, rows, columns, semantic_legend=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none'})
    nrow = len(rows); ncol = len(columns)
    # Fixed pixel aspect and equal axes; text is vector in PDF/SVG.
    width = 7.2; panel_w = (width-.16-(ncol-1)*.05)/ncol
    panel_h = panel_w*360/640; row_h = panel_h+.25
    height = .55+nrow*row_h+(.54 if semantic_legend else .08)
    fig = plt.figure(figsize=(width,height),facecolor='white')
    fig.text(.012,1-.08/height,title,ha='left',va='top',fontsize=10,fontweight='bold')
    for j,c in enumerate(columns):
        fig.text((.08+j*(panel_w+.05)+panel_w/2)/width,1-.35/height,c,ha='center',va='center',fontsize=8.6)
    for i,(caption,paths) in enumerate(rows):
        ytop = height-.55-i*row_h
        fig.text(.012,(ytop+.06)/height,caption,fontsize=7.6,ha='left',va='bottom')
        for j,p in enumerate(paths):
            ax = fig.add_axes([(.08+j*(panel_w+.05))/width,(ytop-panel_h)/height,panel_w/width,panel_h/height])
            ax.imshow(np.asarray(Image.open(p)),interpolation='none'); ax.set_axis_off()
    if semantic_legend:
        m=read(RUN/'00_protocol/dataset_manifest.json')
        names=['Piping','Equipment','Building','Frame','Access','Cable tray','HVAC','Valve','Invalid / uncovered']
        colors=[np.array(c['color'])/255 for c in m['classes']['classes']]+[np.array([25,25,25])/255]
        fig.legend(handles=[Patch(color=c,label=n) for n,c in zip(names,colors)],loc='lower center',ncol=5,
                   fontsize=6.3,frameon=False,bbox_to_anchor=(.5,.01),columnspacing=1,handlelength=1.2)
    dest=RUN/'07_figures'/stem
    fig.savefig(str(dest)+'.png',dpi=320)
    fig.savefig(str(dest)+'.pdf',dpi=320)
    fig.savefig(str(dest)+'.svg',dpi=320)
    plt.close(fig)


def figures():
    panels=panel_sources();rs=read(RUN/'07_figures/fixed_rois.json')['ROIs']
    captions={'ROI1':'Valve and piping','ROI2':'Equipment boundaries and elbows','ROI3':'Crossing pipes, cable trays and frames'}
    a=[];b=[];c=[]
    for r in rs:
        k=r['roi_id'];p=panels[k];label=f"{k} / {r['camera_id']} / {captions[k]}"
        a.append((label,[p['GT_RGB'],p['L0_RGB'],p['L1_RGB']]))
        b.append((label,[p['GT_Semantic'],p['L1_Semantic'],p['L2_Semantic']]))
        c.append((label+' / RGB',[p['GT_RGB'],p['B-SEM_RGB'],p['ISLB-Only_RGB']]))
        c.append((label+' / Semantic',[p['GT_Semantic'],p['B-SEM_Semantic'],p['ISLB-Only_Semantic']]))
    figure_grid('F05a_layout_RGB','(a) Industrial layout reconstruction',a,['Ground truth','L0: P0 layout','L1: P1 layout'])
    figure_grid('F05b_sparse_anchors','(b) Geometry-guided sparse semantic anchoring',b,['Ground truth semantic','L1: without anchors','L2: with anchors'],True)
    figure_grid('F05c_final_ISLB','(c) ISLB contribution to the final model',c,['Ground truth','B-SEM','+ISLB'],True)
    print('F05_COMPLETE',flush=True)


def final_integrity():
    references=read(RUN/'00_protocol/readonly_inputs.json')
    mismatches=[r['path'] for r in references if sha(r['path'])!=r['sha256']]
    assert not mismatches,mismatches
    seq={};checkpoints={};audits={};config={}
    for n in MODELS:
        folder=RUN/'05_models'/n
        assert read(folder/'outputs.json')['status']=='completed'
        assert read(folder/'checkpoint_validation.json')['status']=='passed'
        assert read(RUN/'06_evaluation'/n/'metrics.json')['status']=='completed'
        rows=[json.loads(line) for line in (folder/'losses.jsonl').read_text().splitlines()]
        assert [r['step'] for r in rows]==list(range(1,10001))
        seq[n]=[r['view'] for r in rows];checkpoints[n]=read(folder/'checkpoint_validation.json')
        audits[n]=read(folder/'training_audit.json');config[n]=ref(folder/'fixed_request.json')
    paired={pair:seq[a]==seq[b] for pair,a,b in [('L0_L1','L0','L1'),('L1_L2','L1','L2'),('BSEM_L0','B-SEM','L0'),('BRGB_BSEM','B-RGB','B-SEM')]}
    assert all(paired.values()),paired
    assert read(RUN/'05_models/ISLB-Only/initialization_validation.json')['status']=='passed'
    result={'status':'passed','read_only_files_verified':len(references),'changed_historical_inputs':mismatches,
      'same_real_view_sampling_sequence':paired,'all_checkpoints_fresh_this_run':True,
      'checkpoints':checkpoints,'training_audits':audits,'configurations':config,
      'L2_stage_switch':read(RUN/'05_models/ISLB-Only/initialization_validation.json'),
      'preflight':read(RUN/'04_preflight/results.json'), 'finished':time.time()}
    write(RUN/'09_report/final_integrity.json',result);return result


def tables_and_curves():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results={n:read(RUN/'06_evaluation'/n/'metrics.json') for n in MODELS}
    rows=[];perclass=[];perview=[];states=[];secondary=[]
    for n,r in results.items():
        sem=r['semantic_pooled'];out=read(RUN/'05_models'/n/'outputs.json')
        rows.append({'model':DISPLAY[n],'level':r['level'],**r['RGB_mean'],
             'mIoU_percent':100*sem['mIoU'] if sem else None,'mAcc_percent':100*sem['mAcc'] if sem else None,
             'Gaussian_count':r['gaussians'],'iterations_this_stage':10000,'total_iterations_including_layout':20000 if n=='ISLB-Only' else 10000})
        pc={'model':DISPLAY[n],'level':r['level']};pc.update({c:100*sem['IoU'][i] if sem and sem['IoU'][i] is not None else None for i,c in enumerate(r['classes'])});perclass.append(pc)
        for vr in r['rows']:
            v={'model':DISPLAY[n],'level':r['level'],'camera_id':vr['camera_id'],'view':vr['view'],**{k:vr[k] for k in ['PSNR','SSIM','LPIPS']}}
            v.update({k+'_percent':100*vr['semantic'][k] if sem else None for k in ['mIoU','mAcc']});v.update({c+'_IoU_percent':100*vr['semantic']['IoU'][i] if sem and vr['semantic']['IoU'][i] is not None else None for i,c in enumerate(r['classes'])});perview.append(v)
        state={'model':DISPLAY[n],'status':out['status'],'GPU':out['GPU'],'train_minutes':out['seconds_this_invocation']/60,
               'resumed':out['resumed'],'checkpoint':out['model']['path'],'checkpoint_sha256':out['model']['sha256'],
               'config':str(RUN/'05_models'/n/'fixed_request.json'),'loss_log':str(RUN/'05_models'/n/'losses.jsonl'),'evaluation':str(RUN/'06_evaluation'/n/'metrics.json')}
        states.append(state)
        for scope,key in [('primary_valid_real_labels','semantic_pooled'),('secondary_full_consolidated','secondary_full_consolidated_labels'),('secondary_frozen_text_query','frozen_text_query_pooled')]:
            s=r[key];secondary.append({'model':DISPLAY[n],'scope':scope,'mIoU_percent':100*s['mIoU'] if s else None,'mAcc_percent':100*s['mAcc'] if s else None})
    table=RUN/'08_tables'
    for level in ['final','layout']:
        csvsave(table/(level+'_ablation.csv'),[r for r in rows if r['level']==level])
        csvsave(table/(level+'_class_IoU_percent.csv'),[r for r in perclass if r['level']==level])
    csvsave(table/'all_test_view_metrics.csv',perview);csvsave(table/'model_status_paths.csv',states);csvsave(table/'secondary_semantic_metrics.csv',secondary)
    deltas=[];view_deltas=[]
    for a,b,label in [('L0','L1','Geometry: L0 to L1'),('L1','L2','Anchors: L1 to L2'),('B-SEM','ISLB-Only','Final: B-SEM to +ISLB')]:
        ra=next(r for r in rows if r['model']==DISPLAY[a]);rb=next(r for r in rows if r['model']==DISPLAY[b])
        va={v['camera_id']:v for v in results[a]['rows']};vb={v['camera_id']:v for v in results[b]['rows']}
        for camera in va:
            pa,pb=va[camera],vb[camera]
            row={'comparison':label,'camera_id':camera,**{k+'_delta':pb[k]-pa[k] for k in ['PSNR','SSIM','LPIPS']},
                 **{k+'_delta_pp':100*(pb['semantic'][k]-pa['semantic'][k]) for k in ['mIoU','mAcc']}}
            for i,cls in enumerate(results[a]['classes']):
                ia,ib=pa['semantic']['IoU'][i],pb['semantic']['IoU'][i]
                row[cls+'_IoU_delta_pp']=100*(ib-ia) if ia is not None and ib is not None else None
            view_deltas.append(row)
        deltas.append({'comparison':label,**{k:rb[k]-ra[k] for k in ['PSNR','SSIM','LPIPS','mIoU_percent','mAcc_percent','Gaussian_count']},
             'PSNR_improved_views':sum(vb[c]['PSNR']>va[c]['PSNR'] for c in va),
             'mIoU_improved_views':sum(vb[c]['semantic']['mIoU']>va[c]['semantic']['mIoU'] for c in va)})
    csvsave(table/'paired_deltas.csv',deltas)
    csvsave(table/'paired_test_view_deltas.csv',view_deltas)
    summary={'models':rows,'class_IoU_percent':perclass,'states':states,'deltas':deltas,'secondary':secondary,'results':results}
    write(table/'complete_results.json',summary)
    fig,axes=plt.subplots(2,3,figsize=(12,6.5),constrained_layout=True)
    colors=dict(zip(MODELS,['#666666','#1467a6','#a79d25','#478754','#df751d','#ad2c72']))
    for n in MODELS:
        ls=[json.loads(x) for x in (RUN/'05_models'/n/'losses.jsonl').read_text().splitlines()]
        steps=np.array([r['step'] for r in ls]);stride=100
        for ax,key,title in zip(axes.flat,['rgb','semantic_loss','local3d','anchor_weighted_loss','gaussians','semantic_grad_norm'],
                      ['RGB objective','CLIP semantic objective','Neighbor semantic regularizer','Sparse anchor objective (weighted)','Gaussian count','Semantic gradient norm']):
            values=np.array([r[key] for r in ls]);smoothed=np.array([values[i:i+stride].mean() for i in range(0,len(values),stride)])
            ax.plot(steps[stride-1::stride],smoothed,label=DISPLAY[n],color=colors[n],lw=1.3);ax.set_title(title,fontsize=10);ax.set_xlabel('Iteration');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.savefig(RUN/'07_figures/training_curves.png',dpi=200);fig.savefig(RUN/'07_figures/training_curves.pdf');plt.close(fig)
    anchor=read(RUN/'03_anchors/manifest.json')['summary'];fig,ax=plt.subplots(figsize=(9,3.4),constrained_layout=True)
    vals=list(anchor['class_counts'].values());names=list(anchor['class_counts']);bars=ax.bar(names,vals,color=[np.array(c['color'])/255 for c in read(RUN/'00_protocol/dataset_manifest.json')['classes']['classes']])
    ax.bar_label(bars,padding=3,fontsize=8);ax.set_ylim(0,max(vals)*1.15);ax.set_ylabel('Sparse generated-view observations');ax.tick_params(axis='x',rotation=25,labelsize=8)
    fig.savefig(RUN/'07_figures/anchor_distribution.png',dpi=200);plt.close(fig)
    return summary


def htable(rows, fields, formats=None):
    formats=formats or {};e=html.escape
    def cell(row,k):
        value=row.get(k)
        if value is None:return 'N/A'
        return e(formats[k](value) if k in formats else str(value))
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+e(label)+'</th>' for k,label in fields)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+cell(row,k)+'</td>' for k,label in fields)+'</tr>' for row in rows)+'</tbody></table></div>'


def report(summary):
    folder=RUN/'09_report';bundle=folder/'portable';bundle.mkdir(exist_ok=True)
    for name in ['figures','tables','data','gallery','audit','source_code']: (bundle/name).mkdir(exist_ok=True)
    for src in (RUN/'07_figures').iterdir():
        if src.is_file():shutil.copy2(src,bundle/'figures'/src.name)
    shutil.copytree(RUN/'07_figures/panels',bundle/'figures/panels',dirs_exist_ok=True)
    for src in (RUN/'08_tables').glob('*'):shutil.copy2(src,bundle/'tables'/src.name)
    for src in (RUN/'00_protocol').glob('*.json'):shutil.copy2(src,bundle/'data'/src.name)
    for name in ['final_integrity.json','implementation_notes.json','qualitative_review.json','environment.json']:
        if (folder/name).exists():shutil.copy2(folder/name,bundle/'audit'/name)
    shutil.copy2(RUN/'03_anchors/manifest.json',bundle/'audit/anchor_manifest.json')
    for name in ['visual_correspondence_check.png','visual_correspondence_check.json']:
        if (RUN/'03_anchors'/name).exists():shutil.copy2(RUN/'03_anchors'/name,bundle/'audit'/name)
    shutil.copy2(RUN/'04_preflight/results.json',bundle/'audit/preflight.json')
    for n in MODELS:
        src=RUN/'05_models'/n;dest=bundle/'audit'/n;dest.mkdir(exist_ok=True)
        for f in ['fixed_request.json','inputs.json','outputs.json','checkpoint_validation.json','training_audit.json','losses.jsonl','initialization_validation.json','optimizer_parameters.json']:
            if (src/f).exists():shutil.copy2(src/f,dest/f)
        for f in src.glob('failure_*.log'):shutil.copy2(f,dest/f.name)
        shutil.copy2(RUN/'06_evaluation'/n/'metrics.json',dest/'metrics.json')
    shutil.copytree(RUN/'scheduler',bundle/'audit/scheduler',dirs_exist_ok=True,ignore=shutil.ignore_patterns('*.lock'))
    for p in [*ROOT.glob('spcgs/islb_sixmodel_*.py'),ROOT/'tools/build_islb_sixmodel_v2.py',ROOT/'tools/finish_islb_sixmodel.py']:
        shutil.copy2(p,bundle/'source_code'/p.name);shutil.copy2(p,RUN/'source_code'/p.name)
    shutil.copytree(RUN/'source_code',bundle/'source_code',dirs_exist_ok=True)
    shutil.copy2(RUN/'native/build_audit.json',bundle/'audit/native_build.json')
    shutil.copy2(RUN/'native/build_commands.json',bundle/'audit/native_build_commands.json')
    m=read(RUN/'00_protocol/dataset_manifest.json');gallery=[]
    for v in m['test']:
        dest=bundle/'gallery'/v['camera_id'];dest.mkdir(exist_ok=True)
        Image.open(v['image']).save(dest/'GT_RGB.png')
        gt=np.array(Image.open(v['label']));gt[np.array(Image.open(v['supervision_mask']))==0]=8
        colorize(gt,[c['color'] for c in m['classes']['classes']]).save(dest/'GT_Semantic.png')
        for n in MODELS:
            src=RUN/'06_evaluation'/n
            shutil.copy2(src/'native'/(v['name']+'_rgb.png'),dest/(n+'_RGB.png'))
            if n!='B-RGB':shutil.copy2(src/(v['name']+'_semantic.png'),dest/(n+'_Semantic.png'))
        for kind in ['RGB','Semantic']:
            items=[('Ground truth','GT')]+[(DISPLAY[n],n) for n in MODELS if kind=='RGB' or n!='B-RGB']
            gallery.append('<h3>'+v['camera_id']+' · '+kind+'</h3><div class="gallery">'+''.join('<a href="gallery/'+v['camera_id']+'/'+n+'_'+kind+'.png"><span>'+label+'</span><img loading="lazy" src="gallery/'+v['camera_id']+'/'+n+'_'+kind+'.png"></a>' for label,n in items)+'</div>')
    metrics=[('model','模型'),('PSNR','PSNR ↑'),('SSIM','SSIM ↑'),('LPIPS','LPIPS ↓'),('mIoU_percent','mIoU (%) ↑'),('mAcc_percent','mAcc (%) ↑'),('Gaussian_count','高斯数量'),('total_iterations_including_layout','累计迭代')]
    formats={k:(lambda x:f'{x:.4f}') for k in ['SSIM','LPIPS']};formats.update({k:(lambda x:f'{x:.2f}') for k in ['PSNR','mIoU_percent','mAcc_percent']});formats['Gaussian_count']=lambda x:f'{x:,}'
    anchors=read(RUN/'03_anchors/manifest.json')['summary'];pre=read(RUN/'04_preflight/results.json')['anchor_actual_optimizer_update']
    modeltext=[
       {'model':'B-RGB','init':'P0 v2','loss':'真实 RGB','layout':'否','anchors':'否','stage':'10k 正式 RGB'},
       {'model':'B-SEM','init':'P0 v2','loss':'真实 RGB + CLIP cosine / CE','layout':'否','anchors':'否','stage':'10k 正式语义'},
       {'model':'L0','init':'P0 v2','loss':'真实 RGB + CLIP + 邻近语义正则','layout':'增密 + OGR','anchors':'否','stage':'10k 布局'},
       {'model':'L1','init':'P1 v2','loss':'真实 RGB + CLIP + 邻近语义正则','layout':'增密 + OGR','anchors':'否','stage':'10k 布局'},
       {'model':'L2','init':'P1 v2','loss':'真实 RGB + CLIP + 邻近语义正则','layout':'增密 + OGR','anchors':'0.05 × 稀疏 CE，仅更新语义','stage':'10k 布局'},
       {'model':'+ISLB','init':'本轮 L2 完成检查点','loss':'真实 RGB + CLIP cosine / CE','layout':'正式阶段关闭 OGR / 邻近正则','anchors':'正式阶段关闭','stage':'再训练 10k，累计 20k'}]
    findings=[]
    for d in summary['deltas']:
        title={'Geometry: L0 to L1':'几何增强 L0→L1','Anchors: L1 to L2':'稀疏锚点 L1→L2','Final: B-SEM to +ISLB':'最终模型 B-SEM→+ISLB'}[d['comparison']]
        rgb='变化很小' if abs(d['PSNR'])<.1 else '提高' if d['PSNR']>0 else '下降'
        sem='变化很小' if abs(d['mIoU_percent'])<1. else '提高' if d['mIoU_percent']>0 else '下降'
        findings.append(f'<p><b>{title}：</b>PSNR {d["PSNR"]:+.3f} dB，SSIM {d["SSIM"]:+.4f}，LPIPS {d["LPIPS"]:+.4f}；mIoU {d["mIoU_percent"]:+.3f} 个百分点，mAcc {d["mAcc_percent"]:+.3f} 个百分点。PSNR 表现{rgb}，语义 mIoU 表现{sem}。逐图统计中，PSNR 改善 {d["PSNR_improved_views"]}/10 张，mIoU 改善 {d["mIoU_improved_views"]}/10 张。高斯数量变化 {d["Gaussian_count"]:+,}。这些是本次固定测试集上的实际差值。</p>')
    classes=[c['name'] for c in m['classes']['classes']]
    rois=read(RUN/'07_figures/fixed_rois.json')['ROIs']
    failures=list((RUN/'05_models').glob('*/failure_*.log'))+list((RUN/'scheduler').glob('*_exit.json'))
    figblocks=[]
    for stem,title,caption in [
        ('F05a_layout_RGB','F05(a) 工业结构布局重建','相同测试视角、相同 ROI 下比较 GT、P0 布局 L0 与 P1 布局 L1。只改变初始点云，其他布局机制相同。'),
        ('F05b_sparse_anchors','F05(b) 稀疏语义锚点','比较 GT Semantic、L1 与 L2。颜色一致，管道和阀门分别显示。暗色表示无效真值或渲染未覆盖区域。'),
        ('F05c_final_ISLB','F05(c) ISLB 对最终模型的贡献','每个 ROI 依次展示 RGB 行和语义行，对比 GT、B-SEM 与 +ISLB。+ISLB 累计 20k 迭代，B-SEM 为 10k，不能解释为等计算预算比较。')]:
        figblocks.append('<h3>'+title+'</h3><p>'+caption+'</p><p>'+''.join('<a href="figures/'+stem+'.'+ext+'">'+ext.upper()+'</a>　' for ext in ['png','pdf','svg'])+'</p><a href="figures/'+stem+'.png"><img class="figure" src="figures/'+stem+'.png"></a>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>IRIS · ISLB 六模型消融实验</title><style>
    :root{color-scheme:light}body{margin:0;background:#f3f5f8;color:#152033;font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1380px;margin:auto;padding:32px}h1{font-size:34px;line-height:1.25}h2{margin-top:50px;border-bottom:2px solid #dbe3ed;padding-bottom:10px}h3{margin-top:28px}a{color:#1767aa}nav{position:sticky;top:0;background:#fff;padding:12px;z-index:5;border-bottom:1px solid #ddd;display:flex;flex-wrap:wrap;gap:22px}.lead,.box{background:white;border:1px solid #dce3ed;border-radius:10px;padding:20px;margin:18px 0}.muted{color:#526178}.scroll{overflow:auto}table{border-collapse:collapse;background:#fff;min-width:900px;width:100%;font-size:14px}th,td{padding:11px 12px;border-bottom:1px solid #e0e5ec;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:#eaf0f7}code{font:13px/1.5 ui-monospace,monospace;overflow-wrap:anywhere}img.figure{width:100%;max-width:1150px;background:white;border:1px solid #ddd}.gallery{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.gallery a{display:block;background:white;text-decoration:none;padding:5px}.gallery img{width:100%;display:block}.gallery span{display:block;font-size:14px;padding:5px}.small{font-size:14px}@media(max-width:750px){main{padding:16px}.gallery{grid-template-columns:1fr}h1{font-size:27px}}@media print{nav{display:none}body{background:white}main{padding:0}h2{break-before:page}.figure{break-inside:avoid}}
    </style><nav><a href="#results">定量结果</a><a href="#methods">实验协议</a><a href="#figures">F05</a><a href="#analysis">结果分析</a><a href="#gallery">10 张测试图</a><a href="#audit">配置与审计</a></nav><main>'''
    page+=f'<h1>IRIS · ISLB 六模型消融实验</h1><p class="muted">v2 P0 / P1 · 正式 32 张训练 + 10 张测试 · 无验证集 · 六个模型均在本轮重新训练</p><div class="lead"><b>执行完成：</b>全部六个模型已完成各自规定阶段的 10,000 次迭代和固定 10 张测试全景评价；+ISLB 继承本轮 L2，再训练 10,000 次。所有数值和 F05 图像来自实际保存的检查点。未执行新的 Wan360 生成或 SfM 重建。<br>服务器独立目录：<code>{RUN}</code></div>'
    page+='<h2 id="results">两个层级的定量消融</h2><p>RGB 为 10 张完整 2048×1024 ERP 全景的平均值。主要语义评价为 512×256，汇总 10 张图的混淆矩阵后计算类别宏平均；mIoU / mAcc 均以百分比表示。预测未覆盖像素计为错误。</p>'
    for level,title in [('final','大层级：最终高斯模型'),('layout','小层级：布局预优化结束')]:
        page+='<h3>'+title+'</h3>'+htable([r for r in summary['models'] if r['level']==level],metrics,formats)
        page+='<p><a href="tables/'+level+'_ablation.csv">下载定量表 CSV</a> · <a href="tables/'+level+'_class_IoU_percent.csv">逐类别 IoU CSV</a></p>'
        page+=htable([r for r in summary['class_IoU_percent'] if r['level']==level],[('model','模型')]+list(zip(classes,CN)),{k:lambda x:f'{x:.2f}' for k in classes})
    page+='<p><a href="tables/all_test_view_metrics.csv">全部 60 个模型–视角的原始指标 CSV</a> · <a href="tables/complete_results.json">完整指标 JSON（含混淆矩阵）</a> · <a href="tables/secondary_semantic_metrics.csv">辅助语义口径</a></p>'
    page+='<h2 id="methods">实验数据与统一实现</h2><p>使用已冻结的 P0 v2（2,781 点）和 P1 v2（7,265 点，新增 4,484 点）。保持原生 OmniGS 的 ERP 渲染、RGB 损失、优化器和高斯增密。布局阶段参考 <a href="https://arxiv.org/html/2503.12535">SPC-GS 原论文</a>的 SGI 思路，结合当前工程的两阶段初始化和工业稀疏语义锚点；这是面向 ERP 工业全景的工程实现，并非原论文所有设定的逐项复现。</p>'
    page+=htable(modeltext,[('model','模型'),('init','初始化'),('loss','监督'),('layout','布局机制'),('anchors','稀疏锚点'),('stage','训练阶段')])
    page+='<div class="box"><b>CLIP 监督口径：</b>统一使用本地已有 CLIP ViT-B/16 预训练文本编码器生成八类文本原型，沿用工程已验证的官方标签→CLIP 原型监督路径。训练真实图像的每个有效标签对应类别文本特征，并通过低维语义高斯、32→512 映射和类别相关性分类头计算 cosine / CE 损失；五个语义模型共享相同教师、文本提示、类别映射及真实标签。这里没有将生成图像送入 CLIP 提取稠密特征，也没有声称使用逐图像的 CLIP 视觉特征蒸馏。</div>'
    page+='<p><b>标签来源：</b>主监督和主要语义评价使用当前已整理标签中 provenance 1（官方映射）和 2（已记录的视觉修订），且要求 RGB 真实观测掩码有效；排除合成底部地面和残留邻类补全。SAM 候选不作为官方标注。另存全整理标签口径，避免将两种评价混用。八类中管道与阀门独立。</p>'
    page+='<p><b>固定设置：</b>seed=42；RGB 训练/评价 2048×1024；CLIP 监督 256×128；主要语义指标 512×256；F05 语义独立渲染 1024×512，然后仅对类别图做最近邻显示到 2K。所有模型增密区间 500–9000、间隔 100、梯度阈值 0.0002、上限 100 万；opacity 每 3000 重置；布局 OGR 每 3000 执行，radius=1，min_neighbors=5。正式阶段只保留真实 RGB 和普通 CLIP 语义监督。ICSC、生成 RGB 监督和生成稠密语义监督均关闭。</p>'
    page+='<p><b>优化器：</b>沿用原生 Adam；位置基础学习率由 1.6e-4 衰减至 1.6e-6，并乘相同场景空间尺度。DC / 高阶 SH 学习率为 0.0025 / 0.000125，opacity=0.05，scale=0.005，rotation=0.001；语义高斯学习率 0.0025，语义映射/分类器 Adam 学习率 0.0005。每个模型保存实际构造参数，阶段切换清空优化器状态。</p>'
    page+='<p><b>评价设置：</b>PSNR 使用完整浮点 RGB；SSIM 使用 Gaussian window（sigma=1.5）；LPIPS 为 AlexNet v0.1，完整 2K 图像，不裁剪或降采样。语义类别不存在时不纳入该宏平均；所有测试图固定，无基于测试指标的检查点选择。ERP 像素均匀加权。没有额外随机种子或超参数消融。</p>'
    page+='<p><b>当前正式 RGB 数据版本：</b>32 张训练图中 18 张沿用此前底部清理版本、14 张为原始 RGB；10 张测试图中 8 张沿用此前底部清理版本、2 张为原始 RGB。本轮没有再次编辑 RGB，所有模型使用同一份已冻结输入。RGB 指标包含完整全景；语义主要指标额外使用上述有效观测掩码。真实监督和锚点 CE 均沿用 0.05 标签平滑，锚点附加权重也是 0.05，这两个参数含义不同。</p>'
    page+='<details><summary>32 张训练与 10 张测试名单</summary><p>训练：'+', '.join(v['camera_id'] for v in m['train'])+'</p><p>测试：'+', '.join(v['camera_id'] for v in m['test'])+'</p><a href="data/dataset_manifest.json">完整坐标、图像路径和哈希</a></details>'
    page+=f'<h3>Geometry-Guided Sparse Semantic Anchoring</h3><p>保留 {anchors["observations"]:,} 个生成视角稀疏观测，来自 {anchors["unique_tracks"]:,} 条轨迹，覆盖 {anchors["generated_views"]} 个已注册生成视角。每个锚点要求至少两个真实训练视角的有效、几何可靠标签一致，再传播至生成视角对应特征点。使用 2K 角度重投影误差 ≤2 px、现有稀疏深度一致性及边界检查。稀疏深度筛选不能替代完整可见性真值。</p><img class="figure" src="figures/anchor_distribution.png">'
    page+=f'<p>锚点仅用相同 CLIP 兼容分类空间的 CE，权重 0.05。独立一次实际 Adam 更新验证：语义参数最大变化 {pre["semantic_max_change"]:.8f}，映射参数最大变化 {pre["decoder_max_change"]:.8f}，几何与 RGB 参数最大变化为 0。正式 L2 每 100 步记录锚点独立梯度，训练审计验证其参与更新。</p>'
    if (bundle/'audit/visual_correspondence_check.png').exists():
        page+='<p>稀疏对应关系目视抽查：同一阀门轨迹获得 3 个有效真实标签，图中展示其中两个真实训练观测和一个已注册 Wan360 观测。红圈标记同一稀疏特征点，生成观测的 2K 等效重投影误差为 1.320 px。此图只用于点级传播检查，生成 RGB 不进入训练监督。</p><img class="figure" src="audit/visual_correspondence_check.png">'
    page+='<p><b>L2→+ISLB：</b>保留本轮 L2 的有效 XYZ、DC 颜色、语义向量及已学习的语义映射/分类器和固定文本原型；高阶 SH 归零，opacity 重置至 0.1，rotation 归单位，scale 按邻近点距离重建；优化器和增密状态清空。实际阶段初始检查点与 L2 保留参数逐项一致，检查结果保存在审计文件中。</p>'
    page+='<h2 id="figures">F05 论文定性对比图</h2><p>三组图使用相同三个测试视角和固定 640×360 ERP ROI；ROI 在首次评价前仅根据真实图的工业内容和照明情况选定，未按模型胜负更换。下方提供 PNG、带矢量文字的 PDF / SVG 和全部独立子图。保留完整 10 张测试图，便于检查局部图是否有代表性。</p>'+''.join(figblocks)
    page+=htable(rois,[('roi_id','ROI'),('camera_id','相机'),('box_ERP_2k','2K ERP [x,y,w,h]'),('GT_valid_fraction','有效 GT 比例')],{'GT_valid_fraction':lambda x:f'{100*x:.3f}%'})
    page+='<p><a href="figures/fixed_rois.json">固定 ROI 与选择记录 JSON</a> · <a href="figures/panel_manifest.json">独立子图索引</a></p>'
    page+='<h2 id="analysis">结果解释与实际限制</h2>'+''.join(findings)+'<p><a href="tables/paired_test_view_deltas.csv">三组比较的逐图、逐类别差值 CSV</a></p>'
    if (folder/'qualitative_review.json').exists():
        review=read(folder/'qualitative_review.json')
        page+='<h3>固定 ROI 的目视核查</h3>'+''.join('<p><b>'+html.escape(k)+'：</b>'+html.escape(v)+'</p>' for k,v in review['observations'].items())
    page+='<p>六个模型均按事先固定的设置执行，没有为了使结论更显著而删除测试图、调整类别权重、重选 ROI 或替换检查点。L0→L1 用于比较初始增强几何；L1→L2 用于比较新增锚点项。几何增强是否改善 RGB 或语义，应分别按上述真实指标判断，不以点数更多直接推出质量更好。</p><p>本实验为单场景、单随机种子结果；+ISLB 含 10k 布局 + 10k 正式训练，具有更高累计训练预算。阀门只有 24 个有效稀疏锚点，且在语义监督像素中占比较小；其局部边界和逐类 IoU 需与整体 mIoU 分开看。现有标签中的视觉修订和 ERP 语义分辨率也限制了边界精度解释。以上结果不构成跨场景或等预算优势的证明。</p><img class="figure" src="figures/training_curves.png">'
    page+='<h2 id="gallery">全部 10 张测试全景</h2><p>点击查看原图。RGB 保留原始 2K PNG；此处完整语义图显示主要评价使用的 512×256 预测，F05 使用另外保存的 1024×512 语义渲染。所有浮点 RGB、语义 logits 和特征原始文件保留在服务器 06_evaluation。</p>'+''.join(gallery)
    page+='<h2 id="audit">训练状态、路径与可复现记录</h2>'+htable(summary['states'],[('model','模型'),('status','状态'),('GPU','GPU'),('train_minutes','本次训练分钟'),('resumed','断点恢复')],{'train_minutes':lambda x:f'{x:.1f}'})
    page+='<p>服务器为 4 张 NVIDIA RTX A6000（每张 48 GB），采用独立任务并行、未使用 DDP。原生训练使用 LibTorch 2.0.1 / CUDA 11.8 / OpenCV 4.7.0；Python 预处理与评价环境为 PyTorch 2.6.0+cu118。<a href="audit/environment.json">完整环境记录</a> · <a href="audit/native_build_commands.json">原生编译与链接命令</a></p>'
    page+='<p><a href="tables/model_status_paths.csv">检查点 / 配置 / 日志路径及 SHA-256</a> · <a href="audit/final_integrity.json">最终完整性审计</a> · <a href="audit/preflight.json">训练前验证</a> · <a href="audit/anchor_manifest.json">锚点统计</a> · <a href="audit/native_build.json">原生程序构建记录</a></p>'
    page+=f'<p>训练失败 / 非零退出记录数量：{len(failures)}。所有存在的失败记录保留在新实验目录。每 1000 步保存模型、优化器与 RNG 状态，同配置签名允许恢复；完整模型归档保留服务器，不将数十 GB 的检查点重复放入本地报告包。报告包包含原始指标、PNG、独立子图、配置、损失日志和本轮适配源代码。训练用时受 GPU 分配及服务器共享负载影响，不作为等硬件速度比较。</p>'
    for s in summary['states']:page+='<details><summary>'+s['model']+' 检查点</summary><p><code>'+html.escape(s['checkpoint'])+'</code></p><p class="small">SHA-256: '+s['checkpoint_sha256']+'</p><a href="audit/'+('ISLB-Only' if s['model']=='+ISLB' else s['model'])+'/fixed_request.json">训练配置</a></details>'
    page+='</main></html>'
    (bundle/'report.html').write_text(page,encoding='utf-8')
    files=[ref(p) for p in bundle.rglob('*') if p.is_file() and p.name!='bundle_manifest.json']
    write(bundle/'bundle_manifest.json',{'status':'completed','run':str(RUN),'files':[{**r,'path':str(Path(r['path']).relative_to(bundle))} for r in files], 'created':time.time()})
    write(folder/'outputs.json',{'status':'completed','portable_report':str(bundle/'report.html'),'bundle_files':len(files),'figures':[str(RUN/'07_figures'/s) for s in ['F05a_layout_RGB.png','F05b_sparse_anchors.png','F05c_final_ISLB.png']]})
    print('REPORT_COMPLETE',bundle,flush=True)


def build():
    final_integrity();summary=tables_and_curves();figures();report(summary)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['qualitative','build']);p.add_argument('--model',choices=MODELS);p.add_argument('--gpu',type=int,default=2);args=p.parse_args()
    if args.action=='qualitative':qualitative(args.model,args.gpu)
    else:build()
