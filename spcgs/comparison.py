from pathlib import Path
import csv,json
import numpy as np
from PIL import Image,ImageDraw
from .common import *

def compare_pair(left,right,out,views=None):
    left,right,out=Path(left),Path(right),Path(out);out.mkdir(parents=True,exist_ok=True);rows=[]
    names=sorted({p.name.removesuffix('_rgb.png') for p in left.glob('*_rgb.png')}&{p.name.removesuffix('_rgb.png') for p in right.glob('*_rgb.png')})
    if views:names=[n for n in names if n in views]
    for name in names:
        a=Image.open(left/(name+'_rgb.png')).convert('RGB');b=Image.open(right/(name+'_rgb.png')).convert('RGB');assert a.size==b.size
        aa=np.array(a).astype(float);bb=np.array(b).astype(float);diff=Image.fromarray(np.clip(np.abs(aa-bb)*4,0,255).astype('u1'))
        canvas=Image.new('RGB',(a.width*3,a.height+32),'white');canvas.paste(a,(0,32));canvas.paste(b,(a.width,32));canvas.paste(diff,(a.width*2,32));draw=ImageDraw.Draw(canvas)
        for i,label in enumerate([left.parent.name,right.parent.name,'absolute difference x4']):draw.text((i*a.width+8,8),label,fill='black')
        canvas.save(out/(name+'_rgb_comparison.png'))
        rows.append({'view':name,'mean_abs_rgb_change_8bit':float(np.abs(aa-bb).mean()),'max_abs_rgb_change_8bit':float(np.abs(aa-bb).max())})
        sa,sb=left/(name+'_semantic.png'),right/(name+'_semantic.png')
        if sa.exists() and sb.exists():
            a,b=Image.open(sa),Image.open(sb);canvas=Image.new('RGB',(a.width+b.width,max(a.height,b.height)));canvas.paste(a,(0,0));canvas.paste(b,(a.width,0));canvas.save(out/(name+'_semantic_comparison.png'))
    write(out/'metrics.json',rows);return rows

def make_report(run,cfg):
    run=Path(run);out=run/STAGES['compare'];out.mkdir(parents=True,exist_ok=True);report=run/STAGES['report'];report.mkdir(exist_ok=True)
    paths={'baseline':run/STAGES['baseline'],'M1':run/STAGES['m1'],'CLIP':run/STAGES['semantic']/'cache','S0':run/STAGES['semantic']/'s0','SGI':run/STAGES['sgi'],
           'C0':run/STAGES['spc']/'c0','C1':run/STAGES['spc']/'c1','C2':run/STAGES['spc']/'c2','SAM2':run/STAGES['isp']}
    rows=[]
    for name,p in paths.items():
        status=read(p/'outputs.json') if (p/'outputs.json').exists() else {'status':'not_run'}
        metrics=read(p/'metrics.json') if (p/'metrics.json').exists() else {}
        rows.append({'stage':name,'status':status['status'],'psnr':metrics.get('test_psnr',metrics.get('psnr')),'ssim':metrics.get('test_ssim',metrics.get('ssim')),'lpips':metrics.get('lpips'),'gaussians':metrics.get('gaussians',status.get('gaussians')),'reason':status.get('reason'),'path':str(p)})
    for a,b in [('baseline','S0'),('baseline','SGI'),('S0','C0'),('C0','C1'),('C1','C2'),('baseline','C2')]:
        if (paths[a]/'renders').exists() and (paths[b]/'renders').exists():compare_pair(paths[a]/'renders',paths[b]/'renders',out/(a+'_vs_'+b))
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(out/'metrics.json',rows);write(out/'inputs.json',{'stages':{k:str(v) for k,v in paths.items()}});write(out/'outputs.json',{'status':'completed','table':str(out/'summary.csv')});write(out/'config.json',cfg)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name,p in paths.items():
        if (p/'losses.jsonl').exists():
            losses=[json.loads(x) for x in (p/'losses.jsonl').read_text().splitlines()];fig,axes=plt.subplots(2,2,figsize=(12,7))
            for ax,keys in zip(axes.flat,[['rgb','clip','local3d'],['inter','intra','spc3d'],['gaussians'],['semantic_grad_norm','xyz_grad_norm']]):
                for k in keys:ax.plot([r['step'] for r in losses],[r[k] for r in losses],label=k)
                ax.legend();ax.set_xlabel('Iteration');ax.grid(alpha=.25)
            fig.suptitle(name);fig.tight_layout();fig.savefig(report/(name+'_curves.png'),dpi=140);plt.close(fig)
    lines=['# OmniGS → SPC-GS 工业全景研究运行报告','',f'运行目录：`{run}`',f'模式：`{cfg["mode"]}`；训练视角 {cfg["train_views"]}；RGB {cfg["width"]}×{cfg["height"]}；语义 {cfg["semantic_width"]}×{cfg["semantic_height"]}。','',
      '## 实际状态与测试集 RGB 指标','', '| 阶段 | 状态 | PSNR | SSIM | LPIPS | 高斯数 |','|---|---|---:|---:|---:|---:|']
    for r in rows:lines.append('| '+' | '.join(str(r[k]) if r[k] is not None else '未计算' for k in ['stage','status','psnr','ssim','lpips','gaussians'])+' |')
    lines+=['','## 输入输出与失败原因','']
    for r in rows:lines +=[f'- **{r["stage"]}**：`{r["path"]}`。状态：{r["status"]}。'+(f'原因：{r["reason"]}' if r['reason'] else '')]
    lines+=['','## 结果解释','',
      '本报告保留所有结果。快速运行主要验证数据流、梯度和保存重载，不能证明方法提升；不能把短训练的 SGI/C0 与已训练 32,010 次的基线当作公平性能竞赛。',
      'S0 从已有 RGB 模型开始学习语义；C0/C1/C2 从独立 SGI 的位置、DC 颜色和语义属性重新初始化，重置尺度、旋转、透明度、高阶 SH 和优化器。',
      'C0/C1/C2 迭代预算一致，用于检查 2D/3D SPC 的边际变化。M1 跳过时，无增强点云，也不构成完整 SGI 复现。',
      f'CLIP 为真实 ViT-B/16 预训练模型，语义后端为 `{cfg.get("feature_backend","patch_tokens")}`，由六面局部透视特征回投影；这是可运行的密集特征适配。SAM2 使用 Hiera-tiny，单次真实图加两个当前模型伪视角；环形平移接缝后再映回原全景。',
      'SAM2 有效率仅表示掩码面积/非空检查通过率，不代表物体对应准确率。SPC 按配置间隔在线刷新并应用；快速模式不同于论文每步执行。',
      '无独立类别语义真值；mIoU/mAcc 不计算。CLIP 置信度、熵和区域一致性均为代理观察量。LPIPS 是否完成以上表为准。','',
      '## 产物与比较','',f'统一对比：`{out}`。每对模型使用相同相机、分辨率和词表，保存 RGB、语义并排图和 RGB 差异图。',
      '每个原生模型的 `model.pt` 保存完整高斯、32 维语义、映射头、优化器和 RNG；`point_cloud.ply` 可由原 OmniGS 加载，但语义必须同时读取 sidecar 或原生 archive。','',
      '## 参数快照','', '```json',json.dumps(cfg,indent=2,ensure_ascii=False),'```','',
      '## 方法来源','', '[SPC-GS 原论文](https://arxiv.org/html/2503.12535)；[OmniGS 原工程](../../../code/README.md)；[SAM2](https://github.com/facebookresearch/sam2)；[CLIP ViT-B/16](https://huggingface.co/openai/clip-vit-base-patch16)。']
    lines+=['','## 实际模块变化','']
    lookup={r['stage']:r for r in rows}
    for a,b in [('baseline','S0'),('SGI','C0'),('C0','C1'),('C1','C2')]:
        x,y=lookup[a],lookup[b]
        if x['psnr'] is not None and y['psnr'] is not None:
            lines.append(f'- {a} → {b}：测试 PSNR 变化 {y["psnr"]-x["psnr"]:+.4f} dB，SSIM 变化 {y["ssim"]-x["ssim"]:+.5f}。')
    for name in ['SGI','C0','C1','C2']:
        p=paths[name]/'metrics.json'
        if p.exists():
            m=read(p);lines.append(f'- {name}：高斯数 {m.get("gaussians")}；OGR 实际剔除 {m.get("ogr_removed",0)}；有效 ISP 对应 {m.get("valid_isp_pairs",0)}。')
    rp=out/'fixed_region_consistency/metrics.json'
    if rp.exists():
        measurements=read(rp)['measurements'];lines+=['','固定 C2 快照区域和相机上的一致性代理统计（非准确率）：','']
        for name in ['c0','c1','c2']:
            subset=[r for r in measurements if r['model']==name]
            if subset:lines.append(f'- {name}：{len(subset)} 对区域，平均特征余弦 {np.mean([r["feature_cosine"] for r in subset]):.5f}，平均类别分布 KL {np.mean([r["region_class_distribution_KL"] for r in subset]):.5f}。')
    manifest=read(run/'data_manifest.json');name=manifest['test'][0]['name']
    lines+=['','## 关键可视化','']
    examples=[out/'initialization'/(name+'_initialization.png'),out/'initialization/gaussian_distribution.png',out/'baseline_vs_C2'/(name+'_rgb_comparison.png'),out/'C0_vs_C1'/(name+'_semantic_comparison.png'),out/'C1_vs_C2'/(name+'_semantic_comparison.png'),paths['C2']/'renders'/(name+'_semantic.png')]
    import os
    for p in examples:
        if p.exists():lines.append(f'- [{p.name}]({os.path.relpath(p,report)})')
    lines+=['','## 数值检查与未完成事项','',
      'M1 的 MotionCtrl 生成未执行；导入/在线模型的注册全景接入代码已提供，但本次无生成观测，不能判断视角增强效果。',
      '正式 10k+10k 预算未执行；本次保存的是配置所示迭代预算下的阶段模型。语义边界仍较粗，存在设备类别混淆，不能报告正式语义识别精度。',
      f'数值检查：`{ROOT}/tests/results/native_gradient_checks.json`；原版等价/恢复：`{ROOT}/tests/regression/results.json`；独立真实 SPC：`{ROOT}/tests/real_spc/results.json`。']
    (report/'report.md').write_text('\n'.join(lines),encoding='utf-8');write(report/'outputs.json',{'status':'completed','report':str(report/'report.md')});write(report/'config.json',cfg)
    return report/'report.md'

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('left');p.add_argument('right');p.add_argument('output');a=p.parse_args();compare_pair(a.left,a.right,a.output)
