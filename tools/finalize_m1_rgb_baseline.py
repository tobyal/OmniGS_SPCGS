"""Collect completed baseline-controlled RGB training and per-view comparisons."""
import argparse
import csv
from pathlib import Path
import re
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spcgs.common import ROOT, read, write, ref


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',default='m1_rgb_baseline_11901_20261024')
    args=parser.parse_args()
    run=ROOT/'runs/industrial_spcgs'/args.run
    state=read(run/'status.json');assert state['status']=='completed'
    spec=read(run/'experiment.json');comparison=read(run/'comparison.json')
    training=read(run/'training_result.json')
    baseline=read(run/'evaluation/baseline_train/metrics.json')
    enhanced=read(run/'evaluation/enhanced_train/metrics.json')
    assert baseline['exporter']==enhanced['exporter']==spec['exporter']
    assert baseline['camera_json']==enhanced['camera_json']==spec['train_cameras']
    assert baseline['metric_sources']==enhanced['metric_sources']
    for record in [spec['config'],spec['trainer'],spec['initial_cloud'],spec['source_m1_cloud'],
                   spec['baseline_checkpoint'],*spec['native_libraries'],*spec['training_images']]:
        assert ref(record['path'])==record, 'Experiment input changed: '+record['path']
    previous={r['name']:r for r in baseline['per_view']}
    rows=[]
    for current in enhanced['per_view']:
        old=previous[current['name']]
        row={'name':current['name']}
        for key in ['PSNR','SSIM','LPIPS']:
            row[key+'_baseline']=old[key];row[key+'_enhanced']=current[key]
            row[key+'_delta']=current[key]-old[key]
        rows.append(row)
    assert len(rows)==56
    with (run/'per_view_comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    improvements={k:sum(r[k+'_delta']*(1 if k!='LPIPS' else -1)>0 for r in rows) for k in ['PSNR','SSIM','LPIPS']}
    trace=[]
    pattern=r'Training iteration (\d+)/32010, time elapsed:([\d.]+)s, ema_loss:([\d.]+), num_points:(\d+)'
    for step,seconds,loss,count in re.findall(pattern,(run/'training.log').read_text(errors='replace')):
        trace.append({'step':int(step),'reported_iteration_seconds':float(seconds),'ema_loss':float(loss),'gaussians':int(count)})
    write(run/'training_trace.json',trace)
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    preview=run/'preview';preview.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].plot([r['step'] for r in trace],[r['ema_loss'] for r in trace]);axes[0].set_ylabel('EMA RGB training loss')
    axes[1].plot([r['step'] for r in trace],[r['gaussians'] for r in trace]);axes[1].set_ylabel('Gaussian count')
    for axis in axes:axis.set_xlabel('Iteration');axis.grid(alpha=.25)
    fig.tight_layout();fig.savefig(preview/'training_curves.png',dpi=160);plt.close(fig)
    views=sorted(previous);chosen=[views[i] for i in np.linspace(0,len(views)-1,4,dtype=int)]
    new={r['name']:r for r in enhanced['per_view']}
    fig,axes=plt.subplots(4,3,figsize=(18,12))
    for row,name in enumerate(chosen):
        records=[('Real training image',previous[name]['reference']),('Original initialization',previous[name]['render']),
                 ('M1-enhanced initialization',new[name]['render'])]
        for col,(title,path) in enumerate(records):
            with Image.open(path) as image:axes[row,col].imshow(image.convert('RGB').resize((768,384)))
            axes[row,col].axis('off')
            label=f'{name}\n{title}'
            if col:label+=f" | PSNR {(previous if col==1 else new)[name]['PSNR']:.3f} dB"
            axes[row,col].set_title(label,fontsize=10)
    fig.tight_layout();fig.savefig(preview/'train_rgb_comparison.png',dpi=140);plt.close(fig)
    tz=ZoneInfo('Asia/Shanghai')
    audit={'same_training_config_and_binary':True,'original_point_xyzrgb_preserved':True,
           'all_56_training_views_used_for_32010_updates':True,'same_evaluator_and_renderer_for_both_models':True,
           'baseline_initialization_and_source_images_unchanged':True}
    result={'run':args.run,'comparison':comparison,'training':training,'improved_view_counts':improvements,
            'checks':audit,'started_beijing':datetime.fromtimestamp(state['started'],tz).isoformat(),
            'completed_beijing':datetime.fromtimestamp(state['completed'],tz).isoformat(),
            'run_directory':str(run),'per_view':rows,
            'experiment_config':ref(run/'experiment.json'),'baseline_metric_file':ref(run/'evaluation/baseline_train/metrics.json'),
            'enhanced_metric_file':ref(run/'evaluation/enhanced_train/metrics.json'),
            'preview_views':chosen,'preview_selection':'Four evenly spaced names in sorted 56-view training list, selected without looking at metric deltas.'}
    write(run/'final_result.json',result)
    write(ROOT/'docs/results/m1_rgb_baseline_20261024.json',result)
    b=comparison['baseline'];n=comparison['enhanced'];d=comparison['delta_enhanced_minus_baseline']
    lines=['# M1 增强点云：同配置 OmniGS RGB 训练与训练集评估','',
        f"完成时间（北京时间）：{result['completed_beijing']}。Run：`{args.run}`。",'',
        '仅替换稀疏初始化点云：3,979 → 11,901。使用原基线训练可执行文件、完全相同的 YAML 配置、56 张真实训练图、2048×1024、32,010 步、seed=0、PF_MODE=0；原有学习率、增密、裁剪、透明度重置、SH 和损失保持一致。',
        '初始化沿用原生 `createFromPcd()`：颜色转 SH，最近邻距离决定球形尺度，单位四元数旋转，初始透明度 0.1。原始 3,979 点的位置和颜色完全保留。训练监督仅使用同一组真实 RGB，相机与训练集划分不变。','',
        '## 全部 56 张训练图的最终指标','',
        '| 指标 | 原始点初始化基线 | M1 增强点初始化 | 新 − 旧 |','|---|---:|---:|---:|',
        f"| PSNR ↑ | {b['PSNR']:.6f} dB | {n['PSNR']:.6f} dB | {d['PSNR']:+.6f} dB |",
        f"| SSIM ↑ | {b['SSIM']:.6f} | {n['SSIM']:.6f} | {d['SSIM']:+.6f} |",
        f"| LPIPS ↓ | {b['LPIPS']:.6f} | {n['LPIPS']:.6f} | {d['LPIPS']:+.6f} |",
        f"| 最终高斯数 | 647,929 | {training['final_gaussians']:,} | {training['final_gaussians']-647929:+,} |",'',
        f"逐视角改善数量：PSNR {improvements['PSNR']}/56、SSIM {improvements['SSIM']}/56、LPIPS {improvements['LPIPS']}/56。完整逐视角数值保存在 run 下的 `per_view_comparison.csv`。",'',
        '## 评估口径','',
        '两模型使用同一原生渲染器重新渲染全部 56 张真实训练图。指标从 2048×1024 浮点 RGB 计算，渲染值裁剪到 [0,1]，不使用有损 JPEG，不缩小图像，不挑选训练子集。各视角指标取等权算术平均。',
        'PSNR 使用全 RGB 像素的均方误差，与原生 OmniGS `psnr` 定义一致。SSIM 使用原有 ODGS 的 11 像素 Gaussian 窗、sigma=1.5。LPIPS 使用 AlexNet v0.1，并将 [0,1] RGB 映射至 [-1,1]。',
        '历史辅助评估脚本另有逐通道平均 PSNR 与直接输入 [0,1] 的 LPIPS 口径。为防止混用，JSON 同时保存 `legacy_channel_mean_PSNR` 和 `legacy_input01_LPIPS`，主表采用上述统一定义。',
        '这些是训练集重建指标；本次没有计算验证集或测试集指标，也没有进行语义训练。','',
        '## 验证与产物','',
        f"- 原训练二进制 SHA256：`{spec['trainer']['sha256']}`。",
        f"- 配置 SHA256：`{spec['config']['sha256']}`，与基线训练记录一致。",
        '- 原生训练日志确认载入 11,901 个初始点；56 个相机均被使用，总更新次数 32,010。',
        '- 原基线模型、M1 增强输入、所有训练 RGB 和原生库的哈希保持不变；两次评估使用同一相机、渲染器与指标实现。',
        f"- 新训练耗时：约 {training['seconds']/60:.2f} 分钟；计时来源：{training.get('timing_source','Controller wall time, including native save and report.')} 旧基线记录来自并发运行，不能据此作训练速度比较。",
        f"- 原生 PLY 透明度字段保存 logit：本模型有 {training.get('opacity_logits',{}).get('positive_infinity',0):,} 个 +∞，经原生 sigmoid 激活为有限透明度 1；其余字段有限，透明度无 NaN。",
        *(['- 本次自动收尾最初误将无限透明度 logit 判为异常；修正校验后直接评估已完成的模型，没有修改模型或重新训练。原始检查日志与恢复脚本保留在 run 目录。'] if training.get('postprocessing_recovery') else []),'',
        '完整服务器目录：','',f'```text\n{run}\n```','',
        '- `model/32010_shutdown/ply/point_cloud/iteration_32010/point_cloud.ply`：最终模型。',
        '- `evaluation/baseline_train/`、`evaluation/enhanced_train/`：全量训练视角 float RGB、PNG、逐视角 CSV 与指标 JSON。',
        '- `experiment.json`、`training_result.json`：配置、数据、二进制、模型校验与相机使用次数。',
        '- `preview/train_rgb_comparison.png`：固定均匀抽取的四个训练视角，真实图 / 基线 / 增强初始化模型。',
        '- `preview/training_curves.png`：新训练的损失与高斯数量曲线。','',
        '机器可读结果：[results/m1_rgb_baseline_20261024.json](results/m1_rgb_baseline_20261024.json)。']
    (ROOT/'docs/M1_RGB_BASELINE_20261024.md').write_text('\n'.join(lines)+'\n')
    print(__import__('json').dumps({'comparison':comparison,'improved_view_counts':improvements,'checks':audit},indent=2))


if __name__=='__main__':main()
