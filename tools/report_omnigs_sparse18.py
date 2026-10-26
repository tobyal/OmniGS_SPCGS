"""Create a reviewable six-view report from completed OmniGS 18/6 results."""
import argparse
import csv
import html
from pathlib import Path
import re
import shutil
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_omnigs_sparse18 import read, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    run = args.run.resolve()
    out = run / 'report'
    out.mkdir(exist_ok=True)
    (out / 'images').mkdir(exist_ok=True)
    test, train = [read(run / f'evaluation/{split}/metrics.json') for split in ['test', 'train']]
    result = read(run / 'training_result.json')
    quality = read(run / 'sfm_quality.json')
    experiment = read(run / 'experiment.json')
    approved = read(run / 'approved_split.json')
    selected = approved['selected_cameras']
    identities = {x['image_id']: x for x in selected}
    rows = sorted(test['per_view'], key=lambda x: identities[x['name']]['role_index'])
    assert len(rows) == 6
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.facecolor': 'white'})
    previews = []
    for row in rows:
        name, cid = row['name'], row['camera_id']
        gt = np.asarray(Image.open(row['reference']).convert('RGB'))
        rgb = np.fromfile(row['render_float'], dtype='<f4').reshape(3, 1024, 2048).transpose(1, 2, 0)
        pred = np.rint(rgb * 255).astype(np.uint8)
        error = np.mean(np.abs(rgb - gt.astype(np.float32) / 255.), axis=2)
        heat = np.rint(plt.get_cmap('magma')(np.clip(error / .25, 0, 1))[..., :3] * 255).astype(np.uint8)
        for suffix, array in [('gt', gt), ('pred', pred), ('error', heat)]:
            Image.fromarray(array).save(out / 'images' / f'{cid}_{suffix}.png')
        previews.append((row, gt, pred, heat))

    fig, axs = plt.subplots(6, 3, figsize=(18, 19.8), constrained_layout=True)
    for i, (row, gt, pred, heat) in enumerate(previews):
        for j, im in enumerate([gt, pred, heat]):
            axs[i, j].imshow(im)
            axs[i, j].set_axis_off()
        axs[i, 0].set_title(f"{row['camera_id']} | Ground truth | {row['height_band']}", loc='left')
        axs[i, 1].set_title(f"OmniGS | PSNR {row['PSNR']:.2f} | SSIM {row['SSIM']:.3f}", loc='left')
        axs[i, 2].set_title(f"Mean absolute RGB error | LPIPS {row['LPIPS']:.3f}", loc='left')
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, .25), cmap='magma'), ax=axs[:, 2],
                 shrink=.5, aspect=45, label='Mean absolute error [0,1]; values > 0.25 saturated')
    fig.suptitle('IRIS | 18 train / 6 held-out test | original OmniGS | final 32,010 steps', fontsize=17)
    fig.savefig(out / 'comparison_all6.png', dpi=180)
    plt.close(fig)
    fig, axs = plt.subplots(3, 2, figsize=(16, 13.2), constrained_layout=True)
    for ax, (row, gt, pred, heat) in zip(axs.flat, previews):
        ax.imshow(pred)
        ax.set_axis_off()
        ax.set_title(f"{row['camera_id']} | {row['name']} | PSNR {row['PSNR']:.2f} dB", loc='left')
    fig.suptitle('Six held-out test predictions | 2048 x 1024 native renders', fontsize=17)
    fig.savefig(out / 'predictions_all6.png', dpi=160)
    plt.close(fig)
    # A fixed central crop is shown identically for every view, without per-view selection.
    fig, axs = plt.subplots(6, 2, figsize=(14, 22), constrained_layout=True)
    for i, (row, gt, pred, heat) in enumerate(previews):
        for j, im in enumerate([gt, pred]):
            axs[i, j].imshow(im[256:768, 512:1536])
            axs[i, j].set_axis_off()
            axs[i, j].set_title(f"{row['camera_id']} | {'Ground truth' if j == 0 else 'OmniGS'} | fixed central crop", loc='left')
    fig.savefig(out / 'central_crops_all6.png', dpi=160)
    plt.close(fig)

    text = (run / 'training.log').read_text(errors='replace')
    history = np.array([list(map(float, match)) for match in re.findall(
        r'Training iteration (\d+)/32010, time elapsed:([\d.]+)s, ema_loss:([\d.]+), num_points:(\d+)', text)])
    assert len(history) > 10
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    axs[0].plot(history[:, 0], history[:, 2], color='#087e8b', linewidth=1)
    axs[0].set(xlabel='Training iteration', ylabel='EMA training loss', title='RGB training loss')
    axs[1].plot(history[:, 0], history[:, 3] / 1000., color='#e57e26')
    axs[1].set(xlabel='Training iteration', ylabel='Gaussians (thousands)', title='Densification and pruning')
    for ax in axs:
        ax.grid(alpha=.2)
        ax.axvline(15000, color='#777', linestyle='--', linewidth=1)
    fig.savefig(out / 'training_curves.png', dpi=160)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for role, color, marker in [('train', '#087e8b', 'o'), ('test', '#e57e26', '^')]:
        points = [x for x in selected if x['candidate_role'] == role]
        for ax, xk, yk in [(axs[0], 'x_m', 'y_m'), (axs[1], 'y_m', 'z_m')]:
            ax.scatter([x[xk] for x in points], [x[yk] for x in points], color=color, marker=marker,
                       s=62 if role == 'train' else 95, label=f'{role} ({len(points)})', edgecolor='white')
            for p in points:
                ax.annotate(p['camera_id'], (p[xk], p[yk]), xytext=(4, 4), textcoords='offset points', fontsize=8)
    axs[0].set(xlabel='Official world X (m)', ylabel='Official world Y (m)', title='Approved split: plan view')
    axs[0].set_aspect('equal', adjustable='datalim')
    axs[1].set(xlabel='Official world Y (m)', ylabel='Official world Z (m)', title='Approved split: elevation')
    for ax in axs:
        ax.grid(alpha=.2)
        ax.legend(loc='best')
        ax.ticklabel_format(useOffset=False, style='plain')
    fig.savefig(out / 'camera_distribution.png', dpi=180)
    plt.close(fig)

    for source, name in [(run / 'approved_split.json', 'approved_split.json'), (run / 'experiment.json', 'experiment.json'),
              (run / 'sfm_quality.json', 'sfm_quality.json'), (run / 'training_result.json', 'training_result.json'),
              (run / 'baseline.yaml', 'baseline.yaml'), (run / 'dataset/train.txt', 'train18.txt'),
              (run / 'dataset/test.txt', 'test6.txt')]:
        shutil.copy2(source, out / name)
    for split in ['train', 'test']:
        for file in ['metrics.json', 'per_view.csv']:
            shutil.copy2(run / 'evaluation' / split / file, out / (split + '_' + file))
    with (out / 'selected_cameras.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        keys = ['candidate_role', 'camera_id', 'image_id', 'source_group', 'height_band', 'has_official_semantics', 'x_m', 'y_m', 'z_m']
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(selected)
    zero = [identities[Path(name).stem]['camera_id'] for name, count in quality['camera_point_support'].items() if count == 0]
    table = ''.join(f"<tr><td><a href='#{r['camera_id']}'>{r['camera_id']}</a></td><td>{r['name']}</td>"
          f"<td>{'低层' if r['height_band']=='lower' else '高层'}</td><td>{r['PSNR']:.2f}</td>"
          f"<td>{r['SSIM']:.4f}</td><td>{r['LPIPS']:.4f}</td></tr>" for r in rows)
    body = []
    for row in rows:
        cid = row['camera_id']
        body.append(f"""<section id="{cid}" class="view"><h2>{cid} <span>{row['name']}</span></h2>
<p>PSNR <b>{row['PSNR']:.2f} dB</b> · SSIM <b>{row['SSIM']:.4f}</b> · LPIPS <b>{row['LPIPS']:.4f}</b></p>
<div class="pair"><figure><figcaption>真实图像 GT</figcaption><a href="images/{cid}_gt.png"><img loading="lazy" src="images/{cid}_gt.png"></a></figure>
<figure><figcaption>OmniGS 最终预测</figcaption><a href="images/{cid}_pred.png"><img loading="lazy" src="images/{cid}_pred.png"></a></figure></div>
<details><summary>展开滑动对照与误差图</summary><p>拖动分界线：左侧为预测，右侧为 GT。点击上方图片可打开 2048×1024 原尺寸 PNG。</p>
<div class="slider"><img src="images/{cid}_gt.png"><img class="overlay" style="clip-path:inset(0 50% 0 0)" src="images/{cid}_pred.png"></div>
<input type="range" min="0" max="100" value="50" aria-label="预测覆盖范围" oninput="this.previousElementSibling.lastElementChild.style.clipPath='inset(0 '+(100-this.value)+'% 0 0)'">
<img class="error" loading="lazy" src="images/{cid}_error.png"><p>误差图：逐像素 RGB 绝对误差均值，黑→紫→橙→黄表示 0→0.25，超出 0.25 饱和。</p></details></section>""")
    a, b = test['aggregate'], train['aggregate']
    elapsed = read(run / 'training_timing.json')['seconds']
    gap = b['PSNR'] - a['PSNR']
    unsupported = ('、'.join(zero) + ' 在至少 3 视角的初始点云中没有直接轨迹支持；这两张图仍完整参与了 RGB 训练。'
                   if zero else '所有训练相机都有合格的初始点云轨迹支持。')
    report = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>IRIS · OmniGS 18+6 测试效果</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f3f6f8;color:#163040;font:16px/1.65 system-ui,'Microsoft YaHei',sans-serif}}main{{max-width:1540px;margin:auto;padding:36px 28px}}h1{{font-size:36px;line-height:1.25}}h2{{font-size:24px;margin-top:0}}h2 span{{font-size:16px;font-weight:400;color:#5e7180}}p{{max-width:1120px}}a{{color:#087e8b}}.eyebrow{{color:#087e8b;font-weight:700;letter-spacing:2px}}.chips,.metrics{{display:flex;gap:12px;flex-wrap:wrap}}.chips a{{background:white;padding:5px 16px;border-radius:18px;text-decoration:none}}.metrics div{{background:white;border-top:4px solid #087e8b;border-radius:8px;padding:14px 24px;min-width:180px}}.metrics b{{display:block;font-size:30px}}section,.card{{background:white;margin:24px 0;padding:24px;border-radius:12px;box-shadow:0 3px 20px #102d3e08}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}figure{{margin:0}}figcaption{{font-weight:600;margin-bottom:8px}}img{{display:block;width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}td,th{{padding:10px 12px;border-bottom:1px solid #dfe7ec;text-align:left}}.scroll{{overflow-x:auto}}.note{{border-left:4px solid #e57e26;padding:8px 20px;background:#fff7ed}}.slider{{position:relative;width:100%}}.overlay{{position:absolute;inset:0}}input[type=range]{{width:100%}}summary{{cursor:pointer;color:#087e8b;padding:12px 0}}.error{{max-width:1000px}}code{{overflow-wrap:anywhere;font-size:13px}}footer{{color:#5e7180}}@media(max-width:850px){{main{{padding:20px 12px}}.pair{{grid-template-columns:1fr}}h1{{font-size:28px}}section,.card{{padding:16px}}}}
</style></head><body><main><div class="eyebrow">IRIS / ORIGINAL OMNIGS / HELD-OUT RGB</div>
<h1>18 张训练，6 张测试：最终预测效果</h1><p>严格沿用确认的图像名单，无验证集。使用 18 张真实训练图像从头训练原版 OmniGS，最终 32,010 步模型在 6 张测试视角渲染；本次不启用 Wan360、语义监督或 SPC-GS M2–M5。</p>
<div class="metrics"><div>测试 PSNR ↑<b>{a['PSNR']:.2f} dB</b></div><div>测试 SSIM ↑<b>{a['SSIM']:.4f}</b></div><div>测试 LPIPS ↓<b>{a['LPIPS']:.4f}</b></div><div>分辨率 / 训练步数<b style="font-size:24px">2048×1024 / 32,010</b></div></div>
<p><a href="comparison_all6.png">六视角 GT / 预测 / 误差总览</a> · <a href="predictions_all6.png">只看六张预测图</a> · <a href="central_crops_all6.png">固定中央裁剪对照</a> · <a href="review_bundle.zip">下载评审包</a></p>
<nav class="chips">{''.join(f'<a href="#{r["camera_id"]}">{r["camera_id"]}</a>' for r in rows)}</nav>
<section><h2>六张测试图的完整结果</h2><div class="scroll"><table><thead><tr><th>相机</th><th>图像 ID</th><th>高度层</th><th>PSNR ↑</th><th>SSIM ↑</th><th>LPIPS ↓</th></tr></thead><tbody>{table}<tr><th colspan="3">6 张等权平均</th><th>{a['PSNR']:.2f}</th><th>{a['SSIM']:.4f}</th><th>{a['LPIPS']:.4f}</th></tr></tbody></table></div>
<p>指标以最终模型的浮点渲染和原始 2K PNG 计算，覆盖整个全景，包括极区与三脚架；无裁剪、无掩膜。LPIPS 使用 AlexNet v0.1，输入范围为 [-1,1]。所有六张均有官方语义标注，但本次只训练 RGB。</p></section>
{''.join(body)}
<section><h2>训练与初始化核查</h2><p>复用本次已确认划分中，仅由这 18 张训练图进行 SIFT ULTRA 匹配、固定官方位姿三角化得到的初始点云，共 <b>{quality['exported_points']:,}</b> 个至少 3 视角支持的真实点；所有高斯参数从头优化，最终得到 <b>{result['final_gaussians']:,}</b> 个高斯。训练耗时约 <b>{elapsed/60:.1f} 分钟</b>。未加载旧实验检查点，测试图观测和语义标签均未用于训练或初始化。</p>
<p>这版划分按已确认的轨迹插值与图像重叠分析选定；选图阶段使用过测试图的特征匹配来检查训练视角支持。测试图没有参与初始点云三角化或高斯优化，本报告用于评估这版工业区域稀疏输入实验的可行性。</p>
<p class="note">{unsupported} 因此空间距离近和高度覆盖充分，并不自动等于具有足够的共同可见表面；本次结果是检查这版划分可行性的实测依据。</p>
<p>训练集重建均值：PSNR {b['PSNR']:.2f} dB / SSIM {b['SSIM']:.4f} / LPIPS {b['LPIPS']:.4f}。训练与测试 PSNR 差为 {gap:.2f} dB；训练指标用于诊断拟合，不作为测试效果。未根据测试结果选择检查点或调整本次训练参数。</p>
<img loading="lazy" src="training_curves.png"><details><summary>查看训练相机分布与名单</summary><img src="camera_distribution.png"><p><a href="train18.txt">18 张训练名单</a> · <a href="test6.txt">6 张测试名单</a> · <a href="selected_cameras.csv">24 张相机信息 CSV</a> · <a href="approved_split.json">确认的完整划分 JSON</a></p></details></section>
<section><h2>文件与可复现记录</h2><p><a href="test_per_view.csv">逐测试图指标 CSV</a> · <a href="test_metrics.json">测试指标定义和明细</a> · <a href="train_metrics.json">训练集诊断</a> · <a href="sfm_quality.json">初始点云质量</a> · <a href="experiment.json">实验配置与文件哈希</a> · <a href="baseline.yaml">原版基线参数</a></p><p>服务器实验目录：<code>{html.escape(str(run))}</code></p><p>最终模型：<code>{html.escape(result['checkpoint']['path'])}</code></p></section>
<footer>固定划分 · 从头训练 · 全部 6 张测试 · 最终模型一次评估。PNG 为显示用 8 位量化，指标来自 float32 原始渲染。</footer></main></body></html>"""
    (out / 'index.html').write_text(report, encoding='utf-8')
    write(out / 'verification.json', {'test_names_exactly_match_approved': True, 'test_images': len(rows),
        'train_images': train['views'], 'all_18_train_cameras_used': len(result['per_camera_updates']) == 18,
        'total_training_updates': sum(result['per_camera_updates'].values()),
        'metrics_from_float_rgb': True, 'initial_points': quality['exported_points'],
        'initial_cloud_cameras_without_tracks': zero, 'test_aggregate': a, 'train_aggregate': b})
    with zipfile.ZipFile(out / 'review_bundle.zip', 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for path in sorted(out.rglob('*')):
            if path.is_file() and path.suffix != '.zip':
                archive.write(path, path.relative_to(out))
    print('REPORT_COMPLETE', out, flush=True)


if __name__ == '__main__':
    main()
