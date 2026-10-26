"""Compare all six held-out SPaGS and OmniGS renders under the same protocol."""
import argparse
import csv
import html
import json
from pathlib import Path
import shutil
import zipfile

from run_omnigs_sparse18 import read, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    args = p.parse_args()
    import numpy as np
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    run, baseline = args.run.resolve(), args.reference.resolve()
    out = run / 'report'
    (out / 'images').mkdir(parents=True, exist_ok=True)
    spa = read(run / 'evaluation/test/metrics.json')
    omni = read(baseline / 'evaluation/test/metrics.json')
    spa_train = read(run / 'evaluation/train/metrics.json')
    omni_train = read(baseline / 'evaluation/train/metrics.json')
    selected = read(run / 'approved_split.json')['selected_cameras']
    assert selected == read(baseline / 'approved_split.json')['selected_cameras']
    assert spa['views'] == omni['views'] == 6
    assert [x['sha256'] for x in spa['metric_sources']] == [x['sha256'] for x in omni['metric_sources']]
    assert read(run / 'experiment.json')['initial_cloud']['sha256'] == read(baseline / 'experiment.json')['initial_cloud']['sha256']
    by_name = {x['name']: x for x in omni['per_view']}
    assert set(by_name) == {x['name'] for x in spa['per_view']}
    order = {x['image_id']: x['split_index'] for x in selected}
    rows = sorted(spa['per_view'], key=lambda x: order[x['name']])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'savefig.facecolor': 'white',
                         'axes.spines.top': False, 'axes.spines.right': False})
    comparisons, previews = [], []
    for s in rows:
        o = by_name[s['name']]
        cid = s['camera_id']
        gt = np.asarray(Image.open(s['reference']).convert('RGB'))
        assert np.array_equal(gt, np.asarray(Image.open(o['reference']).convert('RGB')))
        floats = [np.fromfile(x['render_float'], dtype='<f4').reshape(3, 1024, 2048).transpose(1, 2, 0) for x in [o, s]]
        preds = [np.rint(x*255).astype(np.uint8) for x in floats]
        errors = [np.abs(x - gt.astype(np.float32)/255.).mean(axis=2) for x in floats]
        heats = [np.rint(plt.get_cmap('magma')(np.clip(x/.25, 0, 1))[..., :3]*255).astype(np.uint8) for x in errors]
        for suffix, im in zip(['gt', 'omnigs', 'spags', 'omnigs_error', 'spags_error'], [gt, *preds, *heats]):
            Image.fromarray(im).save(out / 'images' / f'{cid}_{suffix}.png')
        r = dict(camera_id=cid, image_id=s['name'], height_band=s['height_band'])
        for key in ['PSNR', 'SSIM', 'LPIPS']:
            r.update({key+'_OmniGS': o[key], key+'_SPaGS': s[key], key+'_delta': s[key]-o[key]})
        comparisons.append(r)
        previews.append((r, gt, preds, heats))
    with (out / 'comparison_per_view.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(comparisons[0]))
        writer.writeheader(); writer.writerows(comparisons)
    a, b = omni['aggregate'], spa['aggregate']
    delta = {k: b[k]-a[k] for k in a}
    write(out / 'comparison_metrics.json', dict(OmniGS=a, SPaGS=b, delta_SPAGS_minus_OmniGS=delta,
        per_view=comparisons, train_OmniGS=omni_train['aggregate'], train_SPaGS=spa_train['aggregate'],
        protocol=dict(train=18, test=6, validation=0, width=2048, height=1024, iterations=32010,
        seed=0, initial_points=1347, pixels='Full ERP, no crop or mask', LPIPS='AlexNet v0.1',
        initial_cloud_sha256=read(run / 'experiment.json')['initial_cloud']['sha256'],
        native_method_specific_settings_preserved=True)))
    for crop, filename, figsize in [(False, 'comparison_all6.png', (18, 20)), (True, 'central_crops_all6.png', (18, 20))]:
        fig, axs = plt.subplots(6, 3, figsize=figsize, constrained_layout=True)
        for i, (row, gt, preds, heats) in enumerate(previews):
            for j, im in enumerate([gt, *preds]):
                axs[i, j].imshow(im[256:768, 512:1536] if crop else im)
                axs[i, j].set_axis_off()
                method = ['Ground truth', 'OmniGS', 'SPaGS'][j]
                detail = '' if j == 0 else f" | PSNR {row['PSNR_'+method]:.2f} | SSIM {row['SSIM_'+method]:.3f}"
                axs[i, j].set_title(f"{row['camera_id']} | {method}{detail}", loc='left')
        fig.suptitle('IRIS | same 18 train / 6 test | 32,010 steps | ' + ('fixed central crop' if crop else 'full panorama'), fontsize=16)
        fig.savefig(out / filename, dpi=160)
        plt.close(fig)
    fig, axs = plt.subplots(6, 2, figsize=(14, 20), constrained_layout=True)
    for i, (row, gt, preds, heats) in enumerate(previews):
        for j, im in enumerate(heats):
            axs[i, j].imshow(im); axs[i, j].set_axis_off()
            method = ['OmniGS', 'SPaGS'][j]
            axs[i, j].set_title(f"{row['camera_id']} | {method} absolute RGB error", loc='left')
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, .25), cmap='magma'), ax=axs[:, 1], shrink=.5,
                 label='Mean absolute RGB error; values above 0.25 saturated')
    fig.savefig(out / 'errors_all6.png', dpi=140); plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, metric in zip(axs, ['PSNR', 'SSIM', 'LPIPS']):
        bars = ax.bar(['OmniGS', 'SPaGS'], [a[metric], b[metric]], color=['#64748b', '#087e8b'], width=.55)
        ax.bar_label(bars, fmt='%.4f' if metric != 'PSNR' else '%.2f', padding=4)
        ax.set_ylim(0, max(a[metric], b[metric])*1.2)
        ax.set_title(metric + (' (lower is better)' if metric == 'LPIPS' else ' (higher is better)'))
    fig.savefig(out / 'metrics_comparison.png', dpi=160); plt.close(fig)
    history = [json.loads(x) for x in (run / 'training_history.jsonl').read_text().splitlines()]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axs[0].plot([x['step'] for x in history], [x['loss_last'] for x in history], color='#087e8b', linewidth=.8)
    axs[0].set(xlabel='Iteration', ylabel='Last sampled training loss', title='SPaGS training loss (not averaged)')
    axs[1].plot([x['step'] for x in history], [x['gaussians']/1000 for x in history], color='#087e8b')
    axs[1].set(xlabel='Iteration', ylabel='Gaussians (thousands)', title='SPaGS densification and pruning')
    fig.savefig(out / 'training_curves.png', dpi=160); plt.close(fig)
    for source in ['approved_split.json', 'experiment.json', 'adapter_validation.json', 'camera_id_adapter.json',
                   'training_result.json', 'training_timing.json', 'spags.yaml']:
        shutil.copy2(run / source, out / source)
    for method, root in [('spags', run), ('omnigs', baseline)]:
        for split in ['train', 'test']:
            shutil.copy2(root / f'evaluation/{split}/metrics.json', out / f'{method}_{split}_metrics.json')
    for split in ['train', 'test']:
        shutil.copy2(run / f'dataset/{split}.txt', out / f'{split}.txt')
    build_html(out, run, baseline, comparisons, a, b, delta, spa_train, omni_train)
    with zipfile.ZipFile(out / 'comparison_bundle.zip', 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        for path in sorted(out.rglob('*')):
            if path.is_file() and path.suffix != '.zip':z.write(path, path.relative_to(out))
    print('COMPARISON_REPORT_COMPLETE', out, flush=True)


def build_html(out, run, baseline, rows, a, b, delta, spa_train, omni_train):
    summary = ''.join(f'<tr><th>{method}</th><td>{m["PSNR"]:.2f}</td><td>{m["SSIM"]:.4f}</td><td>{m["LPIPS"]:.4f}</td></tr>'
                      for method, m in [('OmniGS', a), ('SPaGS', b), ('SPaGS − OmniGS', delta)])
    table = ''.join(f'<tr><td><a href="#{r["camera_id"]}">{r["camera_id"]}</a></td>'
        f'<td>{r["PSNR_OmniGS"]:.2f}</td><td>{r["PSNR_SPaGS"]:.2f}</td><td>{r["PSNR_delta"]:+.2f}</td>'
        f'<td>{r["SSIM_OmniGS"]:.4f} / {r["SSIM_SPaGS"]:.4f}</td>'
        f'<td>{r["LPIPS_OmniGS"]:.4f} / {r["LPIPS_SPaGS"]:.4f}</td></tr>' for r in rows)
    sections = []
    for r in rows:
        cid = r['camera_id']
        panels = ''.join(f'<figure><figcaption>{label}</figcaption><a href="images/{cid}_{suffix}.png">'
            f'<img loading="lazy" src="images/{cid}_{suffix}.png"></a></figure>' for suffix, label in
            [('gt', '真实图 GT'), ('omnigs', 'OmniGS'), ('spags', 'SPaGS')])
        sections.append(f'''<section id="{cid}"><h2>{cid} <small>{r['image_id']}</small></h2>
<p>PSNR：OmniGS {r['PSNR_OmniGS']:.2f} → SPaGS {r['PSNR_SPaGS']:.2f} dB（{r['PSNR_delta']:+.2f}）；
SSIM：{r['SSIM_OmniGS']:.4f} → {r['SSIM_SPaGS']:.4f}；LPIPS：{r['LPIPS_OmniGS']:.4f} → {r['LPIPS_SPaGS']:.4f}。</p>
<div class="triple">{panels}</div><details><summary>SPaGS 与真值滑动对照</summary>
<p>左侧为 SPaGS，右侧为真值；拖动滑块检查细节。上方每张图片可点击打开原尺寸 PNG。</p>
<div class="slider"><img src="images/{cid}_gt.png"><img class="overlay" style="clip-path:inset(0 50% 0 0)" src="images/{cid}_spags.png"></div>
<input type="range" min="0" max="100" value="50" aria-label="SPaGS 覆盖范围" oninput="this.previousElementSibling.lastElementChild.style.clipPath='inset(0 '+(100-this.value)+'% 0 0)'"></details></section>''')
    tr = read(run / 'training_result.json')
    elapsed = read(run / 'training_timing.json')['seconds']
    omtr = read(baseline / 'training_result.json')
    st, ot = spa_train['aggregate'], omni_train['aggregate']
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>IRIS 18+6 · OmniGS / SPaGS 对比</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f2f5f8;color:#183342;font:16px/1.7 system-ui,'Microsoft YaHei',sans-serif}}main{{max-width:1700px;margin:auto;padding:32px 28px}}h1{{font-size:36px;line-height:1.3}}h2{{font-size:24px;margin-top:0}}small{{font-weight:400;font-size:15px;color:#64748b}}a{{color:#087e8b}}section{{background:white;border-radius:12px;padding:24px;margin:24px 0}}.triple{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}figure{{margin:0}}figcaption{{font-weight:650;margin:0 0 8px}}img{{width:100%;display:block;height:auto}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}td,th{{text-align:left;padding:10px 12px;border-bottom:1px solid #dce4ea}}.scroll{{overflow:auto}}nav{{display:flex;gap:14px;flex-wrap:wrap}}nav a{{background:white;border-radius:18px;padding:5px 16px;text-decoration:none}}.eyebrow{{color:#087e8b;font-weight:700;letter-spacing:2px}}.slider{{position:relative}}.overlay{{position:absolute;inset:0}}input[type=range]{{width:100%}}summary{{cursor:pointer;color:#087e8b;padding:14px 0}}code{{font-size:13px;overflow-wrap:anywhere}}.note{{background:#f1f7f7;border-left:4px solid #087e8b;padding:12px 18px}}@media(max-width:850px){{main{{padding:18px 12px}}section{{padding:16px}}.triple{{grid-template-columns:1fr}}h1{{font-size:28px}}}}
</style></head><body><main><div class="eyebrow">IRIS / SAME INPUTS / TWO NATIVE METHODS</div>
<h1>OmniGS 与 SPaGS：同一版 18 张训练、6 张测试</h1>
<p>两种方法均从头训练 32,010 步，使用相同图像、官方相机位姿和 1,347 个训练集 SfM 点，无验证集。分辨率 2048×1024，种子 0，黑背景，SH degree 3。SPaGS 使用独立原生实现，不启用 SPC-GS 的语义训练流程。</p>
<p><a href="comparison_all6.png">六图 GT / OmniGS / SPaGS 静态总览</a> · <a href="central_crops_all6.png">固定中央区域放大对照</a> · <a href="errors_all6.png">统一色标误差图</a> · <a href="comparison_bundle.zip">下载完整对比包</a></p>
<nav>{''.join(f'<a href="#{r["camera_id"]}">{r["camera_id"]}</a>' for r in rows)}</nav>
<section><h2>六张测试图等权平均</h2><table><thead><tr><th>方法</th><th>PSNR ↑</th><th>SSIM ↑</th><th>LPIPS ↓</th></tr></thead><tbody>{summary}</tbody></table>
<p>差值为 SPaGS − OmniGS；PSNR / SSIM 为正、LPIPS 为负表示改善。完整 ERP 指标包含极区与三脚架，无裁剪、无掩膜。两者调用同一份指标代码，使用 float32 RGB，SSIM 采用 11×11 高斯窗口，LPIPS 为 AlexNet v0.1。</p>
<img src="metrics_comparison.png"></section>
<section><h2>逐测试视角比较</h2><div class="scroll"><table><thead><tr><th>相机</th><th>OmniGS PSNR</th><th>SPaGS PSNR</th><th>Δ PSNR</th><th>SSIM：OmniGS / SPaGS</th><th>LPIPS：OmniGS / SPaGS</th></tr></thead><tbody>{table}</tbody></table></div></section>
{''.join(sections)}
<section><h2>共同设置与方法差异</h2><p>图像像素逐一核对完全一致；SPaGS 原生数据加载器中的相机位姿转换误差小于 2×10⁻⁶，点云坐标与颜色通过核对。两者均不加载旧模型，不使用测试图三角化或训练。SPaGS 最终模型已保存并通过原生加载器重新载入后渲染。</p>
<p class="note">共同输入与训练预算固定，方法自身机制保留：SPaGS 使用原生球面渲染、3D 过滤、距离缩放以及默认增密（开始于 200 步，阈值 5×10⁻⁵）；OmniGS 使用其既有基线配置（开始于 500 步，阈值 2×10⁻⁴）。本次比较的是两种方法的原生配置，不是仅替换光栅器的消融。</p>
<p>SPaGS 训练约 {elapsed/60:.1f} 分钟，最终 {tr['final_gaussians']:,} 个高斯；OmniGS 最终 {omtr['final_gaussians']:,} 个高斯。SPaGS 训练集 PSNR / SSIM / LPIPS：{st['PSNR']:.2f} / {st['SSIM']:.4f} / {st['LPIPS']:.4f}；OmniGS：{ot['PSNR']:.2f} / {ot['SSIM']:.4f} / {ot['LPIPS']:.4f}。训练集指标仅作拟合诊断。</p>
<p>这一划分此前按轨迹插值及真实图像重叠选定，选图时查看过测试图特征匹配。本次参数与最终检查点没有根据测试结果调优。</p>
<img loading="lazy" src="training_curves.png"></section>
<section><h2>数据与复现记录</h2><p><a href="comparison_per_view.csv">逐图对比 CSV</a> · <a href="comparison_metrics.json">指标汇总 JSON</a> · <a href="spags_test_metrics.json">SPaGS 指标明细</a> · <a href="omnigs_test_metrics.json">OmniGS 指标明细</a> · <a href="spags.yaml">SPaGS 配置</a> · <a href="experiment.json">运行与代码记录</a> · <a href="adapter_validation.json">数据一致性核查</a> · <a href="approved_split.json">18/6 划分</a></p>
<p>SPaGS：<code>{html.escape(str(run))}</code><br>OmniGS：<code>{html.escape(str(baseline))}</code></p></section>
<p>全部六张测试 · 最终模型 · 相同输入和统一评测。PNG 仅作展示，指标从浮点渲染计算。</p></main></body></html>'''
    (out / 'index.html').write_text(page, encoding='utf-8')


if __name__ == '__main__':
    main()
