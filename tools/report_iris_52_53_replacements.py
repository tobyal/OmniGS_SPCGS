"""Static review figures and a proposal-only report for the paused IRIS run."""
import csv
import itertools
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_iris_52_53_replacements import ANALYSIS, OUT, RUN, records, xyz
from run_omnigs_sparse18 import read, write, sha


def main():
    by, selected = records()
    variants = {p.parent.name: read(p) for p in (OUT / 'variants').glob('*/comparison.json')}
    original, recommended, wider = [variants[k] for k in ['original', 'B_47_60', 'I_48_60']]
    connections = {x['camera_id']: x for x in read(OUT / 'candidate_connections.json')}
    train = [x for x in selected if x['candidate_role'] == 'train']
    tests = [x for x in selected if x['candidate_role'] == 'test']
    kept = [x for x in train if x['camera_id'] not in ['1-52', '1-53']]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.facecolor': 'white'})

    def hull(ax, ids, color, label, alpha=0):
        xy = np.array([xyz(by[k])[:2] for k in ids])
        poly = xy[ConvexHull(xy).vertices]
        closed = np.vstack([poly, poly[0]])
        ax.plot(closed[:, 0], closed[:, 1], color=color, linestyle='--', linewidth=1.6, label=label)
        if alpha:
            ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=alpha)

    fig, axs = plt.subplots(1, 2, figsize=(17, 10.7), gridspec_kw={'width_ratios': [1.15, 1]}, constrained_layout=True)
    ax = axs[0]
    hull(ax, original['retained_train_ids'], '#a8adb4', 'Original hull: 30.59 m2')
    hull(ax, wider['retained_train_ids'], '#5988ba', 'Alternative 1-48/60 hull: 23.05 m2')
    hull(ax, recommended['retained_train_ids'], '#108478', 'Recommended 1-47/60 hull: 20.03 m2', .08)
    core = [x for x in by.values() if x['is_main_core']]
    ax.scatter([x['x_m'] for x in core], [x['y_m'] for x in core], s=10, color='#d7dbde', zorder=1)
    ax.scatter([x['x_m'] for x in kept], [x['y_m'] for x in kept], s=55, color='#3c627e', label='Unchanged training (16)', zorder=4)
    # Co-located high/low cameras share one plan-view label.
    grouped, consumed = [], set()
    for p in kept:
        if p['camera_id'] in consumed:
            continue
        group = [q for q in kept if np.linalg.norm(xyz(q)[:2] - xyz(p)[:2]) < .075]
        consumed.update(q['camera_id'] for q in group)
        grouped.append(group)
    for group in grouped:
        center = np.mean([xyz(p) for p in group], axis=0)
        label = ' / '.join(x['camera_id'] for x in group)
        offset = (7, 5)
        if group[0]['camera_id'] in ['1-37', '1-40']:
            offset = (-48, 8)
        ax.annotate(label, center[:2], xytext=offset, textcoords='offset points', fontsize=9, color='#294d68')
    for i, p in enumerate(tests):
        ax.scatter(p['x_m'], p['y_m'], marker='D', s=220 if p['camera_id']=='1-61' else 58,
                   facecolor='none' if p['camera_id']=='1-61' else '#e89027', edgecolor='#e89027' if p['camera_id']=='1-61' else 'white', zorder=6,
                   label='Unchanged test (6)' if i == 0 else None)
        offset = (-49, 8) if p['camera_id'] in ['1-61', '1-71'] else (6, 7)
        if p['camera_id']=='1-61':
            offset=(-7,21)
        ax.annotate(p['camera_id'] + ' T', xyz(p)[:2], xytext=offset, textcoords='offset points', fontsize=9, color='#a65704')
    for i, cid in enumerate(['1-47', '1-60']):
        p = by[cid]
        ax.scatter(p['x_m'], p['y_m'], marker='*', s=250, color='#108478', edgecolor='white', zorder=9,
                   label='Recommended replacements (official)' if i == 0 else None)
        ax.annotate(cid + (' H' if cid == '1-47' else ' L'), xyz(p)[:2], xytext=(10, 5) if cid=='1-47' else (-43,-21), textcoords='offset points', color='#087568', fontweight='bold')
    p = by['1-48']
    ax.scatter(p['x_m'], p['y_m'], marker='*', s=180, facecolor='white', edgecolor='#5988ba', linewidth=1.8, zorder=8, label='Alternative 1-48 H (no official)')
    ax.annotate('1-48 H', xyz(p)[:2], xytext=(9, 4), textcoords='offset points', color='#416e9d', fontweight='bold')
    oldcenter = np.mean([xyz(by[c]) for c in ['1-52', '1-53']], axis=0)
    ax.scatter(oldcenter[0], oldcenter[1], marker='x', s=120, linewidth=2.5, color='#c74848', zorder=8, label='Removed 1-52 H / 1-53 L')
    ax.annotate('1-52 H / 1-53 L', oldcenter[:2], xytext=(-28, -24), textcoords='offset points', color='#b73535', fontweight='bold')
    ax.annotate('Reduced direct coverage\nof the right-side equipment area', (30.2, 12.2), xytext=(29.75, 15.1),
                fontsize=10, color='#965043', arrowprops={'arrowstyle': '->', 'color': '#965043'})
    ax.set(xlabel='Official world X (m)', ylabel='Official world Y (m)', title='Plan view: keep 16 training + all 6 test cameras')
    ax.set_xlim(24.7, 32.0)
    ax.set_ylim(8.6, 19.0)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.075), ncol=2, fontsize=8, framealpha=.95)
    ax = axs[1]
    for role, data, color, marker, size in [('Retained train', kept, '#3c627e', 'o', 50), ('Test', tests, '#e89027', 'D', 65),
              ('Removed', [by['1-52'], by['1-53']], '#c74848', 'x', 100),
              ('Recommended', [by['1-47'], by['1-60']], '#108478', '*', 240),
              ('Alternative', [by['1-48']], '#5988ba', '*', 160)]:
        ax.scatter([p['y_m'] for p in data], [p['z_m'] for p in data], s=size, c=color, marker=marker, label=role, zorder=5)
        for p in data:
            dx, dy = (4, 7)
            offsets={'1-52':(-38,-24),'1-53':(-25,16),'1-58':(-17,17),'1-61':(9,-18),
                     '1-66':(9,7),'1-48':(-15,20),'1-47':(9,34),'1-60':(-20,-29),
                     '1-67':(12,10),'1-71':(-34,-20),'1-02':(-30,-11),'1-07':(4,12),
                     '1-01':(-32,-9),'1-08':(4,14),'1-18':(-29,17),'1-19':(9,-2)}
            dx,dy=offsets.get(p['camera_id'],(dx,dy))
            ax.annotate(p['camera_id'], (p['y_m'], p['z_m']), xytext=(dx, dy), textcoords='offset points', fontsize=8,
                        color=color, fontweight='bold' if role in ['Recommended', 'Removed'] else 'normal')
    ax.set(xlabel='Official world Y (m)', ylabel='Official world Z (m)', title='Elevation: 9 lower + 9 upper training views maintained', ylim=(151.5, 152.85))
    ax.legend(loc='center right', fontsize=9)
    ax.text(.03, .48, 'Z is world elevation, not height above the floor.\nAll 6 tests remain inside the train XYZ hull.\nTest positions and images are unchanged.', transform=ax.transAxes, fontsize=10, color='#425669')
    for ax in axs:
        ax.grid(alpha=.18)
        ax.ticklabel_format(useOffset=False, style='plain')
    fig.suptitle('IRIS 18 + 6 | Replacement proposal only | Training remains paused', fontsize=18)
    fig.savefig(OUT / '01_replacement_distribution.png', dpi=180)
    plt.close(fig)

    ids = ['1-52', '1-53', '1-47', '1-60']
    fig, axs = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    for ax, cid in zip(axs.flat, ids):
        p = connections[cid]
        image = Image.open(OUT / 'pool_images' / (p['image_id'] + '.png')).convert('RGB')
        ax.imshow(image)
        ax.set_axis_off()
        ax.set_title(f"{cid} | {p['height_band']} | official | {p['base16_connected_cameras']} base-camera links, {p['base16_geometric_matches']} matches", loc='left',
                     color='#b73535' if cid in ids[:2] else '#087568')
    fig.suptitle('Top: removed pair near occluding equipment | Bottom: recommended open-corridor views', fontsize=16)
    fig.savefig(OUT / '02_original_vs_recommended.png', dpi=160)
    plt.close(fig)

    order = ['original', 'D_50_51', 'C_46_48', 'A_46_47', 'I_48_60', 'B_47_60', 'E_34_35']
    labels = ['Original 1-52/53', 'Nearby 1-50/51', 'Wider 1-46/48', 'Paired height 1-46/47', 'Wider + official 1-48/60', 'Recommended 1-47/60', 'Central density 1-34/35']
    colors = ['#a8adb4', '#b8c8d3', '#b8c8d3', '#b8c8d3', '#5988ba', '#108478', '#b8c8d3']
    fig, axs = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True, sharey=True)
    for ax, key, title in zip(axs, ['points', 'error', 'area'], ['Validated 3+ view seed points', 'Median angular error (degrees)', 'Training XY hull area (m2)']):
        vals = []
        for name in order:
            v = variants[name]
            vals.append(v['sfm']['exported_points'] if key == 'points' else v['sfm']['angular_error_deg_quantiles_min_median_p90_p99_max'][1] if key == 'error' else v['spatial']['train_xy_hull_area_m2'])
        ax.barh(range(len(order)), vals, color=colors)
        for i, v in enumerate(vals):
            ax.text(v, i, (' '+str(v)) if key == 'points' else f' {v:.3f}' if key == 'error' else f' {v:.2f}', va='center', fontsize=10)
        ax.set_xlim(0, max(vals) * 1.25)
        ax.set_title(title)
        ax.grid(axis='x', alpha=.18)
    axs[0].set_yticks(range(len(order)), labels)
    axs[0].invert_yaxis()
    fig.suptitle('Same geometry checks for every proposal | No Gaussian training or test-image matching', fontsize=15)
    fig.savefig(OUT / '03_variant_geometry_comparison.png', dpi=160)
    plt.close(fig)

    # Full-resolution RGB evidence and machine-readable, explicitly non-active proposal.
    (OUT / 'rgb_review').mkdir(exist_ok=True)
    for cid in ['1-52', '1-53', '1-47', '1-60', '1-48', '1-46', '1-50', '1-51']:
        shutil.copy2((OUT / 'pool_images' / (by[cid]['image_id'] + '.png')).resolve(), OUT / 'rgb_review' / (cid + '.png'))
    proposal_rows = []
    for p in selected:
        if p['camera_id'] not in ['1-52', '1-53']:
            proposal_rows.append(p)
        else:
            cid = '1-47' if p['camera_id'] == '1-52' else '1-60'
            q = dict(by[cid], candidate_role='train', role_index=p['role_index'], replaces_camera_id=p['camera_id'])
            proposal_rows.append(q)
    write(OUT / 'proposed_split_47_60.json', {'status': 'proposal_only_not_applied', 'training_started': False,
          'replaces': {'1-52': '1-47', '1-53': '1-60'}, 'source_split_sha256': sha(RUN / 'approved_split.json'),
          'selected_cameras': proposal_rows, 'geometry_analysis': recommended})
    for split, n in [('train', 18), ('test', 6)]:
        names = [x['image_id'] for x in proposal_rows if x['candidate_role'] == split]
        assert len(names) == n
        (OUT / f'proposed_{split}{n}.txt').write_text('\n'.join(names) + '\n')
    csv_rows = []
    for name, v in variants.items():
        q, s = v['sfm'], v['spatial']
        csv_rows.append(dict(variant=name, replacements=';'.join(v['replacements']), points=q['exported_points'],
             supported_train_cameras=q['supported_train_cameras'], median_angular_error_deg=q['angular_error_deg_quantiles_min_median_p90_p99_max'][1],
             p90_angular_error_deg=q['angular_error_deg_quantiles_min_median_p90_p99_max'][2], official_train=s['official_train_count'],
             train_xy_hull_area_m2=s['train_xy_hull_area_m2'], main_core_max_nearest_train_m=s['main_core_nearest_train_m_quantiles'][-1],
             all_tests_inside_xyz_hull=all(t['inside_xyz_hull'] for t in s['test_support'])))
    with (OUT / 'variant_summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    write(OUT / 'variant_summary.json', csv_rows)
    write(OUT / 'all_variant_details.json', variants)
    table = ''.join(f"<tr><td>{label}</td><td>{variants[name]['sfm']['exported_points']}</td><td>{variants[name]['sfm']['supported_train_cameras']}/18</td>"
          f"<td>{variants[name]['sfm']['angular_error_deg_quantiles_min_median_p90_p99_max'][1]:.4f}°</td><td>{variants[name]['spatial']['official_train_count']}/18</td>"
          f"<td>{variants[name]['spatial']['train_xy_hull_area_m2']:.2f}</td></tr>" for name, label in zip(order, labels))
    test_table = ''.join(f"<tr><td>{new['camera_id']}</td><td>{old['nearest_train_3d_m']:.3f}</td><td>{new['nearest_train_3d_m']:.3f}</td>"
          f"<td>{new['same_height_enclosing_triangle']['max_distance_m']:.3f}</td><td>是 / 是</td></tr>" for old, new in zip(original['spatial']['test_support'], recommended['spatial']['test_support']))
    candidates_table = ''.join(f"<tr><td><a href='rgb_review/{cid}.png'>{cid}</a></td><td>{by[cid]['image_id']}</td><td>{by[cid]['z_m']:.3f}</td>"
          f"<td>{'有' if by[cid]['has_official_semantics'] else '无'}</td><td>{connections[cid]['base16_connected_cameras']}</td><td>{connections[cid]['base16_geometric_matches']}</td></tr>" for cid in ['1-52','1-53','1-47','1-60','1-48','1-46','1-50','1-51'])
    report = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>IRIS 1-52/53 替换分析</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f1f5f7;color:#173244;font:16px/1.7 system-ui,'Microsoft YaHei',sans-serif}}main{{max-width:1450px;margin:auto;padding:36px 24px}}h1{{font-size:34px}}h2{{font-size:24px}}section{{background:white;padding:24px;border-radius:10px;margin:24px 0}}img{{display:block;width:100%;height:auto}}a{{color:#087d72}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}td,th{{border-bottom:1px solid #dbe4eb;text-align:left;padding:10px}}.scroll{{overflow:auto}}.note{{background:#fff4e6;border-left:4px solid #e9972d;padding:14px 18px}}.status{{display:inline-block;background:#edf1f5;padding:6px 14px;border-radius:16px}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}summary{{cursor:pointer;color:#087d72}}code{{overflow-wrap:anywhere}}@media(max-width:800px){{.pair{{grid-template-columns:1fr}}main{{padding:20px 12px}}section{{padding:16px}}}}
</style><main><div class="status">训练已暂停 · 只做替换分析 · 未应用新划分</div><h1>建议：1-52 → 1-47；1-53 → 1-60</h1>
<p>保持其余 16 张训练图、全部 6 张测试图不变，训练仍为 9 张高层 + 9 张低层、16/18 张有官方语义。1-47 连接中段与下段通道，1-60 补充下段低层观察；两张都有官方语义标注。</p>
<p><a href="01_replacement_distribution.png">相机分布与高度图</a> · <a href="02_original_vs_recommended.png">原图对照</a> · <a href="03_variant_geometry_comparison.png">几何对比图</a> · <a href="proposed_split_47_60.json">候选划分 JSON</a> · <a href="variant_summary.csv">所有方案指标 CSV</a></p>
<section><h2>为什么替换 1-52、1-53</h2><p>两者之间有 199 对通过几何验证的匹配，但与保留的 16 张训练图没有达到现有验证标准的连接。它们形成独立的两相机子图，在当前“至少 3 个视角支持”的初始点云规则下都没有直接轨迹支持。</p>
<p>实际全景中，近处的大型红色设备、管道和蓝色罐体占据较大画面，主通道主要通过狭窄间隙可见。结合匹配结果，更支持共同可见区域不足的解释；这并不能据此证明官方相机标注错误。</p>
<a href="02_original_vs_recommended.png"><img src="02_original_vs_recommended.png"></a></section>
<section><h2>推荐方案与覆盖取舍</h2><p><b>方案 B：1-47（高层）+1-60（低层），优先用于当前 6 张测试视角。</b> 二者各与保留训练集中的 8 张相机形成可靠连接，分别有 349、686 对通过几何验证的匹配。在共同的 18 张候选训练集中，分别获得 99、153 个合格初始点的观测支持。</p>
<p>初始点从 269 → 453（约 +68.4%），有初始轨迹支持的相机从 16 → 18；球面角重投影误差中位数从 0.0992° → 0.0946°。这些是初始化诊断结果，尚未得到替换方案的 OmniGS 测试预测。</p>
<p class="note">代价：右侧设备背面的直接相机覆盖减少。训练相机 XY 凸包由 30.59 m² 缩至 20.03 m²；原主区所有相机到训练相机的最远距离由 1.58 m 增至 2.93 m。凸包面积只是相机位置指标，不等于可重建表面面积。如果仍需完整重建原 1-52/53 所在右侧区域，需要增加该侧过渡视角，不能只靠这次两张替换保证。</p>
<p><b>备选方案 I：1-48（高层）+1-60（低层）。</b> 向右侧多保留一些位置覆盖：凸包 23.05 m²，421 个合格点、18/18 相机有轨迹支持，角误差中位数 0.0922°；训练官方标注为 15/18，1-48 需后续补标。其近处设备遮挡仍比 1-47 明显。如果右侧范围优先于当前主通道的多视角连接，这一方案更合适。</p>
<a href="01_replacement_distribution.png"><img src="01_replacement_distribution.png"></a></section>
<section><h2>候选图像与标注</h2><div class="scroll"><table><thead><tr><th>相机 / 点击原图</th><th>图像 ID</th><th>世界 Z（m）</th><th>官方语义</th><th>连接到保留相机数</th><th>几何匹配对数</th></tr></thead><tbody>{candidates_table}</tbody></table></div><p>匹配对数是候选与保留 16 张相机的逐对匹配之和，可能包含同一三维点的重复观测，不能当作新增三维点数量。官方状态来自 IRIS 官方标注清单，不使用 SAM 候选分割。</p>
<details><summary>查看右侧覆盖备选 1-48 的真实画面</summary><img loading="lazy" src="rgb_review/1-48.png"></details></section>
<section><h2>不同替换方案的共同条件对比</h2><div class="scroll"><table><thead><tr><th>方案</th><th>合格初始点</th><th>有轨迹支持相机</th><th>角误差中位数</th><th>官方标注训练图</th><th>XY 凸包 m²</th></tr></thead><tbody>{table}</tbody></table></div>
<p>1-50/51 虽然离原位置最近，但几何改善很小；1-34/35 点数更多，但新增观察集中在已有较密的中段，不优先用它们填补中下段过渡。1-46/47 是同位置高低层成对方案，但 1-46 无官方语义，初始化点数也低于推荐方案。</p>
<img loading="lazy" src="03_variant_geometry_comparison.png"></section>
<section><h2>六张测试视角保持不变</h2><p>推荐方案中，6/6 测试相机仍处于训练 XY 与 XYZ 凸包内，且均有同高度三角形包围支持。最近训练相机距离为 0.661–1.174 m；同高度包围三角形的最远支持距离不超过 2.431 m。这里验证的是位置支持，不能替代最终渲染检验。</p>
<div class="scroll"><table><thead><tr><th>测试相机</th><th>原最近训练距离 m</th><th>替换后最近距离 m</th><th>同高三角最远支持 m</th><th>在 XY / XYZ 凸包内</th></tr></thead><tbody>{test_table}</tbody></table></div></section>
<section><h2>检查范围与文件</h2><p>检查 19 个替换候选的真实 2K RGB、官方 RMX 位姿及官方语义记录。复用原 18 张图的描述子与原有匹配；新增候选使用相同 SIFT ULTRA、比率 0.8、球面极线验证阈值 0.5°。各组合单独三角化，只保留至少 3 视角、角残差 ≤0.35°、三角化角 ≥2° 的点。原方案重算精确恢复 269 个点，参数没有随候选改变。</p>
<p>候选池和三角化均未读取测试图像特征；官方位姿固定，无位姿优化。这里只做数据诊断，没有执行 Wan360、OmniGS 或 M2–M5 新训练。原训练进程已暂停在约 4,900 步，保留内存状态，GPU 显存尚未释放；不会自动恢复。</p>
<p><a href="proposed_train18.txt">推荐的 18 张候选训练名单</a> · <a href="proposed_test6.txt">不变的 6 张测试名单</a> · <a href="proposed_split_47_60.json">完整候选划分</a> · <a href="candidate_connections.json">候选几何连接明细</a> · <a href="all_variant_details.json">全部方案数据</a></p>
<details><summary>全部候选原图缩略图</summary>{''.join(f'<img loading="lazy" src="candidate_contact_{i}.jpg">' for i in [1,2,3])}</details></section>
<footer>所有替换方案均为评审候选。当前训练数据和已确认的划分尚未修改。</footer></main></html>"""
    (OUT / 'report.html').write_text(report, encoding='utf-8')
    write(OUT / 'review_verification.json', {'status': 'analysis_complete_training_paused', 'new_training_started': False,
        'proposal_applied': False, 'recommended_replacements': {'1-52':'1-47', '1-53':'1-60'},
        'alternative_replacements': {'1-52':'1-48', '1-53':'1-60'}, 'original_sfm_reproduced_points': original['sfm']['exported_points'],
        'test_images_used_for_features_or_triangulation': 0, 'active_split_sha256': sha(RUN / 'approved_split.json'),
        'recommended_all_6_test_inside_xy_xyz': all(x['inside_xy_hull'] and x['inside_xyz_hull'] for x in recommended['spatial']['test_support'])})
    print('REPLACEMENT_REPORT_COMPLETE', OUT)


if __name__ == '__main__':
    main()
