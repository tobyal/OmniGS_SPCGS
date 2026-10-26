"""Static plots and an HTML review for the official-only IRIS split."""
import csv
from pathlib import Path
import shutil
import sys
import zipfile
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.patches import Rectangle
from scipy.spatial import ConvexHull

sys.path.insert(0,str(Path(__file__).resolve().parent))
from select_iris_official_overlap import OUT, CACHE, ANALYSIS, Diagnostics, read, write, xyz, ROOT


def main():
    d=Diagnostics(); s=read(OUT/'selected_split.json'); tr=s['train_ids']; te=s['test_ids']
    support={x['test']:x for x in read(OUT/'selected_test_feature_support.json')}
    structure={x['test']:x for x in read(OUT/'test_to_train_structure_support.json')}
    brackets={x['test']:x for x in s['test_brackets']}
    comparison=read(OUT/'design_comparison.json')
    review=OUT/'review';review.mkdir(exist_ok=True);(review/'rgb').mkdir(exist_ok=True)
    train_color,test_color='#147c9c','#df7c16'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def image_path(cid):return CACHE/'pool_images'/(d.by[cid]['image_id']+'.png')
    def ids_features(t):
        counts={}
        for c in tr:
            for k in set(d.match(t,c)[:,0].tolist()):counts[k]=counts.get(k,0)+1
        return sorted(k for k,n in counts.items() if n>=2)

    # Main extent and separate height layers prevent high/low cameras hiding one another.
    fig,axs=plt.subplots(1,3,figsize=(20,8.3),gridspec_kw={'width_ratios':[.9,1.2,1.2]},constrained_layout=True)
    ax=axs[0]
    for official in [False,True]:
        rows=[r for r in d.rows if r['group']==1 and r['has_official_semantics']==official]
        ax.scatter([r['x_m'] for r in rows],[r['y_m'] for r in rows],s=22,
                   facecolor='#b7c3cc' if official else 'white',edgecolor='#b7c3cc',
                   label='Other official cameras' if official else 'No official semantics')
    p=np.array([xyz(d.by[c]) for c in tr]); hull=ConvexHull(p[:,:2]);poly=p[hull.vertices,:2];poly=np.vstack([poly,poly[0]])
    ax.fill(poly[:,0],poly[:,1],color=train_color,alpha=.12)
    ax.plot(poly[:,0],poly[:,1],color=train_color,lw=1.2)
    for ids,col,marker,label in [(tr,train_color,'o',f'Train ({len(tr)})'),(te,test_color,'D',f'Test ({len(te)})')]:
        ax.scatter([d.by[c]['x_m'] for c in ids],[d.by[c]['y_m'] for c in ids],s=36,c=col,marker=marker,label=label,zorder=4)
    for ids,label,offset in [(['1-52','1-53'],'1-52 / 1-53',(-65,-18)),(['1-78','2-01'],'1-78 / 2-01',(7,-5))]:
        pp=np.mean([xyz(d.by[c]) for c in ids],axis=0)
        ax.scatter(*pp[:2],s=60,marker='x',c='#aa6d6d');ax.annotate(label,pp[:2],xytext=offset,textcoords='offset points',color='#8c5151',fontsize=9)
    ax.set(title='Context: compact part of xphase_1',xlabel='World X (m)',ylabel='World Y (m)',xlim=(24.6,31.8),ylim=(8.7,18.5))
    ax.set_aspect('equal');ax.legend(loc='upper left',fontsize=8)
    offsets={'1-24':(-44,9),'1-25':(-44,9),'1-26':(-13,12),'1-27':(-13,12),'1-28':(7,7),'1-29':(7,7),
             '1-34':(7,-17),'1-35':(7,-17),'1-36':(-43,7),'1-37':(-43,7),'1-44':(8,8),'1-47':(8,-17),
             '1-58':(8,-15),'1-59':(8,-15),'1-60':(7,9),'1-61':(7,9),'1-62':(7,9),'1-63':(7,9),
             '1-64':(-45,8),'1-65':(-45,8),'1-66':(-25,-20),'1-67':(-25,-20),'1-68':(-19,-22),'1-69':(-19,-22),
             '1-70':(8,-9),'1-71':(8,-9)}
    for ax,band,label in zip(axs[1:],['lower','upper'],['Lower layer','Upper layer']):
        for t,b in brackets.items():
            if d.by[t]['height_band_relative_to_core']!=band:continue
            pp=np.array([xyz(d.by[c]) for c in [b['before'],t,b['after']]])
            ax.plot(pp[:,0],pp[:,1],c=test_color,lw=2,alpha=.7,zorder=2)
        ids=[c for c in tr+te if d.by[c]['height_band_relative_to_core']==band]
        for c in ids:
            r=d.by[c];test=c in te;col=test_color if test else train_color
            ax.scatter(r['x_m'],r['y_m'],c=col,s=100 if test else 70,marker='D' if test else 'o',zorder=4)
            ax.annotate(c+(' T' if test else ''),xyz(r)[:2],xytext=offsets[c],textcoords='offset points',
                        color='#a34f00' if test else '#125a72',fontsize=10,fontweight='bold' if test else 'normal')
        nt=sum(c in tr for c in ids);ne=sum(c in te for c in ids)
        ax.set(title=f'{label}: {nt} train + {ne} test (all official)\nZ = {min(d.by[c]["z_m"] for c in ids):.3f} to {max(d.by[c]["z_m"] for c in ids):.3f} m',
               xlabel='World X (m)',ylabel='World Y (m)',xlim=(25.05,29.6),ylim=(11.0,15.4))
        ax.set_aspect('equal')
    for ax in axs:ax.grid(alpha=.2);ax.ticklabel_format(useOffset=False,style='plain')
    fig.suptitle(f'IRIS official-only candidate | {len(tr)} train + {len(te)} test | no validation',fontsize=18)
    fig.supxlabel('Blue circles: train. Orange diamonds: test. Orange segments: same-height chronological training neighbors. Selected cameras: 26/26 official.',fontsize=11)
    fig.savefig(review/'01_camera_distribution.png',dpi=180);plt.close(fig)

    # Two elevations plus a per-camera strip show every measured Z without XY overlap.
    fig,axs=plt.subplots(1,2,figsize=(18,6),gridspec_kw={'width_ratios':[1,1.35]},constrained_layout=True)
    ax=axs[0]
    for ids,col,marker,label in [(tr,train_color,'o','Train'),(te,test_color,'D','Test')]:
        ax.scatter([d.by[c]['y_m'] for c in ids],[d.by[c]['z_m'] for c in ids],s=55,c=col,marker=marker,label=label,zorder=3)
    for t,b in brackets.items():
        pp=np.array([xyz(d.by[c]) for c in [b['before'],t,b['after']]])
        ax.plot(pp[:,1],pp[:,2],c=test_color,lw=1.2,alpha=.5)
        ax.annotate(t,(d.by[t]['y_m'],d.by[t]['z_m']),xytext=(3,11 if d.by[t]['height_band_relative_to_core']=='upper' else -18),textcoords='offset points',color='#a34f00')
    ax.set(xlabel='World Y (m)',ylabel='World Z (m)',title='Actual measured elevation along the region',ylim=(151.53,152.73))
    ax.legend(loc='center right')
    ax=axs[1];ordered=sorted(tr+te)
    for i,c in enumerate(ordered):
        test=c in te
        ax.scatter(i,d.by[c]['z_m'],c=test_color if test else train_color,marker='D' if test else 'o',s=65)
    ax.set_xticks(range(len(ordered)),ordered,rotation=65,ha='right',fontsize=9)
    ax.set(xlabel='Camera ID (chronological order)',ylabel='World Z (m)',title='Every selected camera | 11 upper + 9 lower train / 3 + 3 test',ylim=(151.53,152.73))
    for ax in axs:ax.grid(alpha=.18);ax.ticklabel_format(axis='y',useOffset=False,style='plain')
    fig.suptitle('Height coverage | Z is survey-world elevation, not height above floor',fontsize=17)
    fig.savefig(review/'02_height_distribution.png',dpi=180);plt.close(fig)

    mat=np.array([[len(d.match(t,c)) for c in tr] for t in te])
    fig,ax=plt.subplots(figsize=(19,5.6),constrained_layout=True)
    im=ax.imshow(mat,cmap='YlGnBu',norm=PowerNorm(.6,vmin=0,vmax=500),aspect='auto')
    for i,t in enumerate(te):
        for j,c in enumerate(tr):
            v=mat[i,j];ax.text(j,i,str(v) if v else '—',ha='center',va='center',fontsize=9,color='white' if v>180 else '#16303c')
            if c in [brackets[t]['before'],brackets[t]['after']]:ax.add_patch(Rectangle((j-.48,i-.48),.96,.96,fill=False,edgecolor='#f19b32',lw=2.5))
    ax.set_xticks(range(len(tr)),tr,rotation=45,ha='right');ax.set_yticks(range(len(te)),[t+(' L' if d.by[t]['height_band_relative_to_core']=='lower' else ' H') for t in te])
    ax.set(xlabel='Officially annotated training cameras',ylabel='Test cameras',title='Actual test-to-train geometric matches | orange frames: trajectory anchors')
    fig.colorbar(im,ax=ax,label='Geometrically verified pair matches',shrink=.8)
    fig.savefig(review/'03_train_test_match_matrix.png',dpi=180);plt.close(fig)

    fig,axs=plt.subplots(len(te),3,figsize=(18,19.5),constrained_layout=True)
    for i,t in enumerate(te):
        b=brackets[t]
        for j,c in enumerate([b['before'],t,b['after']]):
            ax=axs[i,j];ax.imshow(Image.open(image_path(c)));ax.axis('off')
            role=['TRAIN before','TEST between','TRAIN after'][j]
            cnt='' if j==1 else f' | {b["matches_before" if j==0 else "matches_after"]} matches'
            ax.set_title(f'{c} | {role}{cnt}',loc='left',fontsize=11,color='#a34f00' if j==1 else '#125a72')
    fig.suptitle('Six official-only trajectory triplets | inspect shared structures and occlusion directly',fontsize=17)
    fig.savefig(review/'04_trajectory_rgb_triplets.jpg',dpi=150,pil_kwargs={'quality':94});plt.close(fig)

    fig,axs=plt.subplots(3,2,figsize=(17,14),constrained_layout=True)
    for ax,t in zip(axs.flat,te):
        ax.imshow(Image.open(image_path(t)));fids=ids_features(t);xy=d.feats[d.index[t]][fids]
        ax.scatter(xy[:,0],xy[:,1],s=9,facecolor='none',edgecolor='#00efff',linewidth=.6)
        vv=[x['test_feature_id'] for x in structure[t]['correspondences']];xy=d.feats[d.index[t]][vv]
        ax.scatter(xy[:,0],xy[:,1],s=16,marker='+',c='#ffe83b',linewidth=.8)
        ax.set_title(f'{t} | {len(fids)} multi-view features | {len(vv)} train-only 3D correspondences',loc='left',fontsize=12);ax.axis('off')
    fig.suptitle('Test-view evidence | cyan: >=2 train matches | yellow: train-only 3D reprojection verified',fontsize=16)
    fig.supxlabel('Sparse feature support is not a dense pixel-visibility percentage.',fontsize=12)
    fig.savefig(review/'05_test_feature_support.jpg',dpi=150,pil_kwargs={'quality':95});plt.close(fig)

    # Unmarked RGB contact sheet is useful when feature markers would hide thin boundaries.
    fig,axs=plt.subplots(3,2,figsize=(17,14),constrained_layout=True)
    for ax,t in zip(axs.flat,te):
        ax.imshow(Image.open(image_path(t)));ax.axis('off');ax.set_title(f'{t} | {d.by[t]["image_id"]} | {d.by[t]["height_band_relative_to_core"]}',loc='left',fontsize=12)
    fig.suptitle('Six proposed test RGB panoramas | all have official semantics',fontsize=17)
    fig.savefig(review/'06_test_rgb_contact.jpg',dpi=140,pil_kwargs={'quality':95});plt.close(fig)

    for c in tr+te:shutil.copy2(image_path(c).resolve(),review/'rgb'/(c+'.png'))
    files=['selected_split.json','selected_cameras.csv',f'train{len(tr)}.txt',f'test{len(te)}.txt','sfm_quality.json','verification.json',
           'official_annotation_audit.json','all_official_trajectory_brackets.json','design_comparison.json',
           'selected_test_feature_support.json','test_to_train_structure_support.json']
    for f in files:shutil.copy2(OUT/f,review/f)
    summary=[]
    for t in te:
        b=brackets[t];m=support[t];g=structure[t]
        summary.append(dict(test_id=t,image_id=d.by[t]['image_id'],height=d.by[t]['height_band_relative_to_core'],
            has_official_semantics=True,before_train=b['before'],after_train=b['after'],distance_before_m=b['distance_before_m'],
            distance_after_m=b['distance_after_m'],offset_from_train_segment_m=b['segment_offset_m'],segment_parameter=b['segment_parameter'],
            matches_before=b['matches_before'],matches_after=b['matches_after'],multi_view_features=m['features_matched_to_at_least_two_train_views'],
            validated_train_only_3d_correspondences=g['validated_train_only_landmark_correspondences'],
            median_test_reprojection_error_deg=g['median_angular_error_deg'],multi_view_feature_grid_fraction=m['two_view_occupancy']['fraction']))
    with (review/'test_support_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    write(review/'test_support_summary.json',summary)
    def camera_table(ids):
        return ''.join(f"<tr><td><a href='rgb/{c}.png'>{c}</a></td><td>{d.by[c]['image_id']}</td><td>{'高' if d.by[c]['height_band_relative_to_core']=='upper' else '低'}</td><td>{d.by[c]['z_m']:.3f}</td><td>官方 6 面</td></tr>" for c in ids)
    bracket_table=''.join(f"<tr><td>{x['test_id']}</td><td>{x['before_train']} → <b>{x['test_id']}</b> → {x['after_train']}</td><td>{x['distance_before_m']:.3f} / {x['distance_after_m']:.3f}</td><td>{100*x['offset_from_train_segment_m']:.1f}</td><td>{x['matches_before']} / {x['matches_after']}</td><td>{x['multi_view_features']}</td><td>{x['validated_train_only_3d_correspondences']}</td></tr>" for x in summary)
    labels={'18_balanced':'18 训练（9 高 / 9 低）','18_feature_gain':'18 训练（10 高 / 8 低）','19_balanced_support':'19 训练（10 高 / 9 低）','20_extra_upper':'20 训练（11 高 / 9 低，本次推荐）'}
    comp_table=''.join(f"<tr><td>{labels[x['name']]}</td><td>{x['sfm_quality']['exported_points']}</td><td>{x['sfm_quality']['angular_error_deg_quantiles_min_median_p90_p99_max'][1]:.4f}</td><td>{' / '.join(str(z['validated_train_only_landmark_correspondences']) for z in x['structure_support'])}</td></tr>" for x in comparison)
    quality=s['initial_cloud_quality'];bounds=np.array(s['xy_bounds_m']['max'])-np.array(s['xy_bounds_m']['min'])
    pairs=' / '.join(te)
    doc=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>IRIS 官方语义限定 · 20+6 候选划分</title><style>
*{{box-sizing:border-box}}body{{background:#f2f5f7;margin:0;color:#173343;font:16px/1.7 system-ui,'Microsoft YaHei',sans-serif}}main{{max-width:1550px;margin:auto;padding:32px 24px}}h1{{font-size:34px;line-height:1.3}}h2{{font-size:24px}}section{{background:white;padding:24px;margin:24px 0;border-radius:12px}}a{{color:#08768b}}img{{display:block;width:100%;height:auto}}.chips,.metrics{{display:flex;gap:12px;flex-wrap:wrap}}.chips a{{background:white;padding:5px 15px;border-radius:18px;text-decoration:none}}.metrics div{{background:white;padding:14px 22px;border-top:4px solid #147c9c;min-width:185px;border-radius:8px}}.metrics b{{display:block;font-size:27px}}.note{{padding:14px 18px;background:#fff5e7;border-left:4px solid #df7c16}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{text-align:left;padding:9px 12px;border-bottom:1px solid #dce5ea}}.scroll{{overflow:auto}}summary{{cursor:pointer;color:#08768b}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}code{{overflow-wrap:anywhere}}@media(max-width:950px){{.pair{{grid-template-columns:1fr}}main{{padding:20px 12px}}section{{padding:16px}}}}
</style></head><body><main><p style="color:#08768b;font-weight:700">IRIS · 官方语义为硬条件 · 候选划分审核 · 本次未启动训练</p>
<h1>推荐 20 张训练 + 6 张测试，全部具有官方语义标注</h1>
<p>仍以紧凑区域和实际轨迹插值为目标。重新选择测试位置，使每张测试图的同高度前、后训练视角也都有官方标注；额外两张训练图用于补充实际共同观察，不再为扩大覆盖而加入外围位置。26 张均来自 images_xphase_1，不设验证集。</p>
<div class="metrics"><div>训练 / 测试 / 验证<b>20 / 6 / 0</b></div><div>官方语义覆盖<b>26 / 26 张</b></div><div>相机范围 X × Y<b>{bounds[0]:.2f} × {bounds[1]:.2f} m</b></div><div>训练图三角化有效点<b>{quality['exported_points']:,}</b></div></div>
<p>测试有 3 个水平站位，每处高、低各一张，共 6 个三维视角；它们不是 6 个独立水平地点。该设计优先检查相同工业通道内不同高度的插值可行性。</p>
<nav class="chips"><a href="#split">完整名单</a><a href="#maps">相机与高度</a><a href="#support">匹配证据</a><a href="#images">真实图像</a><a href="#compare">18 / 19 / 20 对比</a><a href="review_bundle.zip">下载全部图表与名单</a></nav>
<section id="maps"><h2>相机位置与高度覆盖</h2><p>蓝色为训练，橙色为测试。中、右两图分开显示低层和高层，橙线连接实际拍摄顺序中同高度的前后训练视角。所有彩色相机都有官方语义。1-52/53、1-78、2-01 继续排除。</p>
<a href="01_camera_distribution.png"><img src="01_camera_distribution.png"></a>
<p>训练为低层 9 张、高层 11 张；测试为低层 3 张、高层 3 张。相机范围向小 X 方向略扩至 1-66/67，目的是给 1-68/69 提供双侧支持；没有回到此前大范围覆盖的方案。XY 凸包面积为 {s['xy_hull_area_m2']:.2f} m²。范围描述相机位置，不代表所有可见物体的边界。</p>
<a href="02_height_distribution.png"><img loading="lazy" src="02_height_distribution.png"></a><p>Z 是 RMX 中的测量世界坐标高程，不是相机离地高度。</p></section>
<section id="split"><h2>候选图像名单</h2><div class="pair"><div><h3>20 张训练</h3><div class="scroll"><table><thead><tr><th>相机</th><th>图像 ID</th><th>层</th><th>Z / m</th><th>语义</th></tr></thead><tbody>{camera_table(tr)}</tbody></table></div></div>
<div><h3>6 张测试</h3><div class="scroll"><table><thead><tr><th>相机</th><th>图像 ID</th><th>层</th><th>Z / m</th><th>语义</th></tr></thead><tbody>{camera_table(te)}</tbody></table></div>
<p>点击编号可查看 2048×1024 全景 RGB。官方资格已逐张对照原始 COCO 文件：每张对应六个立方体面，且每面都有标注对象；没有把 SAM 或此前自动分割结果作为官方依据。官方标签来自 CAD 投影，“有官方记录”不等于每个像素都有精确人工标注。</p>
<p><a href="train20.txt">train20.txt</a> · <a href="test6.txt">test6.txt</a> · <a href="selected_cameras.csv">含 XYZ / 标注来源的 CSV</a> · <a href="selected_split.json">完整划分 JSON</a> · <a href="official_annotation_audit.json">官方来源核对</a></p></div></div></section>
<section><h2>相对上一版改变了什么</h2><p>移除没有官方标注的训练图 <b>1-32、1-33、1-43、1-45、1-46、1-56</b>。上一版测试中的 <b>1-34、1-35、1-44、1-59</b> 改作训练：这些图本身有官方标注，但原先直接支撑它们的近邻中存在无官方标注视角；强行保留原测试会削弱双侧支持。</p>
<p>保留测试 <b>1-26、1-61</b>，新增 <b>1-27、1-60、1-68、1-69</b>。1-26/27、1-60/61、1-68/69 分别为三个站位的低、高层组合。中段的 1-34/35/36/37、1-44/47 用作共同观察与连接支持；下段的 1-64/65 补充不同观察方向。</p></section>
<section id="support"><h2>用实际匹配检查测试是否有训练支持</h2><p>六张测试图均落在同高度、真实拍摄顺序的前后训练位置之间。到两侧相机约 0.76–0.89 m；偏离三维连线约 4.6–6.9 cm。每侧通过固定官方位姿几何检查的匹配数为 145–469 对。判断没有依赖“落在三角形里”这一条件。</p>
<div class="scroll"><table><thead><tr><th>测试</th><th>同高度真实轨迹</th><th>两侧距离 / m</th><th>离连线 / cm</th><th>两侧匹配数</th><th>≥2 训练共同特征</th><th>训练点云验证特征</th></tr></thead><tbody>{bracket_table}</tbody></table></div>
<p>“训练点云验证特征”要求三维点只由至少 3 张训练图重建，至少 2 张训练图匹配到同一测试特征，且投影回测试视角角误差 ≤0.35°。测试图没有参与三维点坐标拟合。当前每张测试有 289–435 个多训练视角共同特征，96–165 个训练点云验证特征。</p>
<a href="03_train_test_match_matrix.png"><img loading="lazy" src="03_train_test_match_matrix.png"></a>
<p>1-69 的双侧循环一致三角化支持弱于其余五张；它还与 1-62 有 427 对几何匹配，并获得多方向训练视角补充。全体训练图支持验证后有 96 个有效三维对应，因此本版保留它检验第三站位高层；这不等于保证渲染无误。</p>
<a href="05_test_feature_support.jpg"><img loading="lazy" src="05_test_feature_support.jpg"></a>
<p class="note">这些是稀疏特征与重投影证据，不是像素重叠率。无纹理表面、近处遮挡边界和细管线仍不能仅靠稀疏匹配保证。测试图参与了划分诊断，后续应将本实验描述为面向插值可行性的选择，而非盲测随机划分。</p></section>
<section id="images"><h2>直接检查真实全景内容</h2><p>每行依次为同高度的“训练前侧 → 测试 → 训练后侧”。先看大结构是否共同可见，再结合上面的特征分布判断遮挡。</p><a href="04_trajectory_rgb_triplets.jpg"><img loading="lazy" src="04_trajectory_rgb_triplets.jpg"></a>
<details><summary>查看六张无标记测试 RGB</summary><a href="06_test_rgb_contact.jpg"><img loading="lazy" src="06_test_rgb_contact.jpg"></a></details></section>
<section id="compare"><h2>为什么建议略增至 20 张训练</h2><p>固定这 6 张测试，使用同一特征缓存、官方位姿和点云过滤条件，比较了四个全官方训练方案。下表最后一列顺序为 {pairs}；不能把不同测试图之间的计数直接解释为画质排名。</p>
<div class="scroll"><table><thead><tr><th>训练方案</th><th>有效三视角点</th><th>训练角误差中位数 / °</th><th>各测试的训练点云验证特征</th></tr></thead><tbody>{comp_table}</tbody></table></div>
<p>相对 9 高 / 9 低的 18 张方案，增加 <b>1-65</b> 加强小 X 侧高层观察，再增加 <b>1-36</b> 加强上段的共同观察。20 张方案保留两层空间覆盖，1-26/27 的三维支持从 145/139 增至 159/165，1-69 从 74 增至 96；因此使用允许的小幅数量浮动。并非为了凑够数量扩大到外围。</p>
<p>20 张训练图在三视角点轨迹图中构成一个连通分量，所有相机都有支持。保留 1,654 个初始点，角误差中位数 0.0995°、P90 0.2371°。这些是固定官方位姿的稀疏三角化检查，不是高斯训练结果。</p>
<p class="note">本轮未启动任何 OmniGS、SPAGS 或 M1–M5 训练，也未改动现有实验配置。新测试 1-27/60 曾是上一版训练图；后续若采用本划分，应从头训练，不能拿旧模型作为这组测试的独立结果。</p></section>
<section><h2>文件与复现</h2><p><a href="test_support_summary.csv">逐测试支持 CSV</a> · <a href="test_support_summary.json">逐测试支持 JSON</a> · <a href="selected_test_feature_support.json">逐对匹配</a> · <a href="test_to_train_structure_support.json">训练三维点关联</a> · <a href="design_comparison.json">数量方案对比</a> · <a href="verification.json">隔离检查</a> · <a href="sfm_quality.json">点云质量</a></p>
<p>复用前一轮 79 张候选图的 2048×1024 特征与匹配缓存。SIFT ULTRA / RootSIFT，匹配比率 0.8；固定官方位姿极线误差 ≤0.5°。训练三角化保持至少 3 视角、角误差 ≤0.35°、视差角 ≥2°、范围 0.15–200 m。仅使用所选训练图的匹配和观测，无 BA，无测试点参与训练初始化。复用统一坐标平移以便对照，世界 XYZ 仍读取原 RMX。没有修改 M1 的筛选规则。</p>
<p>候选属于 images_xphase_1 的中下部；未为了使用更多官方标签而加入更远的 xphase_2/3/4。这里给出经过实际匹配与几何检查的实用候选，不声称穷举全局最优。</p><p>服务器目录：<code>{OUT}</code></p></section></main></body></html>'''
    (review/'report.html').write_text(doc,encoding='utf-8')
    with zipfile.ZipFile(review/'review_bundle.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(review.rglob('*')):
            if p.is_file() and p.suffix!='.zip':z.write(p,p.relative_to(review))
    print('REPORT_READY',review,flush=True)


if __name__=='__main__':main()
