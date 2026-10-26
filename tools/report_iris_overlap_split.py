"""Review the compact, appearance-supported IRIS 18/6 selection with static plots."""
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
from reselect_iris_interpolation import Diagnostics,OUT,RUN,read,write,xyz


def main():
    d=Diagnostics();s=read(OUT/'selected_split.json');tr=s['train_ids'];te=s['test_ids']
    support={x['test']:x for x in read(OUT/'selected_test_feature_support.json')}
    structure={x['test']:x for x in read(OUT/'test_to_train_structure_support.json')}
    old_support={x['test']:x for x in read(OUT/'old_split_test_support.json')}
    old_structure={x['test']:x for x in read(OUT/'old_common_test_structure_support.json')}
    brackets={x['test']:x for x in s['test_brackets']}
    review=OUT/'review';review.mkdir(exist_ok=True);(review/'rgb').mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    old=read(RUN/'approved_split.json')['selected_cameras']
    train_color,test_color='#147c9c','#e88925'
    # Overview plus separate height layers prevents coincident high/low labels hiding each other.
    fig,axs=plt.subplots(1,3,figsize=(19,9),gridspec_kw={'width_ratios':[.92,1,1]},constrained_layout=True)
    ax=axs[0];all_main=[x for x in d.rows if x['group']==1]
    ax.scatter([x['x_m'] for x in all_main],[x['y_m'] for x in all_main],s=15,c='#d7dce0',label='Main-region cameras')
    op=np.array([xyz(x) for x in old if x['candidate_role']=='train']);oh=ConvexHull(op[:,:2]);poly=op[oh.vertices,:2];poly=np.vstack([poly,poly[0]])
    ax.plot(poly[:,0],poly[:,1],'--',color='#aaadb2',label='Previous train hull: 30.59 m2')
    npnt=np.array([xyz(d.by[x]) for x in tr]);nh=ConvexHull(npnt[:,:2]);poly=npnt[nh.vertices,:2];poly=np.vstack([poly,poly[0]])
    ax.fill(poly[:,0],poly[:,1],color=train_color,alpha=.12)
    ax.plot(poly[:,0],poly[:,1],color=train_color,label='New train hull: 6.60 m2')
    for ids,color,marker,label in [(tr,train_color,'o','New training (18)'),(te,test_color,'D','New test (6)')]:
        ax.scatter([d.by[x]['x_m'] for x in ids],[d.by[x]['y_m'] for x in ids],s=33,c=color,marker=marker,label=label,zorder=5)
    for ids,label,offset in [(['1-78','2-01'],'1-78 / 2-01',(-32,-22)),(['1-52','1-53'],'1-52 / 1-53',(-50,-24)),(['1-01','1-02'],'1-01 / 1-02',(8,8))]:
        pt=np.mean([xyz(d.by[x]) for x in ids],axis=0)
        ax.scatter(pt[0],pt[1],s=90,marker='x',c='#aa6d6d',zorder=6)
        ax.annotate(label,pt[:2],xytext=offset,textcoords='offset points',fontsize=9,color='#8c5151')
    ax.set(title='Scope contracted around supported test views',xlabel='World X (m)',ylabel='World Y (m)',xlim=(24.8,31.8),ylim=(8.7,18.5))
    ax.set_aspect('equal',adjustable='box');ax.legend(loc='upper left',fontsize=8)
    offsets={'1-24':(-45,9),'1-25':(-46,9),'1-26':(-7,14),'1-27':(-12,13),'1-29':(4,-18),
             '1-32':(4,10),'1-33':(4,10),'1-34':(-7,14),'1-35':(-7,14),
             '1-36':(-46,6),'1-37':(-47,7),'1-43':(-45,4),'1-44':(-7,14),
             '1-45':(-43,-20),'1-46':(8,-16),'1-47':(7,-18),'1-56':(5,-18),'1-58':(4,-20),
             '1-59':(-3,13),'1-60':(-40,-18),'1-61':(-10,15),'1-62':(-40,10),'1-70':(5,-20),'1-71':(5,-20)}
    for ax,band,label in zip(axs[1:],['lower','upper'],['Lower layer: 9 train + 3 test','Upper layer: 9 train + 3 test']):
        for t,b in brackets.items():
            if d.by[t]['height_band_relative_to_core']!=band:continue
            p=np.array([xyz(d.by[c]) for c in [b['before'],t,b['after']]])
            ax.plot(p[:,0],p[:,1],color=test_color,lw=2,alpha=.65,zorder=2)
        for cid in tr+te:
            row=d.by[cid]
            if row['height_band_relative_to_core']!=band:continue
            test=cid in te;color=test_color if test else train_color
            ax.scatter(row['x_m'],row['y_m'],s=92 if test else 68,marker='D' if test else 'o',
                       facecolor=color if row['has_official_semantics'] else 'white',edgecolor=color,linewidth=1.7,zorder=4)
            text=cid+(' T' if test else '')+(' *' if not row['has_official_semantics'] else '')
            ax.annotate(text,xyz(row)[:2],xytext=offsets[cid],textcoords='offset points',fontsize=10,
                        color='#a34f00' if test else '#125a72',fontweight='bold' if test else 'normal')
        z=[d.by[x]['z_m'] for x in tr+te if d.by[x]['height_band_relative_to_core']==band]
        ax.set(title=label+f'\nZ = {min(z):.3f} to {max(z):.3f} m',xlabel='World X (m)',ylabel='World Y (m)',
               xlim=(26.2,29.45),ylim=(11.05,15.25))
        ax.set_aspect('equal',adjustable='box')
    for ax in axs:ax.grid(alpha=.18);ax.ticklabel_format(useOffset=False,style='plain')
    fig.suptitle('IRIS | 18 train + 6 test | Actual trajectory interpolation with verified image overlap',fontsize=17)
    fig.supxlabel('Orange segments join each test to its preceding/following same-height training cameras.  * / hollow circle: no official semantics.',fontsize=11)
    fig.savefig(review/'01_compact_camera_distribution.png',dpi=180);plt.close(fig)

    # Exact per-pair correspondences after the unchanged spherical geometry checks.
    mat=np.array([[len(d.match(t,c)) for c in tr] for t in te])
    fig,ax=plt.subplots(figsize=(18,5.6),constrained_layout=True)
    im=ax.imshow(mat,cmap='YlGnBu',norm=PowerNorm(.6,vmin=0,vmax=500),aspect='auto')
    for i,t in enumerate(te):
        for j,c in enumerate(tr):
            value=mat[i,j];ax.text(j,i,str(value) if value else '—',ha='center',va='center',fontsize=9,color='white' if value>180 else '#16303c')
            if c in [brackets[t]['before'],brackets[t]['after']]:ax.add_patch(Rectangle((j-.48,i-.48),.96,.96,fill=False,edgecolor='#f19b32',lw=2.5))
    ax.set_xticks(range(18),[c+(' *' if not d.by[c]['has_official_semantics'] else '') for c in tr],rotation=45,ha='right')
    ax.set_yticks(range(6),[t+(' L' if d.by[t]['height_band_relative_to_core']=='lower' else ' H') for t in te])
    ax.set(xlabel='Selected training cameras (* = no official semantics)',ylabel='Test cameras',title='Real test-to-train feature matches | orange frames = same-height trajectory neighbors')
    fig.colorbar(im,ax=ax,label='Geometrically verified pair matches',shrink=.8)
    fig.savefig(review/'02_train_test_match_matrix.png',dpi=180);plt.close(fig)

    fig,axs=plt.subplots(6,3,figsize=(18,19.5),constrained_layout=True)
    for i,t in enumerate(te):
        b=brackets[t]
        for j,cid in enumerate([b['before'],t,b['after']]):
            ax=axs[i,j];img=Image.open(OUT/'pool_images'/(d.by[cid]['image_id']+'.png')).convert('RGB')
            ax.imshow(img);ax.set_axis_off()
            role='TRAIN before' if j==0 else 'TEST between' if j==1 else 'TRAIN after'
            count='' if j==1 else f" | {b['matches_before' if j==0 else 'matches_after']} matches to test"
            ax.set_title(f'{cid} | {role}{count}',loc='left',fontsize=11,color='#a34f00' if j==1 else '#125a72')
    fig.suptitle('Six actual interpolation triplets | images, not camera hulls, provide the overlap evidence',fontsize=17)
    fig.savefig(review/'03_trajectory_rgb_triplets.jpg',dpi=150,pil_kwargs={'quality':94});plt.close(fig)

    def feature_ids(t,train_ids):
        count={}
        for c in train_ids:
            for fid in set(d.match(t,c)[:,0].tolist()):count[fid]=count.get(fid,0)+1
        return sorted(k for k,v in count.items() if v>=2)
    old_train=[x['camera_id'] for x in old if x['candidate_role']=='train']
    t='1-44';fig,axs=plt.subplots(2,1,figsize=(16,17),constrained_layout=True)
    for ax,ids,geom,label in [(axs[0],old_train,old_structure[t],'Previous broad split'),(axs[1],tr,structure[t],'New compact split')]:
        ax.imshow(Image.open(OUT/'pool_images'/(d.by[t]['image_id']+'.png')))
        fids=feature_ids(t,ids);xy=d.feats[d.index[t]][fids]
        ax.scatter(xy[:,0],xy[:,1],s=16,facecolor='none',edgecolor='#00efff',linewidth=.8,label='Matched to at least 2 train cameras')
        valid=[x['test_feature_id'] for x in geom['correspondences']];xy=d.feats[d.index[t]][valid]
        ax.scatter(xy[:,0],xy[:,1],s=22,marker='+',color='#ffe83b',linewidth=1,label='Verified against training-only 3D tracks')
        ax.set_title(f"Same test 1-44 | {label}: {len(fids)} multi-view features / {len(valid)} verified 3D correspondences",loc='left',fontsize=15)
        ax.legend(loc='lower left',fontsize=10,framealpha=.9);ax.set_axis_off()
    fig.suptitle('Measured visibility support at the previously weak test view',fontsize=19)
    fig.savefig(review/'04_test44_overlap_evidence.jpg',dpi=150,pil_kwargs={'quality':95});plt.close(fig)

    fig,axs=plt.subplots(3,2,figsize=(16,14.4),constrained_layout=True)
    for ax,t in zip(axs.flat,te):
        ax.imshow(Image.open(OUT/'pool_images'/(d.by[t]['image_id']+'.png')))
        fids=feature_ids(t,tr);xy=d.feats[d.index[t]][fids]
        ax.scatter(xy[:,0],xy[:,1],s=9,facecolor='none',edgecolor='#00efff',linewidth=.6)
        valid=[x['test_feature_id'] for x in structure[t]['correspondences']];xy=d.feats[d.index[t]][valid]
        ax.scatter(xy[:,0],xy[:,1],s=16,marker='+',c='#ffe83b',linewidth=.8)
        ax.set_title(f"{t} | {len(fids)} multi-view features | {len(valid)} train-only 3D correspondences",loc='left',fontsize=11)
        ax.set_axis_off()
    fig.suptitle('All 6 tests | cyan: matches to >=2 train cameras | yellow: verified training-only 3D points',fontsize=15)
    fig.supxlabel('These are sparse feature-support observations, not a dense pixel-visibility guarantee.',fontsize=11)
    fig.savefig(review/'05_all_test_feature_support.jpg',dpi=150,pil_kwargs={'quality':94});plt.close(fig)

    # Readable height strip, with layers separated vertically using the real measured Z values.
    fig,ax=plt.subplots(figsize=(13,5),constrained_layout=True)
    for role,ids,color,marker in [('Training',tr,train_color,'o'),('Test',te,test_color,'D')]:
        ax.scatter([d.by[x]['y_m'] for x in ids],[d.by[x]['z_m'] for x in ids],s=60,c=color,marker=marker,label=role)
    for j,t in enumerate(te):
        b=brackets[t];p=np.array([xyz(d.by[x]) for x in [b['before'],t,b['after']]])
        ax.plot(p[:,1],p[:,2],color=test_color,alpha=.5)
        ax.annotate(t,(d.by[t]['y_m'],d.by[t]['z_m']),xytext=(6,12 if t in ['1-34','1-44','1-61'] else -18),textcoords='offset points',color='#a34f00',fontweight='bold')
    ax.set(xlabel='World Y (m)',ylabel='World Z (m)',title='Measured elevation | 9 + 9 train / 3 + 3 test | Z is not height above floor',ylim=(151.5,152.75))
    ax.grid(alpha=.2);ax.legend(loc='center right');ax.ticklabel_format(useOffset=False,style='plain')
    fig.savefig(review/'06_actual_height_distribution.png',dpi=170);plt.close(fig)

    for cid in tr+te:shutil.copy2((OUT/'pool_images'/(d.by[cid]['image_id']+'.png')).resolve(),review/'rgb'/(cid+'.png'))
    files=['selected_split.json','selected_cameras.csv','train18.txt','test6.txt','sfm_quality.json','verification.json',
           'selected_test_feature_support.json','test_to_train_structure_support.json','old_split_test_support.json','old_common_test_structure_support.json']
    for f in files:shutil.copy2(OUT/f,review/f)
    summary=[]
    for t in te:
        b=brackets[t];m=support[t];g=structure[t]
        summary.append(dict(test_id=t,image_id=d.by[t]['image_id'],height=d.by[t]['height_band_relative_to_core'],
            before_train=b['before'],after_train=b['after'],distance_before_m=b['distance_before_m'],distance_after_m=b['distance_after_m'],
            offset_from_train_segment_m=b['segment_offset_m'],segment_parameter=b['segment_parameter'],
            matches_before=b['matches_before'],matches_after=b['matches_after'],multi_view_features=m['features_matched_to_at_least_two_train_views'],
            validated_train_only_3d_correspondences=g['validated_train_only_landmark_correspondences'],
            multi_view_feature_grid_fraction=m['two_view_occupancy']['fraction']))
    with (review/'test_support_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    write(review/'test_support_summary.json',summary)
    def camera_table(ids):
        return ''.join(f"<tr><td><a href='rgb/{c}.png'>{c}</a></td><td>{d.by[c]['image_id']}</td><td>{'高' if d.by[c]['height_band_relative_to_core']=='upper' else '低'}</td><td>{'有' if d.by[c]['has_official_semantics'] else '待补标'}</td></tr>" for c in ids)
    bracket_table=''.join(f"<tr><td>{x['test_id']}</td><td>{x['before_train']} → <b>{x['test_id']}</b> → {x['after_train']}</td><td>{x['distance_before_m']:.3f} / {x['distance_after_m']:.3f}</td><td>{100*x['offset_from_train_segment_m']:.1f}</td><td>{x['matches_before']} / {x['matches_after']}</td><td>{x['multi_view_features']}</td><td>{x['validated_train_only_3d_correspondences']}</td></tr>" for x in summary)
    common_table=''.join(f"<tr><td>{t}</td><td>{old_support[t]['features_matched_to_at_least_two_train_views']} → <b>{support[t]['features_matched_to_at_least_two_train_views']}</b></td><td>{old_structure[t]['validated_train_only_landmark_correspondences']} → <b>{structure[t]['validated_train_only_landmark_correspondences']}</b></td></tr>" for t in ['1-26','1-44','1-61'])
    missing='、'.join(s['missing_official_train_ids'])
    doc=f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>IRIS 轨迹插值与真实可见重叠划分</title><style>
*{{box-sizing:border-box}}body{{background:#f2f5f7;margin:0;color:#173343;font:16px/1.7 system-ui,'Microsoft YaHei',sans-serif}}main{{max-width:1500px;margin:auto;padding:32px 24px}}h1{{font-size:34px;line-height:1.3}}h2{{font-size:24px}}section{{background:white;padding:24px;margin:24px 0;border-radius:12px}}a{{color:#08768b}}img{{display:block;width:100%;height:auto}}.chips,.metrics{{display:flex;gap:12px;flex-wrap:wrap}}.chips a{{background:white;padding:5px 15px;border-radius:18px;text-decoration:none}}.metrics div{{background:white;padding:14px 22px;border-top:4px solid #147c9c;min-width:185px;border-radius:8px}}.metrics b{{display:block;font-size:27px}}.note{{padding:14px 18px;background:#fff5e7;border-left:4px solid #e88925}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{text-align:left;padding:9px 12px;border-bottom:1px solid #dce5ea}}.scroll{{overflow:auto}}summary{{cursor:pointer;color:#08768b}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}code{{overflow-wrap:anywhere}}@media(max-width:850px){{.pair{{grid-template-columns:1fr}}main{{padding:20px 12px}}section{{padding:16px}}}}
</style></head><body><main><p style="color:#08768b;font-weight:700">IRIS · 重新选定 18 张训练 + 6 张测试 · 无验证集 · 原训练保持暂停</p>
<h1>围绕测试视角的实际可见内容，收缩到一段连续工业通道</h1><p>本版以“同高度真实轨迹前后邻接 + 实际图像匹配 + 训练点云重投影”作为主要依据。全部 24 张均来自 images_xphase_1，取消为覆盖外围而保留的 1-78、2-01、1-52、1-53 等位置。</p>
<div class="metrics"><div>训练 / 测试 / 验证<b>18 / 6 / 0</b></div><div>训练范围 X × Y<b>约 2.45 × 3.55 m</b></div><div>同高轨迹邻居距离<b>0.64–0.84 m</b></div><div>训练点云初始化<b>269 → 1,347 点</b></div></div>
<p>这些范围指相机位置，不是全景中可见物体的空间边界。初始点是相同过滤规则下的稀疏 SfM 点，不是训练后的高斯数量。</p>
<nav class="chips"><a href="#split">图像名单</a><a href="#maps">相机分布</a><a href="#support">测试支持证据</a><a href="#images">真实轨迹图像</a><a href="#changes">新旧对比</a><a href="review_bundle.zip">下载图表与名单</a></nav>
<section id="maps"><h2>相机分布：先收缩区域，再验证每张测试图的支持</h2><p>训练 XY 凸包从 30.59 缩至 6.60 m²。高低层分别画出，避免平面投影把两层相机叠成一个点。橙色线段是测试图实际拍摄顺序中的同高度前、后训练邻居。训练保持高低各 9 张，测试高低各 3 张。</p><a href="01_compact_camera_distribution.png"><img src="01_compact_camera_distribution.png"></a><details><summary>查看真实 Z 高度分布</summary><img src="06_actual_height_distribution.png"></details></section>
<section id="split"><h2>本次选定的图像名单</h2><p>测试图：<b>{'、'.join(te)}</b>。其中 1-34/35 是同一水平站位的不同高度，因此 6 个三维测试视角覆盖 5 个水平站位。</p>
<div class="pair"><div><h3>18 张训练图（点击编号查看完整全景）</h3><div class="scroll"><table><thead><tr><th>相机</th><th>图像 ID</th><th>高度层</th><th>官方语义</th></tr></thead><tbody>{camera_table(tr)}</tbody></table></div></div><div><h3>6 张测试图</h3><div class="scroll"><table><thead><tr><th>相机</th><th>图像 ID</th><th>高度层</th><th>官方语义</th></tr></thead><tbody>{camera_table(te)}</tbody></table></div>
<p>训练官方标注：<b>12/18</b>；测试官方标注：<b>6/6</b>。待补标训练图为 <b>{missing}</b>。优先保留这些视角提供的真实重叠，后续可人工补标；官方状态由 IRIS 官方记录确定，没有把 SAM 分割当作官方标签。</p>
<p><a href="train18.txt">train18.txt</a> · <a href="test6.txt">test6.txt</a> · <a href="selected_cameras.csv">相机信息 CSV</a> · <a href="selected_split.json">完整划分 JSON</a></p></div></div></section>
<section id="support"><h2>每张测试图都有两侧真实训练视角，而不只是落在三角形里</h2><p>测试点在线段上的位置为 43.3%–52.7%，接近两侧训练相机的中间；偏离三维连线仅 4.6–8.2 cm。每侧有 205–469 对通过几何验证的特征匹配。额外训练相机用于增加跨高度和不同遮挡方向的共同观察。</p>
<div class="scroll"><table><thead><tr><th>测试</th><th>真实同高轨迹</th><th>两侧距离 m</th><th>离连线 cm</th><th>两侧几何匹配</th><th>≥2 训练共同特征</th><th>训练点云验证特征</th></tr></thead><tbody>{bracket_table}</tbody></table></div>
<p>“≥2 训练共同特征”对同一测试特征去重；“训练点云验证特征”进一步要求：三维点仅由至少 3 张训练图构建、至少 2 张训练图与该测试特征关联、投影回测试视角角误差 ≤0.35°。测试图没有参与这些三维点的位置拟合。</p>
<a href="02_train_test_match_matrix.png"><img loading="lazy" src="02_train_test_match_matrix.png"></a>
<p class="note">这些计数与网格分布衡量稀疏可匹配特征支持，不等同于整张图每个像素都已被观察。大块无纹理表面、近处遮挡边界和细管线仍需后续实际渲染检验。</p>
<details><summary>查看 6 张测试图上的支持特征分布</summary><img loading="lazy" src="05_all_test_feature_support.jpg"></details></section>
<section id="images"><h2>直接看“训练前侧 → 测试 → 训练后侧”的真实全景</h2><p>每行的三张图来自相同高度层和相邻的实际拍摄顺序。标题给出两侧与测试图的几何匹配数；不是由相机坐标推测出的可见性。</p><a href="03_trajectory_rgb_triplets.jpg"><img loading="lazy" src="03_trajectory_rgb_triplets.jpg"></a></section>
<section id="changes"><h2>对保留的测试图，支持是否实际增加</h2><p>仅对新旧划分共有的测试图比较，避免把换成不同图像后的数值直接当作提升。三张共有测试图是 1-26、1-44、1-61。</p><table><thead><tr><th>同一测试图</th><th>至少两张训练共同特征：旧 → 新</th><th>训练点云验证特征：旧 → 新</th></tr></thead><tbody>{common_table}</tbody></table>
<p>1-78/2-01 对旧测试 1-13、1-18、1-26、1-44 均无满足当前标准的连接；对 1-61 分别有 9/10 对，对 1-71 有 34/46 对。它们并非绝对没有贡献，但当前 18 张预算更适合放到测试附近的连续通道。</p>
<a href="04_test44_overlap_evidence.jpg"><img loading="lazy" src="04_test44_overlap_evidence.jpg"></a>
<p>原测试 1-13/18 位于本次收缩范围以北，不再保留。原测试 1-71 转为训练视角，为下段测试提供另一个观察方向。1-59 从原训练改为测试，新增 1-34/35 用于检查中段同一站位的不同高度。</p></section>
<section><h2>训练初始化、标注取舍与复现</h2><p>本次只用选定 18 张训练图的匹配进行固定官方位姿三角化，保留 1,347 个至少三视角支持的点，18/18 训练相机都有轨迹支持。角重投影误差中位数为 0.1025°（旧方案为 0.0992°），初始点增加并非通过放宽阈值获得。测试图只参与划分与支持诊断。</p>
<p>首先固定 6 组同高度轨迹前后邻居，共 12 张训练图；其余 6 张根据实测共同特征和图像网格分布补充，并维持 9/9 高低层。也检查了官方标注更多的替代：以 1-63 换 1-46 会使弱测试 1-44 的多视角共同特征由 358 降至 325，本版优先保留 1-46 的补充观察。当前方案是经过图像与几何检查的实用选择，不声称全局最优。</p>
<p class="note">本轮没有启动任何新训练。旧 OmniGS 实验仍暂停在约 4,900 步；由于 1-59 已改为测试图，后续应从头训练这版新划分，不能把旧模型继续训练后当作新的独立测试结果。</p>
<p><a href="test_support_summary.csv">逐测试视角支持 CSV</a> · <a href="test_support_summary.json">支持摘要 JSON</a> · <a href="selected_test_feature_support.json">所有逐对匹配数据</a> · <a href="test_to_train_structure_support.json">三维点关联证据</a> · <a href="sfm_quality.json">训练点云质量</a> · <a href="verification.json">隔离检查</a></p>
<details><summary>方法细节</summary><p>检查 79 张真实图像（主区 76 张及此前保留的 3 张第二组参考），并为新加入候选提取 SIFT ULTRA / RootSIFT。匹配比率 0.8；固定官方位姿球面极线过滤阈值 0.5°，每对至少 8 个有效匹配。初始化与之前一致：至少 3 训练视角、角误差 ≤0.35°、三角化角 ≥2°、范围 0.15–200 m，不优化相机位姿。比较了中下部紧凑区域、较长中央区域及北部区域；本版选取空间最紧凑且各测试均有双侧真实支持的中下部方案。</p><p>网格统计为测试 ERP 有效特征区域的 12×6 网格；至少有 5 个原始特征的格子才进入分母，至少有 3 个支持特征才算支持格。该比例是特征分布指标，不是像素重叠率。本次经用户授权使用测试图设计划分，因此记录为基于几何重叠选择的实验划分。</p></details>
<p>服务器目录：<code>{OUT}</code></p></section></main></body></html>"""
    (review/'report.html').write_text(doc,encoding='utf-8')
    with zipfile.ZipFile(review/'review_bundle.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for path in sorted(review.rglob('*')):
            if path.is_file() and path.suffix!='.zip':z.write(path,path.relative_to(review))
    print('REPORT_READY',review,flush=True)


if __name__=='__main__':main()
