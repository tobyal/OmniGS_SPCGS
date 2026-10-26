"""Complete current26 compact labels by merging residual holes into adjacent classes."""
from pathlib import Path
import collections
import copy
import csv
import json
import shutil
import sys
import zipfile

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, find_objects, label

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spcgs.common import ROOT, read, write, ref, sha
from consolidate_iris26_semantics import save, font, EN

RUN = ROOT / 'runs/iris26_semantic_consolidation_20261025'
OUT = RUN / 'review'
DATA = ROOT.parent / 'datasets/iris_official20_6_no_tripod_2k_20261025'
BASE = DATA / 'semantics_iris26_compact7_v3'
EXPORT = DATA / 'semantics_iris26_compact7_v4'
CONFIG = ROOT / 'configs/iris26_compact7_classes.json'


def merge_ignored(before):
    """Assign each 8-connected hole to its modal 4-neighbor boundary class.

    All decisions use immutable v3 labels, so iteration order cannot affect them.
    This current26 revision has no ignored pixels at the ERP left/right seam.
    Equal boundary votes resolve to the smallest class ID for reproducibility.
    """
    missing = before == 255
    assert not missing[:, [0, -1]].any(), 'Current26 audit expected no seam holes'
    cc, count = label(missing, np.ones((3, 3)))
    after = before.copy()
    edits = []
    for component, sl in enumerate(find_objects(cc), 1):
        if sl is None:
            continue
        ys, xs = sl
        y0, y1 = max(0, ys.start-1), min(before.shape[0], ys.stop+1)
        x0, x1 = max(0, xs.start-1), min(before.shape[1], xs.stop+1)
        region = cc[y0:y1, x0:x1] == component
        old = before[y0:y1, x0:x1]
        border = binary_dilation(region) & ~region & (old != 255)
        votes = np.bincount(old[border], minlength=7)
        assert votes[:7].sum() > 0
        target = int(votes.argmax())
        after[y0:y1, x0:x1][region] = target
        edits.append({'component': component, 'pixels': int(region.sum()),
                      'bbox_xyxy': [xs.start, ys.start, xs.stop, ys.stop],
                      'target_id': target, 'adjacent_class_counts': votes.tolist()})
    assert len(edits) == count
    assert np.array_equal(before[~missing], after[~missing])
    assert set(np.unique(after)) <= set(range(7))
    return after, cc, edits


def detail_image(cid, rgb, before, after, components, edits, palette, cfg):
    chosen = sorted(edits, key=lambda x: -x['pixels'])[:3]
    sheet = Image.new('RGB', (1200, max(1, len(chosen))*330), '#f2f5f7')
    draw = ImageDraw.Draw(sheet)
    old_palette = palette.copy(); old_palette[255] = [255, 0, 160]
    for row, entry in enumerate(chosen):
        x0,y0,x1,y1 = entry['bbox_xyxy']
        box = (max(0,x0-26),max(0,y0-26),min(2048,x1+26),min(1024,y1+26))
        draw.text((8,row*330+2), f'{cid} · {entry["pixels"]}像素 → {cfg["classes"][entry["target_id"]]["zh"]}', font=font(19), fill='#193846')
        for column, (arr, title) in enumerate([(rgb,'RGB'),(old_palette[before],'修改前：粉色标出待合并区域'),(palette[after],'修改后')]):
            draw.text((column*400+8,row*330+30),title,font=font(16),fill='#526b7b')
            im = Image.fromarray(arr).crop(box)
            scale = min(390/im.width,265/im.height)
            im = im.resize((round(im.width*scale),round(im.height*scale)), Image.Resampling.LANCZOS if column==0 else Image.Resampling.NEAREST)
            sheet.paste(im,(column*400+(400-im.width)//2,row*330+62))
    save(OUT/'ignore_details'/(cid+'.jpg'),sheet)


def main():
    # Immutable previous exports make repeated runs reproducible.
    previous = read(BASE/'manifest.json')
    audit = read(previous['source_audit']['path'])
    assert sha(audit['source_coco']['path']) == audit['source_coco']['sha256']
    if not (RUN/'review_v3').exists():
        shutil.copytree(OUT,RUN/'review_v3')
    cfg = read(BASE/'classes.json')
    cfg['version'] = 'iris26_compact7_v4'
    cfg.pop('ignore_id',None)
    cfg['has_ignore_class'] = False
    cfg['label_ids'] = list(range(7))
    cfg['semantic_source_codes'].pop('0',None)
    cfg['semantic_source_codes']['4'] = 'former ignored component merged into its adjacent majority class; derived completion, not official truth'
    cfg['residual_official_ids'] = {'9':'Reviewed objects keep their assigned class; remaining holes merge into adjacent classes.', '13':'No separate fire class; residual regions merge into adjacent classes.'}
    cfg.pop('unmapped_official_ids',None)
    cfg['initial_object_corrections'] = cfg.pop('object_corrections')
    cfg['residual_completion'] = {'method':'Per 8-connected v3 ignore component, select most frequent class among 4-neighbor boundary pixels; deterministic lowest-ID tie break. Decisions use unchanged v3 labels.', 'changes_only_previous_id':255, 'creates_new_class':False}
    cfg['artifact_policy'] = 'All pixels have one of seven semantic classes. Reconstructed bottom floor remains building. Original RGB observation masks are preserved independently.'
    cfg['user_requested_revision'] = 'v4: merge all residual ignored regions into neighboring classes; retain v3 curbs and bottom floor as building.'
    write(CONFIG,cfg); write(EXPORT/'classes.json',cfg); write(OUT/'classes.json',cfg)
    shutil.copy2(ROOT/'configs'/cfg['initial_object_corrections'], EXPORT/cfg['initial_object_corrections'])
    palette = np.zeros((256,3),'u1')
    palette[255] = [30,36,45]
    for c in cfg['classes']:
        palette[c['id']] = c['color']
    weights = np.broadcast_to(np.cos((.5-(np.arange(1024)+.5)/1024)*np.pi)[:,None],(1024,2048))
    records=[]; components_log=[]; class_rows=[]; destination=collections.Counter()
    for original in previous['records']:
        cid=original['camera_id']; name=original['image_id']+'.png'
        for role in ['labels','rgb','source_labels','rgb_observation_mask']:
            assert sha(original[role]['path'])==original[role]['sha256'],(cid,role)
        before=np.array(Image.open(BASE/'labels'/name))
        after, components, edits=merge_ignored(before)
        missing=before==255
        rgb=np.array(Image.open(original['rgb']['path']).convert('RGB'))
        raw=np.array(Image.open(original['source_labels']['path']))
        observed=np.array(Image.open(original['rgb_observation_mask']['path']))>0
        provenance=np.array(Image.open(BASE/'provenance'/name));provenance[missing]=4
        assert np.all(after[~observed]==2)
        for entry in edits:
            entry['camera_id']=cid
            destination[entry['target_id']]+=entry['pixels']
        components_log.extend(edits)
        for folder,arr in [('labels',after),('valid_masks',np.full(after.shape,255,'u1')),('observed_valid_masks',observed.astype('u1')*255),('provenance',provenance)]:
            save(EXPORT/folder/name,arr)
        save(OUT/'before_semantics'/(cid+'.png'),palette[before])
        save(OUT/'final_semantics'/(cid+'.png'),palette[after])
        save(OUT/'final_overlays'/(cid+'.jpg'),(.55*rgb+.45*palette[after]).astype('u1'))
        save(OUT/'ignore_changes'/(cid+'.png'),missing.astype('u1')*255)
        highlight=Image.fromarray(rgb);draw=ImageDraw.Draw(highlight)
        for entry in edits:
            x0,y0,x1,y1=entry['bbox_xyxy'];draw.rectangle((x0-3,y0-3,x1+3,y1+3),outline=(255,0,160),width=2)
        save(OUT/'ignore_highlights'/(cid+'.jpg'),highlight)
        detail_image(cid,rgb,before,after,components,edits,palette,cfg)
        row=copy.deepcopy(original)
        row['previous_labels']=ref(BASE/'labels'/name)
        row['previous_ignore_reason_counts']=row.pop('reason_for_ignore_counts')
        for key,folder in [('labels','labels'),('valid_mask','valid_masks'),('observed_valid_mask','observed_valid_masks'),('provenance','provenance')]:row[key]=ref(EXPORT/folder/name)
        row.update(ignored_erp_percent=0.,ignored_sphere_percent=0.,filled_pixels=int(missing.sum()),filled_components=len(edits))
        row['filled_destinations']={str(k):int(((after==k)&missing).sum()) for k in range(7)}
        row['misc_destinations']={str(k):{'pixels':int(((raw==9)&observed&(after==k)).sum())} for k in range(7)}
        records.append(row)
        for k in range(7):
            mask=after==k
            class_rows.append({'camera_id':cid,'split':row['split'],'class_id':k,'pixels':int(mask.sum()),'erp_percent':float(mask.mean()*100),'sphere_percent':float(weights[mask].sum()/weights.sum()*100)})
        print('FILLED',cid,int(missing.sum()),'pixels',flush=True)
    summary=[]
    for c in cfg['classes']:
        stats=[r for r in class_rows if r['class_id']==c['id']]
        summary.append({'id':c['id'],'name':c['name'],'zh':c['zh'],'pixels':sum(r['pixels'] for r in stats),'erp_percent':sum(r['erp_percent'] for r in stats)/26,'sphere_percent':sum(r['sphere_percent'] for r in stats)/26,'views':sum(r['pixels']>0 for r in stats),'train_views':sum(r['pixels']>0 and r['split']=='train' for r in stats),'test_views':sum(r['pixels']>0 and r['split']=='test' for r in stats)})
    totals={'version':cfg['version'],'classes_after':summary,'per_image':class_rows,'filled_pixels':sum(r['filled_pixels'] for r in records),'filled_components':len(components_log),'filled_destinations':dict(destination),'remaining_ignore_pixels':0,'mean_ignore_sphere_percent':0.,'mean_ignore_erp_percent':0.}
    write(RUN/'statistics.json',totals);write(OUT/'statistics.json',totals)
    write(EXPORT/'completion_components.json',components_log);write(OUT/'completion_components.json',components_log)
    manifest={k:v for k,v in previous.items() if k not in ['records','ignore_id','taxonomy','object_corrections','previous_export']}
    manifest.update(schema='iris26_compact7_label_export_v4',version=cfg['version'],has_ignore_class=False,label_ids=list(range(7)),records=records,taxonomy=ref(EXPORT/'classes.json'),previous_export=ref(BASE/'manifest.json'),completion=ref(EXPORT/'completion_components.json'),label_kind='Derived seven-class labels with reviewed object corrections, reconstructed floor and residual adjacent-class completion',mask_integration='Semantic valid_masks are all255 (mask values, not class IDs). Labels contain only0..6. observed_valid_masks preserve RGB observation provenance independently.')
    write(EXPORT/'manifest.json',manifest);write(OUT/'manifest.json',manifest)
    fields=['id','zh','pixels','erp_percent','sphere_percent','views','train_views','test_views']
    with (OUT/'class_statistics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({k:s[k] for k in fields} for s in summary)
    legend=Image.new('RGB',(1280,270),'white');draw=ImageDraw.Draw(legend)
    for i,c in enumerate(cfg['classes']):
        x=20+(i%4)*315;y=15+(i//4)*125;draw.rectangle((x,y,x+38,y+38),fill=tuple(c['color']));draw.text((x+48,y+2),f'{i} {c["zh"]}',font=font(20),fill='#183848');draw.text((x,y+52),f'{summary[i]["sphere_percent"]:.3f}%',font=font(18),fill='#587180')
    save(OUT/'legend.png',legend)
    qa={'checks_passed':True,'images':26,'train':20,'test':6,'val':0,'remaining_ignore_pixels':0,'label_ids':list(range(7)),'changed_pixels':totals['filled_pixels'],'only_previous_255_changed':True,'RGB_and_raw_labels_hashes_unchanged':True,'bottom_floor_stays_building':True,'original_rgb_observation_masks_preserved':True,'semantic_valid_masks_all_255':True,'original_COCO_preserved':True,'training_started':False,'final_visual_review_pending':True}
    write(OUT/'qa.json',qa);write(EXPORT/'qa.json',qa)
    render_report(cfg,totals,records)
    package()
    dm=read(DATA/'dataset_manifest.json');dm['semantic_annotation']={'version':cfg['version'],'manifest':ref(EXPORT/'manifest.json'),'classes':ref(EXPORT/'classes.json'),'status':'derived_labels_ready_for_review','training_started':False};write(DATA/'dataset_manifest.json',dm)
    print(json.dumps({'pixels':totals['filled_pixels'],'components':len(components_log),'destination':dict(destination),'ignore_remaining':0,'export':str(EXPORT)}))


def package():
    with zipfile.ZipFile(OUT/'iris26_compact7_v4_labels.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in EXPORT.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(EXPORT))
        z.write(OUT/'class_statistics.csv','class_statistics.csv')
    shutil.copy2(OUT/'iris26_compact7_v4_labels.zip',OUT/'iris26_compact7_labels.zip')


def render_report(cfg,stats,records):
    options=''.join(f'<option value="{r["camera_id"]}">{r["camera_id"]} · {"训练" if r["split"]=="train" else "测试"}</option>' for r in records)
    classes=''.join(f'<tr><td><i style="background:rgb{tuple(c["color"])}"></i>{c["id"]} {c["zh"]}</td><td>{s["sphere_percent"]:.3f}%</td><td>{stats["filled_destinations"].get(c["id"],0):,}</td></tr>' for c,s in zip(cfg['classes'],stats['classes_after']))
    public=[{k:r[k] for k in ['camera_id','filled_pixels','filled_components']} for r in records]
    body='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>IRIS 26视角 · 7类完整语义 v4</title><style>*{box-sizing:border-box}body{margin:0;background:#f2f5f7;color:#193846;font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1500px;margin:auto;padding:24px}h1{font-size:31px}section{background:white;border:1px solid #dce5ea;border-radius:12px;padding:22px;margin:22px 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}figure{margin:0}figcaption{font-weight:650;margin:8px 0}img{max-width:100%;display:block}a{color:#087699}.muted{color:#617684}.controls{display:flex;gap:12px;flex-wrap:wrap}select,button{font:inherit;padding:8px;border:1px solid #afc4ce;border-radius:5px;background:white}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:8px;border-bottom:1px solid #dce5ea}i{display:inline-block;width:18px;height:18px;margin-right:8px}code{word-break:break-all}@media(max-width:850px){.grid{grid-template-columns:1fr}main{padding:12px}}</style><main>
<h1>IRIS 当前26张：完整7类语义，取消忽略区域</h1><p class="muted">v4 · 20训练＋6测试 · 未启动训练</p>
<section><h2>本次更新</h2><p>剩余 PIXELS 个忽略像素、COMPONENTS 处小区域，已按各区域边界上主要相邻的语义类别整体合并。最终标签仅含0–6，不再有255忽略标签。</p><p>此前确认的警示边沿、底部地面仍为建筑主体，细长金属支撑仍为支撑类。已有类别像素保持不变，本次只填补原忽略区域。</p><p><a href="iris26_compact7_v4_labels.zip">下载新版标签 v4</a> · <a href="classes.json?v=4">7类定义</a> · <a href="completion_components.json">逐区域合并记录</a> · <a href="qa.json?v=4">检查记录</a></p><img src="legend.png?v=4" alt="仅7类，无忽略类"></section>
<section id="gallery"><h2>逐张查看</h2><div class="controls"><button id="prev">上一张</button><select id="camera">OPTIONS</select><button id="next">下一张</button></div><p id="meta" class="muted"></p><div class="grid"><figure><figcaption>去支架 RGB</figcaption><a id="a0"><img id="im0"></a></figure><figure><figcaption>最新 v4：完整7类语义</figcaption><a id="a1"><img id="im1"></a></figure><figure><figcaption>上一版 v3：尚有少量黑色忽略区</figcaption><a id="a2"><img id="im2"></a></figure><figure><figcaption>粉框：本次合并位置（仅展示用）</figcaption><a id="a3"><img id="im3"></a></figure></div><h3>局部放大：RGB / 修改前 / 修改后</h3><p class="muted">展示本视角面积最大的三处；中间图用粉色突出原忽略区域，最终标签没有粉色忽略类别。</p><a id="detailLink"><img id="detail"></a></section>
<section><h2>类别统计</h2><table><thead><tr><th>类别</th><th>平均球面占比</th><th>本次并入像素</th></tr></thead><tbody>CLASSES</tbody></table><p>剩余忽略像素：0。球面面积按ERP纬度余弦加权。</p><p><a href="class_statistics.csv?v=4">CSV</a> · <a href="manifest.json?v=4">逐图清单</a> · <a href="statistics.json?v=4">完整统计</a></p></section>
<section><h2>输出说明</h2><p>标签文件 <code>labels</code> 仅含7类索引；语义有效掩码覆盖全图。来源图 <code>provenance</code> 的编码4表示本次邻域合并，并非新增语义类别。补画地面的来源编码3与RGB观测掩码继续单独保留。</p><p>原始官方标注及旧版标签保留；当前结果是派生标注，邻域补全不会标作官方原始真值。未修改RGB、相机和数据划分。</p><p>服务器目录：<code>EXPORTPATH</code></p></section></main>
<script>const rows=ROWS;const sel=document.querySelector('#camera');const paths=[['rgb','.jpg'],['final_semantics','.png'],['before_semantics','.png'],['ignore_highlights','.jpg']];function show(){let id=sel.value,r=rows.find(x=>x.camera_id===id);paths.forEach(([f,e],i)=>{let p=f+'/'+id+e+'?v=4';document.querySelector('#im'+i).src=p;document.querySelector('#a'+i).href=p;});document.querySelector('#meta').textContent=id+' · 已合并 '+r.filled_components+' 处、'+r.filled_pixels+' 像素 · 剩余忽略像素 0';let p='ignore_details/'+id+'.jpg?v=4';document.querySelector('#detail').src=p;document.querySelector('#detailLink').href=p;}sel.onchange=show;document.querySelector('#prev').onclick=()=>{sel.selectedIndex=(sel.selectedIndex+rows.length-1)%rows.length;show()};document.querySelector('#next').onclick=()=>{sel.selectedIndex=(sel.selectedIndex+1)%rows.length;show()};sel.value='1-68';show();</script></html>'''
    for key,value in {'PIXELS':f'{stats["filled_pixels"]:,}','COMPONENTS':str(stats['filled_components']),'OPTIONS':options,'CLASSES':classes,'EXPORTPATH':str(EXPORT),'ROWS':json.dumps(public)}.items():body=body.replace(key,value)
    (OUT/'report.html').write_text(body,encoding='utf-8')


if __name__=='__main__':
    main()
