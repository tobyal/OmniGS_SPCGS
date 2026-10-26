"""Review the user-requested curb and reconstructed-floor semantic revision."""
import json
from PIL import Image, ImageDraw
from consolidate_iris26_semantics import OUT, EXPORT, PREVIOUS, font, save, read


def report(cfg, audit, totals, records, corrections):
    old = read(PREVIOUS / 'manifest.json')
    old_ignore = sum(r['ignored_sphere_percent'] for r in old['records']) / len(records)
    # Contact sheets show all 26 lower hemispheres at an inspectable resolution.
    sheets = []
    for start in range(0, len(records), 8):
        subset = records[start:start+8]
        sheet = Image.new('RGB', (1536, ((len(subset)+1)//2)*448), '#f2f5f7')
        draw = ImageDraw.Draw(sheet)
        for i, r in enumerate(subset):
            cid = r['camera_id']; x = (i % 2)*768; y = (i // 2)*448
            draw.text((x+10, y+3), f'{cid} | RGB / v3 建筑主体', font=font(19), fill='#193846')
            for j, folder in enumerate(['rgb', 'final_semantics']):
                ext = '.jpg' if folder == 'rgb' else '.png'
                im = Image.open(OUT/folder/(cid+ext)).crop((0,480,2048,1024))
                im = im.resize((768,204), Image.Resampling.LANCZOS if j == 0 else Image.Resampling.NEAREST)
                sheet.paste(im, (x,y+32+j*204))
        name = f'building_all26_{start//8+1}.jpg'; save(OUT/name, sheet); sheets.append(name)
    # A self-contained full panorama before/after suitable for download.
    for cid in ['1-24', '1-26', '1-66', '1-68']:
        panel = Image.new('RGB', (1536, 4*800), '#f2f5f7')
        draw = ImageDraw.Draw(panel)
        for j, (folder, label, ext) in enumerate([
            ('rgb','去支架 RGB','.jpg'), ('before_semantics','修改前 v2','.png'),
            ('final_semantics','修改后 v3：警示边沿与底部地面 → 建筑主体','.png'),
            ('building_highlights','改动位置：绿 = 边沿；粉 = 底部补全地面','.jpg')]):
            draw.text((12,j*800+3),cid+' | '+label,font=font(23),fill='#193846')
            im=Image.open(OUT/folder/(cid+ext)).resize((1536,768),Image.Resampling.NEAREST if ext=='.png' else Image.Resampling.LANCZOS)
            panel.paste(im,(0,j*800+32))
        save(OUT/'building_comparisons'/(cid+'.jpg'),panel)
    options=''.join(f'<option value="{r["camera_id"]}">{r["camera_id"]} · {"训练" if r["split"]=="train" else "测试"}</option>' for r in records)
    table=''.join(f'<tr><td>{r["camera_id"]}</td><td>{"训练" if r["split"]=="train" else "测试"}</td><td>{r["building_curb_pixels"]:,}</td><td>{r["bottom_building_pixels"]:,}</td><td>{", ".join(map(str,r["building_curb_annotation_ids"]))}</td></tr>' for r in records)
    classes=''.join(f'<tr><td><i style="background:rgb{tuple(c["color"])}"></i>{c["id"]} {c["zh"]}</td><td>{c["definition"]}</td><td>{s["sphere_percent"]:.3f}%</td></tr>' for c,s in zip(cfg['classes'],totals['classes_after']))
    sheetlinks=''.join(f'<a href="{p}">全26张检查图 {i+1}</a>　' for i,p in enumerate(sheets))
    public=[{k:r[k] for k in ['camera_id','building_curb_pixels','bottom_building_pixels','ignored_sphere_percent']} for r in records]
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>IRIS 26视角 · 建筑主体修订 v3</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f2f5f7;color:#193846;font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1500px;margin:auto;padding:24px}h1{font-size:32px;margin:0}h2{font-size:23px}section{background:white;border:1px solid #dce5ea;border-radius:12px;margin:22px 0;padding:22px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}figure{margin:0}figcaption{font-weight:650;margin:8px 0}img{display:block;max-width:100%}a{color:#087699}.muted{color:#617684}.notice{padding:14px;background:#edf5fa;border-left:4px solid #1686a8}select,button{font:inherit;padding:8px 12px;border:1px solid #afc4ce;border-radius:5px;background:white}.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center}table{width:100%;border-collapse:collapse}th,td{text-align:left;border-bottom:1px solid #dce5ea;padding:8px}i{display:inline-block;width:18px;height:18px;margin-right:8px;vertical-align:middle}code{word-break:break-all}.scroll{overflow-x:auto}@media(max-width:850px){main{padding:12px}.grid{grid-template-columns:1fr}h1{font-size:25px}}</style>
<main><h1>26张语义修订：警示边沿与底部地面 → 建筑主体</h1><p class="muted">v3 · 20训练＋6测试 · 7类 · 2048×1024 · 未启动训练</p>
<section><h2>本次改了什么</h2><p>你截图对应1-68，页面默认显示该视角。按每张RGB核对，将贴近地面的黄黑警示边沿、台座裙边及对应侧面归入“建筑主体”（ID 2）。26张图底部已补全的地面也全部标为建筑主体。</p><p class="notice">本次只把确认的边沿与底部地面从 ignore 改为建筑主体。原先纠正的金属立柱、横撑、支撑底脚保持支撑类，其他类别像素保持不变。原始官方标注和 v2 标签均保留。</p><p>逐对象确认 CURBS 条边沿标注记录，覆盖全部26张；这是跨面、跨视角记录数，不是物理实体数量。改动包含 CURBPIX 个边沿像素和 FLOORPIX 个底部地面像素。</p><p><a href="iris26_compact7_v3_labels.zip">下载新版标签 v3（含掩码与来源）</a> · <a href="object_corrections.json?v=3">逐对象修改清单</a> · <a href="qa.json?v=3">检查记录</a></p><img src="legend.png?v=3" alt="7类图例"></section>
<section id="gallery"><h2>逐张查看修改前后</h2><div class="controls"><button id="prev">上一张</button><select id="camera">OPTIONS</select><button id="next">下一张</button><a id="download">下载当前新版语义图</a></div><p id="meta" class="muted"></p><div class="grid">
<figure><figcaption>RGB：去支架全景</figcaption><a id="a0"><img id="im0" alt="RGB"></a></figure>
<figure><figcaption>修改后 v3：建筑主体为灰蓝色</figcaption><a id="a1"><img id="im1" alt="v3语义"></a></figure>
<figure><figcaption>修改前 v2：黑色为 ignore</figcaption><a id="a2"><img id="im2" alt="v2语义"></a></figure>
<figure><figcaption>仅高亮本次改动：绿＝边沿，粉＝底部地面</figcaption><a id="a3"><img id="im3" alt="修改范围"></a></figure>
<figure><figcaption>修改后语义叠加 RGB</figcaption><a id="a4"><img id="im4" alt="语义叠加"></a></figure>
<figure><figcaption>来源参考：原16类（历史展示，屏蔽了底部补画区）</figcaption><a id="a5"><img id="im5" alt="原16类"></a></figure></div><p>点击图像可看原尺寸。完整对照长图：<a href="building_comparisons/1-68.jpg">1-68（你的截图）</a> · <a href="building_comparisons/1-24.jpg">1-24</a> · <a href="building_comparisons/1-26.jpg">1-26</a> · <a href="building_comparisons/1-66.jpg">1-66</a></p><p>SHEETS</p></section>
<section><h2>类别与覆盖</h2><table><thead><tr><th>类别</th><th>定义</th><th>平均球面占比</th></tr></thead><tbody>CLASSES</tbody></table><p>ignore 球面占比从 OLDIGNORE% 降至 NEWIGNORE%。面积按纬度余弦加权、以整个球面为分母；底部补画地面已计入建筑主体。</p><p><a href="classes.json?v=3">类别定义及提示词</a> · <a href="class_statistics.csv?v=3">类别统计</a> · <a href="statistics.json?v=3">完整统计</a></p></section>
<section><h2>逐图改动记录</h2><div class="scroll"><table><thead><tr><th>相机</th><th>用途</th><th>边沿像素</th><th>底部像素</th><th>边沿来源标注 ID</th></tr></thead><tbody>TABLE</tbody></table></div></section>
<section><h2>标签与来源分开保存</h2><p><code>labels</code> 是0–6及255的真实类别索引，底部地面为2；<code>valid_masks</code> 包含这些已赋类地面像素。<code>provenance</code> 单独记录来源：1＝官方类别合并，2＝经RGB检查的杂项改类，3＝补画地面。另附 <code>observed_valid_masks</code> 供只分析原始观测像素时使用。</p><p>这版是基于官方标注的派生语义，补画地面不会被标成“官方真值”。继承原投影轮廓；本次修改类别归属，不重新绘制全部遮挡边界。RGB、相机和20/6划分未改动。</p><p>服务器标签目录：<code>EXPORT</code></p><p><a href="manifest.json?v=3">逐图清单</a></p></section></main>
<script>const rows=ROWS;const sel=document.querySelector('#camera');const paths=[['rgb','.jpg'],['final_semantics','.png'],['before_semantics','.png'],['building_highlights','.jpg'],['final_overlays','.jpg'],['original_semantics','.png']];function show(){const id=sel.value,r=rows.find(x=>x.camera_id===id);paths.forEach(([folder,ext],i)=>{const p=folder+'/'+id+ext+'?v=3';document.querySelector('#im'+i).src=p;document.querySelector('#a'+i).href=p;});document.querySelector('#meta').textContent=id+' · 边沿→建筑 '+r.building_curb_pixels.toLocaleString()+' 像素 · 底部→建筑 '+r.bottom_building_pixels.toLocaleString()+' 像素';document.querySelector('#download').href='final_semantics/'+id+'.png?v=3';}sel.onchange=show;document.querySelector('#prev').onclick=()=>{sel.selectedIndex=(sel.selectedIndex+rows.length-1)%rows.length;show()};document.querySelector('#next').onclick=()=>{sel.selectedIndex=(sel.selectedIndex+1)%rows.length;show()};sel.value='1-68';show();</script></html>'''
    values={'CURBS':str(totals['reviewed_curb_annotation_records']), 'CURBPIX':f"{totals['building_curb_pixels']:,}", 'FLOORPIX':f"{totals['bottom_building_pixels']:,}", 'OPTIONS':options,'SHEETS':sheetlinks,'CLASSES':classes,'OLDIGNORE':f'{old_ignore:.3f}','NEWIGNORE':f"{totals['mean_ignore_sphere_percent']:.3f}",'TABLE':table,'EXPORT':str(EXPORT),'ROWS':json.dumps(public)}
    for key,value in values.items():html=html.replace(key,value)
    (OUT/'report.html').write_text(html,encoding='utf-8')
