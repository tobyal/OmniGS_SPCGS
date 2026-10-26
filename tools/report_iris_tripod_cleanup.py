"""Build a local static review and audit the masked panorama composites."""
from pathlib import Path
import json, sys, shutil, zipfile, html
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spcgs.common import ROOT,read,write,sha

RUN=ROOT/'runs/iris_tripod_cleanup_20261025'
DATA=ROOT.parent/'datasets/iris_official20_6_no_tripod_2k_20261025'
SPLIT=ROOT/'runs/iris_camera_analysis_20261025/official_overlap_split_20261025'
OUT=RUN/'review'

def main():
    result=read(RUN/'cleanup_results.json');records=result['records'];assert len(records)==26
    manifest=read(DATA/'dataset_manifest.json')
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    audit=[]
    weights=np.cos((.5-(np.arange(1024)+.5)/1024)*np.pi)[:,None]
    for r in records:
        src=np.array(Image.open(r['source_rgb']['path']).convert('RGB'))
        dst=np.array(Image.open(r['clean_rgb']['path']).convert('RGB'))
        valid=np.array(Image.open(r['valid_mask']['path']))
        edit=np.array(Image.open(r['edit_mask']['path']))>0
        assert src.shape==dst.shape==(1024,2048,3)
        assert np.array_equal(valid,(~edit).astype('u1')*255)
        assert np.array_equal(src[~edit],dst[~edit])
        assert sha(r['source_rgb']['path'])==r['source_rgb']['sha256']
        assert sha(r['clean_rgb']['path'])==r['clean_rgb']['sha256']
        assert edit.any() and not edit.all() and r['rig_fully_replaced']
        r['edit_fraction_sphere']=float((edit*weights).sum()/(weights.sum()*2048))
        audit.append({'camera_id':r['camera_id'],'resolution_ok':True,'original_sha256_unchanged':True,'pixels_changed_outside_edit_mask':0,'mask_inverse_consistent':True,'rig_mask_fully_replaced':True,'edit_fraction_erp':r['edit_fraction_erp'],'edit_fraction_sphere':r['edit_fraction_sphere']})
    for role,count in [('train',20),('test',6)]:
        old=read(SPLIT/'selected_dataset/openMVG'/f'data_openmvg_{role}.json')
        new=read(DATA/'openMVG'/f'data_openmvg_{role}.json')
        old.pop('root_path');new.pop('root_path');assert old==new
        assert len([r for r in records if r['split']==role])==count
    assert len({r['image_id'] for r in records})==26
    qa={'automated_checks_passed':True,'count':26,'train_count':20,'test_count':6,'validation_count':0,
        'cameras_unchanged_except_image_root':True,'source_images_unchanged':True,'outside_mask_pixel_changes':0,
        'initial_cloud_reused':manifest['initial_cloud_reused'],
        'initial_cloud_observations_on_edited_pixels':manifest['initial_cloud_observations_on_edited_pixels'],
        'visual_review':'pending final Codex inspection','human_approval':False,'training_started':False,'images':audit}
    write(OUT/'qa.json',qa)
    result['records']=records;write(RUN/'cleanup_results.json',result);write(OUT/'cleanup_results.json',result)
    manifest['records']=records;write(DATA/'dataset_manifest.json',manifest);write(OUT/'dataset_manifest.json',manifest)
    for start in [0,13]:
        rows=records[start:start+13];sheet=Image.new('RGB',(4*320,4*350),'#e8edf1');draw=ImageDraw.Draw(sheet)
        for j,r in enumerate(rows):
            x=(j%4)*320;y=(j//4)*350;cid=r['camera_id']
            a=Image.open(RUN/'nadir_final'/(cid+'.png')).crop((320,320,704,704)).resize((312,312))
            sheet.paste(a,(x+4,y+34));draw.text((x+8,y+7),cid+' | '+r['split'],font=font,fill='#15394c')
        sheet.save(OUT/f'final_nadir_{start+1}_{start+len(rows)}.jpg',quality=96)
    for cid in ['1-27','1-47','1-58','1-59','1-64','1-65','1-67','1-69']:
        panel=Image.new('RGB',(1152,416),'#edf1f4');draw=ImageDraw.Draw(panel)
        for i,(folder,label) in enumerate([('inputs','Original'),('nadir_final','Final ERP projected down'),('alpha_down','Edit weight')]):
            im=Image.open(RUN/folder/(cid+'.png')).convert('RGB').crop((320,320,704,704))
            panel.paste(im,(384*i,32));draw.text((384*i+8,6),cid+' | '+label,font=font,fill='#18394b')
        panel.save(OUT/('detail_'+cid+'.jpg'),quality=97)
    public=[{'camera_id':r['camera_id'],'image_id':r['image_id'],'split':r['split'],'edit_pct':round(r['edit_fraction_erp']*100,3),'sphere_pct':round(r['edit_fraction_sphere']*100,3)} for r in records]
    options=''.join(f'<option value="{i}">{r["camera_id"]} · {"训练" if r["split"]=="train" else "测试"} · {r["image_id"]}</option>' for i,r in enumerate(records))
    table=''.join(f'<tr><td>{r["camera_id"]}</td><td>{"训练" if r["split"]=="train" else "测试"}</td><td>{r["image_id"]}</td><td>{r["edit_fraction_erp"]*100:.2f}%</td><td>{r["edit_fraction_sphere"]*100:.2f}%</td><td><a href="clean/{r["image_id"]}.png">图像</a> · <a href="valid_masks/{r["image_id"]}.png">掩码</a></td></tr>' for r in records)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>IRIS · 全景相机支架清理</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f1f4f7;color:#183243;font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1400px;margin:auto;padding:32px 24px}h1{margin:0 0 8px;font-size:32px}h2{font-size:23px;margin:0 0 16px}p{margin:10px 0}a{color:#09688a}section{background:white;border:1px solid #dce5eb;border-radius:14px;padding:24px;margin:22px 0}.muted{color:#526976}.tags{display:flex;gap:10px;flex-wrap:wrap;margin:18px 0}.tag{background:#dceef1;padding:5px 13px;border-radius:20px;font-weight:600}.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:12px 0}select,button{font:inherit;padding:8px 12px;border:1px solid #b8cbd7;background:white;border-radius:7px}button{cursor:pointer}.panorama{position:relative;width:100%;aspect-ratio:2;overflow:hidden;background:#1b2930}.panorama img{position:absolute;width:100%;height:100%;inset:0;object-fit:contain}.panorama #before{clip-path:inset(0 50% 0 0)}.line{position:absolute;left:50%;top:0;bottom:0;border-left:2px solid white;pointer-events:none}.badge{position:absolute;top:10px;padding:3px 10px;background:#132d40d9;color:white;border-radius:5px;pointer-events:none}.left{left:10px}.right{right:10px}input[type=range]{width:100%}#detail{width:100%;display:block}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.grid img{width:100%}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px 10px;border-bottom:1px solid #e0e8ee;text-align:left}.scroll{overflow:auto}code{word-break:break-all;background:#edf2f5;padding:2px 5px}details{margin:12px 0}summary{cursor:pointer;font-weight:650}.note{border-left:4px solid #dfa642;padding:10px 14px;background:#fff7e6}@media(max-width:760px){main{padding:18px 12px}section{padding:16px}.grid{grid-template-columns:1fr}h1{font-size:26px}}</style>
<main><h1>IRIS 全景相机支架清理</h1><p class="muted">当前官方语义候选集 · 2026-10-25 · 保留原始全景与相机参数</p><div class="tags"><span class="tag">20 张训练 + 6 张测试</span><span class="tag">2048 × 1024</span><span class="tag">26 / 26 已合成</span><span class="tag">未启动训练</span></div>
<section><h2>采用向下局部投影修补，再按掩码回贴</h2><p>全景底部属于球面极区，三脚架在等距柱状图像中被横向拉伸，并可能跨越左右边界。先投影成 100°、1024 × 1024 的向下视图，清除小型黑色相机三脚架，再把修补像素按原投影关系回贴。只更新支架附近掩码，掩码之外逐像素保持原图。</p><p>处理范围是当前选定的 26 张 2K 全景图。官方语义标签和源图未覆盖；真实工业立柱、设备基座应保留。1-27 的真实立柱底座使用原图保护区，避免生成编辑误删。</p><p class="note">支架遮挡的地面是生成补全，不能视为真实观测。每张图附有效像素掩码：255 为原始观测，0 为编辑区域。后续 RGB 损失、官方语义有效区与测试指标需使用此掩码；当前训练程序尚未接入这组掩码。本次未运行训练或更新测试结果。</p><p><a href="cleaned_images_and_masks.zip">下载全部清理图像与掩码</a> · <a href="dataset_manifest.json">数据清单</a> · <a href="qa.json">检查结果</a> · <a href="generation_prompts.json">内置 imagegen 提示词记录</a></p></section>
<section><h2>逐张全景对照</h2><div class="controls"><button id="prev">上一张</button><select id="camera">OPTIONS</select><button id="next">下一张</button></div><p id="meta" class="muted"></p><div class="panorama"><img id="after" alt="清理后全景"><img id="before" alt="原始全景"><div id="line" class="line"></div><span class="badge left">原图</span><span class="badge right">清理后</span></div><input id="wipe" type="range" min="0" max="100" value="50" aria-label="原图与清理后分界"><p><a id="rawlink">原图 PNG</a> · <a id="cleanlink">清理后 PNG</a> · <a id="masklink">有效像素掩码</a></p><h2>支架区域放大对照</h2><p class="muted">左：原始向下投影；中：最终全景重新向下投影；右：编辑区域。粉色区域是本次回贴范围。</p><img id="detail" alt="支架清理对照"></section>
<section><h2>26 张最终局部总览</h2><p class="muted">取自最终保存的全景图，中心裁切展示。可点击原尺寸查看。</p><div class="grid"><a href="final_nadir_1_13.jpg"><img src="final_nadir_1_13.jpg" alt="前13张清理结果"></a><a href="final_nadir_14_26.jpg"><img src="final_nadir_14_26.jpg" alt="后13张清理结果"></a></div></section>
<section><h2>图像与改动范围</h2><p class="muted">ERP 像素比例受极区拉伸影响，通常明显高于按球面面积加权的比例。两者仅描述编辑范围。</p><div class="scroll"><table><thead><tr><th>相机</th><th>划分</th><th>图像 ID</th><th>ERP 编辑比例</th><th>球面编辑比例</th><th>文件</th></tr></thead><tbody>TABLE</tbody></table></div></section>
<section><h2>复用与检查</h2><p>相机参数和 20/6 划分保持不变，无验证集。自动检查：26 张源文件 SHA-256 未变，输出尺寸正确，掩码外变化像素为 0，训练和测试图像无交集，原有相机文件仅修改图像根路径。</p><p>CLOUD</p><p>局部编辑使用内置 imagegen；SAM2 只用于标出相机支架，属于采集伪影掩码，不是官方语义标注。官方标签仍来自原始 IRIS 官方文件。生成图像仅用于各自视角的局部修补，没有把测试图内容用于训练图补全。</p><p>本次清理三脚架及邻近接触区域；远离支架的长条投影不作为扩大修补范围的依据。后续仍应检查渲染中底部的结构一致性。</p><p>服务器数据目录：<code>DATAPATH</code></p><p><a href="cleanup_results.json">逐图合成记录</a> · <a href="selected_cameras.csv">相机名单</a></p></section></main>
<script>const rows=DATAJSON;const select=document.querySelector('#camera');function show(){let r=rows[+select.value],n=r.image_id+'.png';document.querySelector('#before').src='original/'+n;document.querySelector('#after').src='clean/'+n;document.querySelector('#detail').src='compare/'+r.camera_id+'.jpg';document.querySelector('#rawlink').href='original/'+n;document.querySelector('#cleanlink').href='clean/'+n;document.querySelector('#masklink').href='valid_masks/'+n;document.querySelector('#meta').textContent=r.camera_id+' · '+(r.split==='train'?'训练':'测试')+' · ERP 编辑 '+r.edit_pct+'% · 球面编辑 '+r.sphere_pct+'%';}select.addEventListener('change',show);document.querySelector('#prev').onclick=()=>{select.value=(+select.value+rows.length-1)%rows.length;show()};document.querySelector('#next').onclick=()=>{select.value=(+select.value+1)%rows.length;show()};document.querySelector('#wipe').oninput=e=>{let p=+e.target.value;document.querySelector('#before').style.clipPath='inset(0 '+(100-p)+'% 0 0)';document.querySelector('#line').style.left=p+'%'};show();</script></html>'''
    count=len(manifest['initial_cloud_observations_on_edited_pixels'])
    cloud=f'现有初始稀疏点云的观测与编辑区相交数量：{count}。'+('未发现相交观测，已复用原初始点云。' if not count else '存在相交观测，因此未复制原点云；训练前需处理这些观测。')
    page=page.replace('OPTIONS',options).replace('TABLE',table).replace('CLOUD',cloud).replace('DATAPATH',str(DATA)).replace('DATAJSON',json.dumps(public,ensure_ascii=False))
    (OUT/'report.html').write_text(page,encoding='utf-8')
    with zipfile.ZipFile(OUT/'cleaned_images_and_masks.zip','w',compression=zipfile.ZIP_STORED) as z:
        for r in records:
            n=r['image_id']+'.png'
            for folder in ['images','valid_masks','edit_masks']:z.write(DATA/folder/n,folder+'/'+n)
        z.write(DATA/'dataset_manifest.json','dataset_manifest.json')
        z.write(OUT/'generation_prompts.json','generation_prompts.json')
        z.write(SPLIT/'selected_cameras.csv','selected_cameras.csv')
    print(json.dumps({'count':26,'outside_mask_changes':0,'track_observations_touching_edits':count,'sphere_percent_range':[min(r['edit_fraction_sphere'] for r in records)*100,max(r['edit_fraction_sphere'] for r in records)*100],'report':str(OUT/'report.html'),'zip_bytes':(OUT/'cleaned_images_and_masks.zip').stat().st_size},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
