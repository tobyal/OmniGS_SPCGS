"""Publish reviewed assisted masks; keep official sources and old ZIPs intact."""
import hashlib
import html
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from spcgs.segmentation import from_faces
from add_iris_category_palette import palette, chip

ROOT = BASE / 'runs/iris_fullclass_gpt_annotation_20261025'
OUT = ROOT / 'assisted_annotations'
IDS = ['1-32', '1-33', '1-43', '1-45', '1-46', '1-56']
FACES = ['front', 'right', 'back', 'left', 'up', 'down']
NOTES = {
    '1-32': '已补近景大管、立式泵及圆形机座；Front 的小型设备和密集构件仍有空缺。Down 三脚架及拼接伪影保持 ignore。',
    '1-33': '已补白色弯管、泵机座、矩形基础及橙色土建柱。远处管道交叉、细支撑与遮挡边缘仍需复核。',
    '1-43': '已补电气柜、黑红泵、穿孔支撑、长条灯及银色通风弯头。阀门周边、细线和暗部仍有未标区域。',
    '1-45': '已补近景弯管、黑色泵、蓝色容器、泵机座及建筑柱；左面大管边界重新检查。机座与管道连接处、后方暗部仍需复核。',
    '1-46': '已补大弯管、泵体及机座、矩形基础。近景遮挡较多，细构件和顶面管线保留较多 ignore。',
    '1-56': '已补蓝色容器、泵座、控制盒及下部设备外壳。Right 控制盒按官方同类对象区分 7 与 9；纸张和无法判断的线缆保留 ignore。',
}

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def write(p, data):
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def figure(src, caption, width=1500, height=1500):
    return f'<figure><a href="{src}" target="_blank"><img loading="lazy" src="{src}" width="{width}" height="{height}" alt="{html.escape(caption)}"></a><figcaption>{caption} · 点击查看原尺寸</figcaption></figure>'

def document(title, navigation, content, style):
    return f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>{style}</head><body><header><h1>{title}</h1><nav>{navigation}</nav></header><main>{content}</main></body></html>'

def main():
    main_path = ROOT / 'report.html'
    main_html = main_path.read_text(encoding='utf-8')
    style = re.search(r'<style>.*?</style>', main_html, flags=re.S).group()
    cfg = read(ROOT / 'taxonomy/classes.json')
    classes = sorted(cfg['classes'], key=lambda c: c['id'])
    allowed = {c['id'] for c in classes} | {255}
    assert len(classes) == 16 and cfg['ignore_id'] == 255
    palette_classes = classes + [{'id':255, 'official_semantic_name':'ignore', 'name_zh':'未标注／忽略', 'display_color':[35,40,50]}]
    by_id = {c['id']:c for c in palette_classes}
    records, checks, cards = [], [], []
    total_source_hashes = 0
    for cid in IDS:
        source = ROOT / 'images' / cid
        dest = OUT / cid
        record = read(dest / 'annotation.json')
        meta = read(source / 'metadata.json')
        assert not meta['has_official_semantics'] and not record['has_official_annotation']
        baseline = read(source / 'checksums.json')
        # Verify immutable RGB and original empty masks against pre-existing checksums.
        immutable = [n for n in baseline if n.startswith(('faces/', 'erp/'))]
        for n in immutable:
            assert digest(source / n) == baseline[n], (cid, n, 'source changed')
        total_source_hashes += len(immutable)
        current_faces = {}
        for face in FACES:
            old = np.asarray(Image.open(source / 'faces' / face / 'labels.png'))
            assert np.all(old == 255), (cid, face, 'unexpected original mask')
            mask = np.asarray(Image.open(dest / 'faces' / face / 'labels.png'))
            assert mask.shape == (1500, 1500) and mask.dtype == np.uint8
            assert set(np.unique(mask).tolist()) <= allowed
            current_faces[face] = mask
        erp = np.asarray(Image.open(dest / 'erp/labels.png'))
        assert erp.shape == (1024, 2048) and erp.dtype == np.uint8
        assert np.array_equal(from_faces(current_faces), erp), (cid, 'ERP projection mismatch')
        assert np.all(np.asarray(Image.open(source / 'erp/labels.png')) == 255)
        record['review_status'] = 'AI_visually_reviewed_ERP_and_all_six_RGB_and_overlay_faces_pending_human_review'
        record['review_notes_zh'] = NOTES[cid]
        record['human_confirmed'] = False
        record['reference_split_information'] = [dict(camera_id=r, split=read(ROOT/'images'/r/'metadata.json')['split']) for r in record['reference_cameras']]
        record['original_transfer_faces'] = record.pop('original_transfer_faces', record['faces'])
        record['faces'] = [dict(face=f, coverage=float((m != 255).mean()), class_pixels={str(int(k)):int((m==k).sum()) for k in np.unique(m)}, label_path=f'faces/{f}/labels.png') for f,m in current_faces.items()]
        record['coverage'] = float((erp != 255).mean())
        record['coverage_definition'] = 'Fraction of ERP pixels assigned an official class ID; this is not an accuracy estimate.'
        record['label_encoding'] = 'uint8 PNG; original official IDs; 255 is ignore; JPEG semantic maps are visualizations only.'
        record['review_method'] = 'Actual visual inspection of all six RGB faces and successive overlays, official neighbour comparison, SAM2 target-image masks refined using object-specific positive/negative points; boxes were prompts, never painted rectangles.'
        record['coordinate_reference'] = f'../../images/{cid}/coordinates.json'
        write(dest/'annotation.json', record)
        hashes = {str(p.relative_to(dest)).replace('\\','/'):digest(p) for p in sorted(dest.rglob('*')) if p.is_file() and p.suffix in ('.png','.jpg')}
        write(dest/'checksums.json', hashes)
        checks.append(dict(camera_id=cid, source_files_checked=len(immutable), original_masks_all_ignore=True, face_dimensions=[1500,1500], erp_dimensions=[2048,1024], official_ids_only=True, erp_projection_exact=True))
        rec = {k:record[k] for k in ['camera_id','image_id','split','has_official_annotation','human_confirmed','coverage','reference_cameras','review_notes_zh']}
        rec['report'] = f'../images/{cid}/index.html'
        rec['erp_labels'] = f'{cid}/erp/labels.png'
        records.append(rec)
        prefix = f'../../assisted_annotations/{cid}'
        refs = ' · '.join(f'<a href="../{r}/index.html">{r} 官方标注参考</a>' for r in record['reference_cameras'])
        nav = '<a href="../../report.html">完整报告</a><a href="../../assisted_annotations/index.html">6 张补标总览</a><a href="#erp">ERP</a>' + ''.join(f'<a href="#{f}">{f.title()}</a>' for f in FACES)
        body = palette(palette_classes, 'taxonomy/index.html', True)
        body += f'<section><p><span class="badge">辅助补标 · 待复核</span><span class="badge missing">无官方标注</span> train · {record["image_id"]}</p><p>ERP 已赋类别像素 <b>{record["coverage"]:.1%}</b>；其余为 ignore（255）。这是覆盖率，不是准确率。</p><p>{NOTES[cid]}</p><p>{refs}</p><p>沿用官方 16 类和当前报告配色；结合邻近官方掩码、实际 RGB 视觉检查及冻结 SAM2 生成对象边界。本页显示辅助标签，不能视为官方或人工确认真值。深色区域为 ignore。</p><p><a download href="{prefix}/erp/labels.png">下载 ERP 类别 ID 标签 PNG</a> · <a href="{prefix}/annotation.json">来源、统计与复核记录 JSON</a> · <a href="coordinates.json">投影坐标说明</a></p></section>'
        body += '<section id="erp"><h2>完整 ERP · 2048 × 1024</h2><div class="pair">'+figure('erp/rgb.png','原始 RGB',2048,1024)+figure(prefix+'/erp/overlay.jpg','辅助标签叠加',2048,1024)+'</div><details><summary>展开纯语义颜色图</summary>'+figure(prefix+'/erp/semantic.jpg','辅助语义',2048,1024)+'</details></section>'
        for face, mask in current_faces.items():
            ids = [int(k) for k in np.unique(mask) if k != 255]
            keys = ''.join(chip(by_id[k], 'taxonomy/index.html') for k in ids)
            body += f'<section id="{face}"><h2>{face.title()} · 90° / 1500 × 1500</h2><p>当前辅助标签覆盖 {(mask!=255).mean():.1%} · <a download href="{prefix}/faces/{face}/labels.png">下载类别 ID PNG</a></p><div>{keys}</div><div class="grid">'
            body += figure(f'faces/{face}/rgb.jpg','原始 RGB')+figure(f'{prefix}/faces/{face}/semantic.jpg','辅助语义')+figure(f'{prefix}/faces/{face}/overlay.jpg','RGB + 辅助标签')+'</div></section>'
        body += '<section><h2>复核说明</h2><p>细线、反光、遮挡边缘与类别依据不足的对象仍可能缺标或有误差。三脚架与无依据的物品保持 ignore；9 / MISCELLANEOUS 不是未知或背景。本轮未对缺乏可靠例证的 13 / FIRE-FIGHTING 自动赋值，类别本身仍保留。</p><p>此前 ZIP 为初始资料快照，不含本轮补标；以本页链接的 PNG 与 JSON 为当前结果。</p></section>'
        (source/'index.html').write_text(document(f'IRIS {cid} · 辅助语义标注',nav,body,style),encoding='utf-8')
        baseline['index.html'] = digest(source/'index.html')
        write(source/'checksums.json',baseline)
        cards.append(f'<article><h2><a href="../images/{cid}/index.html">{cid} · {record["image_id"]}</a></h2><p><span class="badge missing">无官方标注</span><span class="badge">已辅助补标 · 待复核</span>覆盖 {record["coverage"]:.1%}</p>'+figure(f'{cid}/erp/overlay.jpg',f'{cid} 辅助标签叠加',2048,1024)+f'<p>{NOTES[cid]}</p><a href="../images/{cid}/index.html">查看 RGB、六面语义与叠加</a> · <a download href="{cid}/erp/labels.png">ERP 标签 PNG</a></article>')
    summary = dict(official_category_count=16,ignore_id=255,source='AI-assisted, not official ground truth',human_confirmed=False,training_performed=False,zip_packages_regenerated=False,images=records)
    write(OUT/'summary.json',summary)
    validation = dict(passed=True,original_source_files_checked=total_source_hashes,images=checks,official_class_table_sha256=digest(ROOT/'taxonomy/classes.json'),scope='Mask dimensions/IDs, source preservation and projection consistency; not semantic accuracy validation.')
    write(OUT/'validation.json',validation)
    overview = palette(palette_classes,'../taxonomy/index.html',True)
    overview += '<section class="notice"><h2>6 张未标注图像 · 已生成首轮辅助标签</h2><p>每张均有 6 面 1500 × 1500 类别 ID PNG，以及 2048 × 1024 ERP 标签。颜色和 ID 沿用官方 16 类；不确定像素保留 255。已逐面查看 RGB 和叠加结果并修正主要对象，仍需逐图复核。</p><p><a href="summary.json">汇总 JSON</a> · <a href="validation.json">数据与投影检查</a> · <a href="../taxonomy/index.html">完整类别参考</a></p></section>'
    overview += '<div class="pair">'+''.join(cards)+'</div>'
    (OUT/'index.html').write_text(document('IRIS · 6 张图像辅助语义标注','<a href="../report.html">返回完整报告</a>',overview,style),encoding='utf-8')
    start,end='<!-- ASSISTED_ANNOTATION_START -->','<!-- ASSISTED_ANNOTATION_END -->'
    main_html=re.sub(re.escape(start)+'.*?'+re.escape(end),'',main_html,flags=re.S)
    notice=start+'<section class="notice"><h2><a href="assisted_annotations/index.html">更新：6 张未标注图像的辅助语义标签</a></h2><p>1-32、1-33、1-43、1-45、1-46、1-56 已生成并逐面检查首轮补标，沿用官方 16 类及颜色。点击对应图像可查看当前结果；官方标注状态仍为“无官方标注”。此前 ZIP 未更新，不含本轮补标。</p></section>'+end
    main_html=main_html.replace('<main>','<main>'+notice,1)
    for cid in IDS:
        main_html=main_html.replace(f'<h3>{cid} <span class="badge missing">未标注</span>',f'<h3>{cid} <span class="badge missing">辅助补标 · 无官方</span>')
    main_path.write_text(main_html,encoding='utf-8')
    print(json.dumps({'passed':True,'images':[(r['camera_id'],round(r['coverage']*100,1)) for r in records],'source_files_checked':total_source_hashes}))

if __name__ == '__main__':
    main()
