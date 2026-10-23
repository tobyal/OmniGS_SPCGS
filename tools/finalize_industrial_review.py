"""Format already authored Codex observations; never performs image recognition."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spcgs.common import ROOT, read, write, ref

def main():
    p=argparse.ArgumentParser();p.add_argument('--review',default='runs/industrial_semantics_20261023/00_scene_review');a=p.parse_args()
    out=ROOT/a.review;manifest=read(out/'manifest.json');official=read(out/'official_categories.json')
    records=[]
    for f in sorted((ROOT/'docs/industrial_scene_observations').glob('authored*.json')):records+=read(f)['records']
    assert len(records)==56 and {r['name'] for r in records}=={v['name'] for v in manifest['views']}
    assert len({r['index'] for r in records})==56
    reviewed=read(ROOT/'docs/industrial_scene_observations/reviewed_rgb_hashes.json')
    assert all(reviewed.get(v['name'])==v['rgb']['sha256'] for v in manifest['views']), 'RGB differs from the images actually reviewed; new visual analysis is required'
    zooms=read(ROOT/'docs/industrial_scene_observations/zoom_review.json')['records']
    for z in zooms:
        r=next(r for r in records if r['index']==z['index']);r.setdefault('zoom_checks',[]).append(z)
        correction=z.get('correction',{})
        if correction.get('remove'):r['objects'].pop(correction['remove'],None)
        r['objects'].update(correction.get('add',{}))
    for r in records:
        if r['index'] in [52,53,54,55]:
            r['objects']['duct']='left/up：银色横向波纹风管；不包含旁边银色立筒'
            r['objects']['tank']='left：银色封闭立筒连接白色保温管；由56局部复核关联确认，具体用途不明'
        if r['index']==56:r['objects']['duct']='left/up：银色横向波纹风管，不包含下方封闭立筒'
    config=read(ROOT/'configs/industrial_class_definitions.json');mapping={}
    official_names={c['id']:c['name'] for c in official['categories']}
    for i,c in enumerate(config['classes']):
        c['id']=i;c['row']=i
        for k in c.get('candidate_keys',[c['key']]):mapping[k]=c
        c['official_mapping']=[{'id':j,'name':official_names[j],'relation':'category-level reference; not a pixel assignment'} for j in c['official_ids']]
        c['observed_training_images']=[r['name'] for r in records if any(k in r['objects'] for k in c.get('candidate_keys',[c['key']]))]
        c['observed_image_count']=len(c['observed_training_images']);assert c['observed_image_count']>0
        c['evidence']=[{'image':r['name'],'observation':r['objects'][k]} for r in records for k in c.get('candidate_keys',[c['key']]) if k in r['objects']]
    config['training_count']=56;config['official_reference']=ref(out/'official_categories.json')
    write(ROOT/'configs/industrial_classes.json',config);write(out/'industrial_classes.json',config)
    for r in records:
        v=next(v for v in manifest['views'] if v['name']==r['name']);objects=[]
        for k,evidence in r['objects'].items():
            c=mapping.get(k)
            objects.append({'candidate_key':k,'name_en':c['name_en'] if c else k,'name_zh':c['name_zh'] if c else '标志或杂物',
              'category_id':c['id'] if c else None,'visual_features_and_location':evidence,'recognition_evidence':evidence,
              'official_mapping':c['official_mapping'] if c else [],'confidence':c['confidence'] if c else 'not_selected',
              'confusions':c['confusions'] if c else [],'uncertainty':r['uncertainty']})
        record={'name':r['name'],'index':r['index'],'rgb':v['rgb'],'split':'train','status':'directly_reviewed','observer':'Codex native view_image; no external VLM',
          'viewed_material':{'contact_sheet':v['contact_sheet'],'erp_embedded_in_sheet':v['erp'],'six_perspectives_embedded_in_sheet':v['perspective_views'],
                            'zoom_images':[str(out/'zooms'/z['image']) for z in r.get('zoom_checks',[])]},
          'summary_zh':r['summary'],'objects':objects,'uncertainty':r['uncertainty'],'cross_view_association':r['cross_view'],'zoom_checks':r.get('zoom_checks',[])}
        write(out/'observations'/(r['name']+'.json'),record)
        v['visual_analysis_status']='directly_reviewed';v['observation']=str(out/'observations'/(r['name']+'.json'))
    manifest.update(status='completed',reviewed_training_count=56,reviewed_zoom_count=len(zooms),classes=str(ROOT/'configs/industrial_classes.json'))
    write(out/'manifest.json',manifest)
    lines=['# IRIS 工业类别分析','', '已由 Codex 原生图像查看实际检查 56/56 张训练图：每张联系表含完整 ERP 与六个100°透视面；另检查6个60°局部放大图。脚本只做投影、整理和完整性校验，不执行识别。未查看测试标注来设计词表，未使用分割掩膜监督。', '',
      '本地 IRIS 元数据有748个CAD实例名称，但共享16个类别ID；27张训练ERP有对应cubemap元数据。名称映射是语义辅助，不是逐像素真值。RMX位姿、56/6/16划分与2048×1024图像保持不变。', '',
      '选择20类：将泵和电机合为可见机组；将控制柜、接线箱及功能不确定的柜式机械外壳合为设备柜与外壳，避免泛machine与泵重叠。柱、支架、护栏、梯、桥架依据结构分开。新增容器、仪表、灯具和袋垛。色彩是辅助描述，不单独决定类别。', '',
      '局部复核修正：16号最初疑似板式换热器的矩形物是带敲落孔的接线箱背面；52至56号银筒连接保温管，应归容器候选而非风道。原始初判保留在authored记录，最终独立JSON包含修正。柜式外壳工艺、罐体用途、暗部机械细分仍不确定。', '',
      '| ID | 类别 | 出现图数 | 可信程度 | 文本依据/区别 |','|---|---|---:|---|---|']
    for c in config['classes']:lines.append(f"|{c['id']}|{c['name_en']} / {c['name_zh']}|{c['observed_image_count']}|{c['confidence']}|{'；'.join(c['traits'])}；易混：{', '.join(c['confusions'])}|")
    lines+=['','每类描述组合标准对象名、几何、材料、典型外观及位置关系，完整英文prompt和逐图证据见industrial_classes.json。类频率仅表示图像中观察到，不代表像素占比或独立实例数。','',
       '资料：[IRIS/IRIS-VIS官方说明](https://centreborelli.github.io/iris-vis/)、[数据集 DOI](https://doi.org/10.5281/zenodo.13777859)、[WACV2025论文](https://openaccess.thecvf.com/content/WACV2025/papers/Armangeon_IRIS-VIS_A_New_Dataset_for_Visibility_Estimation_in_an_Industrial_WACV_2025_paper.pdf)。该工业场景包含复杂遮挡，官方数据提供LiDAR与CAD支持可见性研究。本研究只利用类别元数据帮助命名。']
    (out/'class_analysis_zh.md').write_text('\n'.join(lines),encoding='utf8')
    print('Finalized',len(records),'observations and',len(config['classes']),'classes')
if __name__=='__main__':main()
