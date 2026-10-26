"""Package the approved 18/6 RGB originals and existing semantic references."""
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE / 'runs/iris_fullclass_gpt_annotation_20261025'
OUT = ROOT / 'downloads/manual_annotation_20261025'
FACES = ['front', 'right', 'back', 'left', 'up', 'down']

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def serialized(value):
    return json.dumps(value, ensure_ascii=False, indent=2) + '\n'

def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    split = read(ROOT/'source_split.json')
    index = read(ROOT/'package_index.json')
    cfg = read(ROOT/'taxonomy/classes.json')
    source_manifest = read(ROOT/'manifest.json')
    approved = read(Path(source_manifest['split_path']))
    assert approved['train_ids'] == split['train_ids'] and approved['test_ids'] == split['test_ids']
    assert len(split['train_ids']) == 18 and len(split['test_ids']) == 6 and not split['validation_ids']
    assert {x['camera_id'] for x in index} == set(split['train_ids']+split['test_ids'])
    assert len(cfg['classes']) == 16 and cfg['ignore_id'] == 255
    allowed = {c['id'] for c in cfg['classes']} | {255}
    original_name = 'IRIS_originals_18train_6test_16K.zip'
    semantic_name = 'IRIS_semantic_reference_18train_6test.zip'
    original_root = 'IRIS_originals_18train_6test'
    semantic_root = 'IRIS_semantic_reference_18train_6test'
    records, originals_hashes = [], {}
    source_hashes_checked = 0
    mask_count = 0
    with zipfile.ZipFile(OUT/original_name, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as original_zip, zipfile.ZipFile(OUT/semantic_name, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as semantic_zip:
        for part in ['train','test']:
            for cid in split[part+'_ids']:
                src = ROOT/'images'/cid
                meta = read(src/'metadata.json')
                assert meta['split'] == part
                original = Path(meta['rgb_highres_source'])
                assert Image.open(original).size == (16384,8192)
                source_hash = digest(original)
                assert source_hash == meta['rgb_highres_sha256'], (cid, 'original source changed')
                original_relative = f'{part}/{original.name}'
                original_arcname = f'{original_root}/{original_relative}'
                original_zip.write(original, original_arcname)
                originals_hashes[original_arcname] = source_hash
                official = bool(meta['has_official_semantics'])
                labels_root = src if official else ROOT/'assisted_annotations'/cid
                source_name = 'official_coco_derived' if official else 'ai_assisted_pending_review'
                folder = f'{part}/{cid}__{meta["image_id"]}'
                current_erp = np.asarray(Image.open(labels_root/'erp/labels.png'))
                current_record = {
                    'camera_id':cid, 'image_id':meta['image_id'], 'split':part,
                    'original_rgb':original_relative,'original_dimensions':[16384,8192],
                    'original_sha256':source_hash, 'semantic_reference_folder':folder,
                    'semantic_source':source_name,'has_official_annotation':official,
                    'assisted_annotation':not official,'human_confirmed':False,
                    'erp_dimensions':[2048,1024],'face_dimensions':[1500,1500],
                    'erp_label_coverage':float((current_erp != 255).mean()),
                    'ignore_id':255,
                }
                records.append(current_record)
                provenance = dict(current_record)
                provenance['coordinates'] = 'coordinates.json'
                if official:
                    provenance['official_coco_sha256'] = source_manifest['official_coco_sha256']
                    provenance['source_detail'] = 'Existing original-16-class masks decoded from IRIS official CAD-projected COCO RLE; ERP uses existing cubemap projection.'
                    semantic_zip.write(src/'official_annotations.json', f'{semantic_root}/{folder}/official_annotations.json')
                else:
                    assisted = read(labels_root/'annotation.json')
                    provenance['assisted_details'] = {k:v for k,v in assisted.items() if k not in ('coordinate_reference','faces')}
                    provenance['source_detail'] = 'AI-assisted partial labels; not official or human-confirmed truth. Unknown pixels remain ignore.'
                semantic_zip.writestr(f'{semantic_root}/{folder}/provenance.json', serialized(provenance))
                semantic_zip.write(src/'coordinates.json', f'{semantic_root}/{folder}/coordinates.json')
                original_checks = read(src/'checksums.json')
                assisted_checks = {} if official else read(labels_root/'checksums.json')
                for view in ['erp'] + ['faces/'+f for f in FACES]:
                    rgb_name = 'rgb.png' if view == 'erp' else 'rgb.jpg'
                    for name in [rgb_name,'labels.png','semantic.jpg','overlay.jpg']:
                        relative = view+'/'+name
                        file = (src if name == rgb_name else labels_root)/relative
                        checks = original_checks if (official or name == rgb_name) else assisted_checks
                        assert digest(file) == checks[relative], (cid, relative, 'reference changed')
                        source_hashes_checked += 1
                        if name == 'labels.png':
                            mask = np.asarray(Image.open(file))
                            assert mask.dtype == np.uint8
                            assert mask.shape == ((1024,2048) if view == 'erp' else (1500,1500))
                            assert set(np.unique(mask).tolist()) <= allowed
                            mask_count += 1
                        semantic_zip.write(file, f'{semantic_root}/{folder}/{relative}')
                print('PACKED',cid,part,source_name,flush=True)
        small_split = {k:split[k] for k in ['status','train_ids','test_ids','validation_ids']}
        for z, root in [(original_zip,original_root),(semantic_zip,semantic_root)]:
            z.writestr(root+'/image_list.json', serialized(records))
            z.writestr(root+'/split.json', serialized(small_split))
            for part in ['train','test']:
                z.writestr(root+'/'+part+'_list.txt', ''.join(f'{r["camera_id"]}\t{r["image_id"]}\t{r["semantic_source"]}\n' for r in records if r['split']==part))
        original_zip.writestr(original_root+'/README.txt',
            'IRIS 当前批准划分：18 张训练 + 6 张测试，不设验证集。\n'
            'train/ 与 test/ 包含服务器上的原始 16384×8192 JPEG，未缩放、未重新压缩；保留原始时间戳文件名。\n'
            'image_list.json 记录相机编号、划分、SHA-256 和对应语义来源。\n'
            '语义参考另见 IRIS_semantic_reference_18train_6test.zip。其 ERP 标签为 2048×1024，并非 16K 标签。\n'
            '如需将标签放大到原图尺寸，只能对类别 ID 做最近邻缩放；放大不会增加真实标注细节。\n')
        for name in ['classes.json','classes.csv','raw_categories.json','legend.png']:
            semantic_zip.write(ROOT/'taxonomy'/name, f'{semantic_root}/taxonomy/{name}')
        palette_lines = ['ID\t中文\tEnglish\tHEX\tRGB']
        for c in sorted(cfg['classes'],key=lambda x:x['id']):
            rgb=c['display_color'];color='#'+''.join(f'{v:02X}' for v in rgb)
            palette_lines.append(f'{c["id"]}\t{c["name_zh"]}\t{c["official_semantic_name"]}\t{color}\t{rgb}')
        palette_lines.append('255\t未标注／忽略\tignore\t#232832\t[35, 40, 50]')
        semantic_zip.writestr(semantic_root+'/taxonomy/category_colors.txt','\n'.join(palette_lines)+'\n')
        semantic_zip.writestr(semantic_root+'/README.txt',
            'IRIS 手动标注参考 — 当前 18 train + 6 test / 无验证集\n\n'
            '每张图独立放在 train/ 或 test/ 下的“相机编号__拍摄时间”目录。\n'
            'erp/：2048×1024 RGB、labels.png 类别 ID、semantic.jpg 彩色语义和 overlay.jpg 叠加图。\n'
            'faces/front,right,back,left,up,down/：1500×1500、90° 对应视图，文件组成同上。\n'
            'taxonomy/：官方 16 类的中文、英文、原始 ID、CAD 名称及显示颜色；255 为 ignore，9 不是未知或背景。\n'
            'coordinates.json：已有 ERP 与 cubemap 的坐标对应。\n\n'
            '标签来源：18 张为官方 COCO RLE 派生标签（训练 12、测试 6）；6 张训练图为辅助补标：1-32、1-33、1-43、1-45、1-46、1-56。\n'
            '每张 provenance.json 和总表 image_list.json 明确记录来源。辅助标签仍有缺失或误差，未经人工确认；官方 CAD 投影标签也需结合 RGB 检查遮挡及错位。\n\n'
            '在 X-AnyLabeling 中建议逐面打开 faces/*/rgb.jpg 标注，参照旁边的 semantic.jpg、overlay.jpg。\n'
            'labels.png 为单通道 uint8 像素类别 ID，不是彩色图，也不是可直接编辑的多边形 JSON。保存你自己的标注时选择新的输出目录。\n'
            '若直接标注全景，可打开 erp/rgb.png；16K 原图请从另一个原图包查看。ERP 标签只对应本包 2048×1024 RGB。\n'
            '本包只汇集已有结果，没有启动训练或重新推理，没有修改原始官方数据。\n')
    assert Counter(x['split'] for x in records) == {'train':18,'test':6}
    assert Counter(x['semantic_source'] for x in records) == {'official_coco_derived':18,'ai_assisted_pending_review':6}
    packages=[]
    for name in [original_name,semantic_name]:
        path=OUT/name
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
            assert len(z.namelist()) == len(set(z.namelist()))
            if name == original_name:
                for entry,want in originals_hashes.items():
                    assert hashlib.sha256(z.read(entry)).hexdigest() == want
            file_count=len(z.namelist())
        packages.append(dict(file=name,bytes=path.stat().st_size,sha256=digest(path),files=file_count))
    result=dict(passed=True,train_images=18,test_images=6,validation_images=0,original_size=[16384,8192],
                official_images=18,assisted_images=6,original_sha256_verified=24,
                reference_source_hashes_verified=source_hashes_checked,label_maps_verified=mask_count,
                packages=packages)
    (OUT/'delivery.json').write_text(serialized(result),encoding='utf-8')
    (OUT/'image_list.json').write_text(serialized(records),encoding='utf-8')
    print(serialized(result),flush=True)

if __name__ == '__main__':
    main()
