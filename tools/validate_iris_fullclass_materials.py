"""Read-only content/geometry/package audit; writes a separate QA report."""
import collections, hashlib, importlib.util, json, sys, zipfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import numpy as np
from PIL import Image
from prepare_iris_fullclass_gpt import OUT, COCO, SPLIT, FACES, read, write, sha, decode_rle, project_faces

class Links(HTMLParser):
    def __init__(self):super().__init__();self.refs=[]
    def handle_starttag(self,tag,attrs):
        for k,v in attrs:
            if k in ('href','src') and v:self.refs.append(v)

def main():
    manifest=read(OUT/'manifest.json');split=read(SPLIT);records=read(OUT/'package_index.json')
    assert sha(SPLIT)==manifest['split_sha256']
    assert sha(COCO)==manifest['official_coco_sha256']
    raw=read(COCO);categories=read(OUT/'taxonomy/raw_categories.json');assert raw['categories']==categories
    ids={c['id'] for c in categories};assert ids=={2,5,6,7,9,12,13,16,19,21,24,27,29,30,31,32}
    assert len(categories)==748 and len(records)==24
    expected={x['camera_id']:x for x in split['selected_cameras']}
    assert {r['camera_id'] for r in records}==set(expected)
    assert collections.Counter(r['split'] for r in records)=={'train':18,'test':6}
    annotations={a['id']:a for a in raw['annotations']};images={a['id']:a for a in raw['images']}
    counts=collections.Counter();stats=[]
    for r in records:
        d=OUT/'images'/r['camera_id'];m=read(d/'metadata.json');subset=read(d/'official_annotations.json');tasks=read(d/'tasks.json')
        assert m['split']==expected[r['camera_id']]['split'] and not m['human_confirmed']
        assert len(subset['images'])==(6 if r['official'] else 0)
        for a in subset['annotations']:assert annotations[a['id']]==a
        for im in subset['images']:assert images[im['id']]==im
        counts['official_images' if r['official'] else 'unlabelled_images']+=1
        grouped=collections.defaultdict(list)
        for a in subset['annotations']:grouped[a['image_id']].append(a)
        face_labels={};face_masks={c:{} for c in ids}
        for face,_ in FACES:
            f=d/'faces'/face;fm=read(f/'statistics.json');label=np.array(Image.open(f/'labels.png'))
            assert label.shape==(1500,1500) and set(np.unique(label))<=ids|{255}
            reconstructed={}
            for a in grouped[fm['official_image_id']]:
                mask=decode_rle(a['segmentation']);assert int(mask.sum())==a['area']
                reconstructed[a['category_id']]=reconstructed.get(a['category_id'],np.zeros((1500,1500),bool))|mask
                counts['rle_annotations']+=1
            occ=np.zeros(label.shape,np.uint8);expected_label=np.full(label.shape,255,np.uint8)
            for c in sorted(ids):
                mask=reconstructed.get(c,np.zeros(label.shape,bool));p=f/f'class_masks/{c:02d}.png'
                if mask.any():assert np.array_equal(np.array(Image.open(p))>0,mask)
                else:assert not p.exists()
                face_masks[c][face]=mask.astype('u1');occ+=mask;expected_label[mask]=c
            expected_label[occ>1]=255;assert np.array_equal(expected_label,label)
            prov=np.where(occ>1,2,np.where(label!=255,1,0)).astype('u1')
            assert np.array_equal(prov,np.array(Image.open(f/'provenance.png')))
            face_labels[face]=label;counts['faces']+=1
        erp=np.array(Image.open(d/'erp/labels.png'));assert np.array_equal(erp,project_faces(face_labels))
        for c,fs in face_masks.items():
            mask=project_faces(fs)>0;p=d/f'erp/class_masks/{c:02d}.png'
            if mask.any():assert np.array_equal(mask,np.array(Image.open(p))>0)
            else:assert not p.exists()
        assert Image.open(d/'erp/rgb.png').size==(2048,1024)
        assert sha(Path(m['rgb_2k_source']))==m['rgb_2k_sha256']
        assert np.array_equal(np.array(Image.open(d/'erp/rgb.png')),np.array(Image.open(m['rgb_2k_source']).convert('RGB')))
        assert not tasks['human_confirmed'] and not tasks['is_precise_mask'] and len(tasks['faces'])==6
        for f in tasks['faces']:
            for t in f['priority_tasks']:
                assert t['final_category_id'] is None and not t['human_confirmed'] and not t['is_precise_mask']
                for p in t['evidence_images']+t['category_reference_images']:assert (d/p).is_file(),(d,p)
                counts['priority_tasks']+=1
        for p in d.rglob('*.html'):
            parser=Links();parser.feed(p.read_text(encoding='utf-8'))
            for link in parser.refs:
                url=urlsplit(link)
                if not url.scheme and url.path:assert (p.parent/unquote(url.path)).is_file(),(p,link)
        checks=read(d/'checksums.json')
        for relative,digest in checks.items():assert sha(d/relative)==digest,(d,relative)
        zp=OUT/r['zip'];assert sha(zp)==r['zip_sha256']
        with zipfile.ZipFile(zp) as z:
            assert z.testzip() is None
            assert len(z.namelist())==len(checks)+1
            for relative,digest in checks.items():
                assert hashlib.sha256(z.read(r['camera_id']+'/'+relative)).hexdigest()==digest
        stats.append({'camera_id':r['camera_id'],'split':r['split'],'official':r['official'],'source_RLE_verified':True,'ERP_masks_verified':True,'ZIP_hashes_verified':True})
        print('validated',r['camera_id'],flush=True)
    for p in [OUT/'report.html',OUT/'taxonomy/index.html']:
        parser=Links();parser.feed(p.read_text(encoding='utf-8'))
        for link in parser.refs:
            url=urlsplit(link)
            if not url.scheme and url.path:assert (p.parent/unquote(url.path)).is_file(),(p,link)
    spec=importlib.util.spec_from_file_location('coord',OUT/'coordinate_tools.py');coord=importlib.util.module_from_spec(spec);spec.loader.exec_module(coord)
    rng=np.random.default_rng(37);errors=[]
    for face,_ in FACES:
        pts=rng.uniform(0,1499,(2000,2));uv=coord.face_to_erp(face,pts);back,valid=coord.erp_to_face(face,uv)
        assert valid.all();errors.append(float(np.abs(back-pts).max()))
    assert max(errors)<1e-8
    try:
        import jsonschema
        jsonschema.Draft202012Validator.check_schema(read(OUT/'decision_schema.json'));schema_status='valid_draft202012'
    except ImportError:schema_status='jsonschema_unavailable; schema parsed as JSON'
    report={'passed':True,'counts':dict(counts),'split_source_unchanged':True,'COCO_source_unchanged':True,'all_748_raw_categories_preserved':True,
      'rle_union_vs_class_masks_exact':True,'composite_conflicts_preserved':True,'ERP_projection_vs_faces_exact':True,'ZIP_CRC_and_SHA256_all_files_checked':True,
      'HTML_local_links_resolve':True,'coordinate_roundtrip_max_face_px':max(errors),'decision_schema':schema_status,'images':stats,
      'limits':'Structural/content checks do not certify official CAD labels or AI prompts as pixel ground truth.'}
    write(OUT/'quality_validation.json',report);print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
