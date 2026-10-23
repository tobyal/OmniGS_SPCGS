"""Geometry, data integrity, approval and concurrent-edit regression tests."""
import sys,io,json,zipfile,base64
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from spcgs.segmentation import *
from iris_review_server import create_app

@pytest.fixture
def dataset(tmp_path):
    cfg=read(CONFIG);write(tmp_path/'classes.json',cfg);rgb=tmp_path/'rgb.png';Image.new('RGB',(2048,1024),'white').save(rgb)
    write(tmp_path/'manifest.json',{'images':[{'id':'sample','index':1,'rgb':str(rgb),'rgb_sha256':sha(rgb)}]})
    s=Store(tmp_path);f=s.folder('sample');f.mkdir(parents=True);write(f/'coverage.json',[])
    labels=np.full((1024,2048),255,np.uint8);labels[400:600,0:100]=0;labels[400:600,-100:]=0
    Image.fromarray(labels).save(f/'official_mapped.png');Image.fromarray(labels).save(f/'official_raw.png')
    s.save('sample',labels,(labels!=255).astype('u1'),None,actor='test fixture');return s

def test_nearest_cube_projection_and_seam():
    faces={n:np.full((32,32),i,np.uint8) for i,(n,_) in enumerate(FACES)};out=from_faces(faces,32)
    assert set(np.unique(out))==set(range(6))
    assert out[512,1024]==0 and out[512,1536]==1 and out[512,512]==3
    assert out[512,0]==out[512,-1]==2
    assert out[0,1024]==4 and out[-1,1024]==5
    mask=mask_to_erp(np.ones((32,32),bool),'back',100)
    assert mask[512,0] and mask[512,-1] and not mask[512,1024]

def test_rle_fortran_order():
    m=decode_rle({'size':[2,3],'counts':[1,2,2,1]})
    assert m.tolist()==[[False,True,False],[True,False,True]]

def test_revision_approval_and_restore_export(dataset):
    s=dataset;labels,source,st=s.arrays('sample');original=st['revision']
    assert s.export(True)[1]==0
    s.review('sample',original,True,'TEST ONLY')
    assert s.export(True)[1]==1
    labels[500,1024]=6;source[500,1024]=3
    changed=s.save('sample',labels,source,original,actor='test')
    assert changed['approved_revision'] is None and s.export(True)[1]==0
    with pytest.raises(RuntimeError):s.save('sample',labels,source,original)
    with pytest.raises(RuntimeError):s.review('sample',original,True,'TEST ONLY')
    z,count=s.export(False);assert count==1
    with zipfile.ZipFile(io.BytesIO(z)) as f:
        im=Image.open(io.BytesIO(f.read('labels/sample.png')))
        assert im.mode=='L' and im.size==(2048,1024) and np.array(im)[500,1024]==6
        m=json.loads(f.read('manifest.json'));assert m['unreviewed_candidates_included'] and not m['images'][0]['human_approved']
        assert json.loads(f.read('metadata/sample.json'))['revision']==changed['revision']
    # Previous immutable revision remains intact.
    assert np.array(Image.open(s.folder('sample')/'revisions'/original/'labels.png'))[500,1024]==255

def test_api_edit_conflicts_validation_approval(dataset):
    app=create_app(dataset.root,False);c=app.test_client();d=c.get('/api/bootstrap').json;h={'X-Review-Token':d['token']};st=dataset.state('sample')
    labels,_,_=dataset.arrays('sample');labels[512,1024]=1
    body={'revision':st['revision'],'labels_u8':base64.b64encode(labels.tobytes()).decode()}
    assert c.post('/api/image/sample/save',json=body).status_code==403
    response=c.post('/api/image/sample/save',json=body,headers=h);assert response.status_code==200
    assert c.post('/api/image/sample/save',json=body,headers=h).status_code==409
    revision=response.json['revision']
    assert c.post('/api/image/sample/review',json={'revision':revision,'approve':True},headers=h).status_code==400
    assert c.post('/api/image/sample/review',json={'revision':revision,'approve':True,'confirmation':'human_reviewed','reviewer':'TEST ONLY'},headers=h).status_code==200
    assert c.get('/api/export?scope=approved').status_code==200
    assert c.post('/api/image/sample/restore',json={'revision':revision,'restore_revision':st['revision']},headers=h).status_code==200
    assert dataset.state('sample')['approved_revision'] is None
    arr=np.array(Image.open(io.BytesIO(c.get('/api/image/sample/labels').data)));assert arr[512,1024]==255
    bad=labels.copy();bad[0,0]=254
    assert c.post('/api/image/sample/save',json={'revision':dataset.state('sample')['revision'],'labels_u8':base64.b64encode(bad.tobytes()).decode()},headers=h).status_code==400
    assert c.get('/api/image/missing').status_code==400

def test_export_refuses_changed_rgb(dataset):
    Image.new('RGB',(2048,1024),'red').save(dataset.items['sample']['rgb'])
    with pytest.raises(ValueError,match='RGB changed'):dataset.export(False)

def test_reject_opposite_face_sam_prompt_without_loading_model():
    sam=SamSegmenter.__new__(SamSegmenter)
    with pytest.raises(ValueError,match='same 100 degree'):
        sam.predict('unused',[[1024,512],[0,512]],[1,0])

def test_export_refuses_label_tampering(dataset):
    f=dataset.folder('sample')/'revisions'/dataset.state('sample')['revision']/'labels.png'
    labels=np.array(Image.open(f));labels[0,0]=1;Image.fromarray(labels).save(f)
    with pytest.raises(ValueError,match='label hash changed'):dataset.export(False)
