"""Prevent explicit sparse view IDs from silently becoming a uniform subset."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest
from spcgs import pipeline

@pytest.fixture
def cameras(monkeypatch):
    source={'train':[{'name':n,'split':'train'} for n in ['first','unseen','third','last']],
      'test':[{'name':'test','split':'test'}],'val':[{'name':'val','split':'val'}]}
    monkeypatch.setattr(pipeline,'cameras',lambda split:source[split])
    monkeypatch.setattr(pipeline,'ref',lambda _: {'fixture':True})
    return source

def test_sparse_ids_preserved_in_order(tmp_path,cameras):
    cfg={'train_names':['third','first'],'train_views':2,'eval_views':1}
    train,test=pipeline.prepare(tmp_path,cfg)
    assert [v['name'] for v in train]==['third','first']
    manifest=pipeline.read(tmp_path/'data_manifest.json')
    assert {v['name'] for v in manifest['train']}=={'first','third'}
    assert [v['name'] for v in test]==['test']

@pytest.mark.parametrize('names',[['first','first'],['first','test'],['first','missing']])
def test_invalid_or_heldout_training_ids_rejected(tmp_path,cameras,names):
    with pytest.raises(AssertionError,match='Invalid explicit training view IDs'):
        pipeline.prepare(tmp_path,{'train_names':names,'train_views':2,'eval_views':1})
