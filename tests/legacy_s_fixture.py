"""Materialize archived test F/T targets as explicit S without editing old runs."""
import copy
from pathlib import Path
import numpy as np

def with_cached_s(request,folder):
    request=copy.deepcopy(request);folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    text=np.fromfile(request['text'],dtype='f4').reshape(len(request['classes']),512)
    for v in request['views']+request.get('generated_views',[]):
        if 'relevancy' in v:continue
        h,w=v['feature_height'],v['feature_width'];f=np.fromfile(v['feature'],dtype='f2').astype('f4').reshape(512,h,w)
        f/=np.maximum(np.linalg.norm(f,axis=0,keepdims=True),1e-8)
        path=folder/(v['name']+'_S.f16');np.einsum('kd,dhw->khw',text,f,optimize=True).astype('f2').tofile(path)
        v.update(relevancy=str(path),relevancy_height=h,relevancy_width=w,relevancy_classes=len(text),semantic_target='hard_argmax')
    return request
