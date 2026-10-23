import sys,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
if __name__=='__main__':
    assert sys.argv[1]=='ogr'
    c=json.loads(Path(sys.argv[2]).read_text());xyz=np.fromfile(c['xyz'],dtype='f4').reshape(-1,3)
    # Exclude self. query_ball_point return_length avoids huge neighborhood lists.
    count=cKDTree(xyz).query_ball_point(xyz,c['radius'],return_length=True,workers=2)-1
    (count<c['min_neighbors']).astype('u1').tofile(c['output'])
    Path(c['output']+'.json').write_text(json.dumps({'count':len(xyz),'removed':int((count<c['min_neighbors']).sum()),'neighbors_quantiles':np.quantile(count,[0,.1,.5,.9,1]).tolist(),'radius':c['radius'],'exclude_self':True},indent=2))
