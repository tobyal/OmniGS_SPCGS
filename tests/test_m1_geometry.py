"""Numerical unit checks, explicitly not Wan360 experiment results."""
import unittest
import numpy as np
from spcgs.m1_geometry import bearings,project_erp,angular_pixels,triangulate_track

class SphericalGeometryTests(unittest.TestCase):
    def test_seam_and_poles(self):
        uv=np.array([[.01,512],[2047.99,512],[1024,.01],[1024,1023.99],[117,291]])
        b=bearings(uv,2048,1024)
        camera={'rotation':np.eye(3).tolist(),'center':[0,0,0]}
        projected=project_erp(b,camera,2048,1024)
        dx=(projected[:,0]-uv[:,0]+1024)%2048-1024
        self.assertLess(np.max(np.abs(dx)),1e-7)
        self.assertLess(np.max(np.abs(projected[:,1]-uv[:,1])),1e-5)
        self.assertLess(angular_pixels(b[:1],b[1:2])[0],.021)

    def test_multiview_point_and_zero_baseline(self):
        cameras=[{'rotation':np.eye(3).tolist(),'center':c} for c in [[0,0,0],[.3,0,0],[0,0,.2]]]
        point=np.array([[.1,-.1,2.]])
        uv=[project_erp(point,c,2048,1024)[0] for c in cameras]
        recovered,metrics=triangulate_track(cameras,uv,[(2048,1024)]*3,{'triangulation_error_px':3,'min_parallax_deg':.5})
        self.assertIsNone(metrics['reason'])
        np.testing.assert_allclose(recovered,point[0],atol=1e-7)
        self.assertGreater(metrics['parallax_deg'],1)
        same=[cameras[0]]*3
        recovered,metrics=triangulate_track(same,[uv[0]]*3,[(2048,1024)]*3,{'triangulation_error_px':3,'min_parallax_deg':.5})
        self.assertIsNone(recovered)
        self.assertEqual(metrics['reason'],'degenerate rays')

if __name__=='__main__':unittest.main()
