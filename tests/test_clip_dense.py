"""Numerical contracts: true patch accumulation, sphere orientation and S."""
import random,unittest
import numpy as np
from spcgs.clip_dense import patch_boxes,accumulate_rectangles,inverse_maps,erp_rays,FACES,unit,spatial_relevancy,perspective

class DenseContracts(unittest.TestCase):
    def test_rectangles_equal_reference_pixel_add(self):
        rng=random.Random(17)
        for p in [12,9,6]:
            boxes=patch_boxes(64,64,p,rng);e=np.random.default_rng(4).normal(size=(len(boxes),8)).astype('f4')
            actual,count=accumulate_rectangles(e,boxes,64,64);expected=np.zeros((8,64,64));coverage=np.zeros((64,64))
            for vector,(l,u,r,b) in zip(e,boxes):expected[:,u:b,l:r]+=vector[:,None,None];coverage[u:b,l:r]+=1
            np.testing.assert_array_equal(count,coverage);np.testing.assert_allclose(actual,expected/coverage,rtol=1e-6,atol=1e-6)
            self.assertGreater(count.min(),0)
    def test_sphere_round_trip_seam_and_poles(self):
        # Use a direction-coded face field: any orientation/flip/half-pixel bug
        # makes reconstructed ERP directions differ from analytic ray values.
        import torch
        import torch.nn.functional as f
        h,w,size=64,128,128;grids,weights=inverse_maps(h,w,100);t=np.tan(np.deg2rad(50));coords=((np.arange(size)+.5)/size*2-1)*t
        x,y=np.meshgrid(coords,coords);rays=np.stack([x,y,np.ones_like(x)],-1)
        face=np.array([unit(rays@R.T,-1).transpose(2,0,1) for _,R in FACES],dtype='f4')
        mapped=f.grid_sample(torch.from_numpy(face),torch.from_numpy(grids),align_corners=False,padding_mode='border').numpy()
        erp=unit((mapped*weights[:,None]).sum(0)/weights.sum(0),0).transpose(1,2,0)
        # At the last half-pixel of a face, border sampling clamps by <=0.5/128.
        # This bound is below 0.1 degree and well below a feature-grid pixel.
        np.testing.assert_allclose(erp,erp_rays(h,w),atol=1.1e-3)
        self.assertLess(np.linalg.norm(erp[:,0]-erp[:,-1],axis=-1).max(),.05)
        self.assertGreater(erp[-1,:,1].min(),.999);self.assertLess(erp[0,:,1].max(),-.999)
    def test_spatial_minmax_not_class_softmax(self):
        x=np.array([[[[.2,.4],[.6,.8]],[[.3,.3],[.3,.3]]]],dtype='f4')
        y=spatial_relevancy(x);np.testing.assert_allclose(y[0,0],[[0,1/3],[2/3,1]],atol=1e-6);self.assertTrue((y[0,1]==0).all())
    def test_actual_forward_and_inverse_projection(self):
        import torch
        import torch.nn.functional as f
        # Float direction channels traverse the actual RGB crop implementation,
        # including periodic horizontal padding and vertical clamping.
        original=erp_rays(512,1024).astype('f4')
        faces=np.array([perspective(original,R,256,100).transpose(2,0,1) for _,R in FACES])
        grids,weights=inverse_maps(64,128,100)
        mapped=f.grid_sample(torch.from_numpy(faces),torch.from_numpy(grids),align_corners=False,padding_mode='border').numpy()
        actual=unit((mapped*weights[:,None]).sum(0)/weights.sum(0),0).transpose(1,2,0)
        np.testing.assert_allclose(actual,erp_rays(64,128),atol=6e-4)
    def test_patch_schedule_deterministic(self):
        a=patch_boxes(128,128,25,random.Random(2));b=patch_boxes(128,128,25,random.Random(2))
        np.testing.assert_array_equal(a,b);self.assertEqual(len(a),361)

if __name__=='__main__':unittest.main()
