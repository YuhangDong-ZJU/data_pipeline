from __future__ import annotations

import unittest
import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None,'GPU runtime installs Torch')
class BatchedTests(unittest.TestCase):
    def test_loss_counts_and_gradients_match_reference(self):
        from recam_refine.batched import ProjectedDepth,pose_matrices
        from recam_refine.pointworld import depth_loss,pose_matrix
        rng = np.random.default_rng(42)
        points = rng.uniform([-.8,-.6,.1],[.8,.6,2.5],(3,900,3)).astype(np.float32)
        points[:,10:20] = points[:,:10]  # duplicate pixels must retain first hit
        points[:,20:30,2] = -.5
        depths = rng.uniform(.1,2.2,(3,48,64)).astype(np.float32)
        k = np.tile([[50.,0,32],[0,50.,24],[0,0,1]],(3,1,1)).astype(np.float32)
        for device in (['cpu','cuda:0'] if torch.cuda.is_available() else ['cpu']):
            pp,dd,kk = [torch.tensor(x,device=device) for x in (points,depths,k)]
            p = torch.tensor([[.003,-.001,.01,.002,-.001,.003]]*3,device=device,requires_grad=True)
            loss,count = ProjectedDepth(pp,dd,kk)(pose_matrices(p))
            grad = torch.autograd.grad(loss.sum(),p)[0]
            for i in range(3):
                q = p[i].detach().clone().requires_grad_()
                ref,n = depth_loss(pp[i],pose_matrix(q),dd[i],kk[i])
                g = torch.autograd.grad(ref,q)[0]
                self.assertEqual(int(count[i]),n)
                self.assertAlmostEqual(float(loss[i]),float(ref),places=6)
                torch.testing.assert_close(grad[i],g,rtol=3e-4,atol=3e-5)

    def test_resume_reproduces_continuous_fit_and_batch_independence(self):
        from recam_refine.batched import CameraBatch
        xy = np.stack(np.meshgrid(np.linspace(-.3,.3,20),np.linspace(-.2,.2,20)),-1).reshape(-1,2)
        points = np.column_stack([xy,np.ones(len(xy))]).astype(np.float32)
        initial = np.eye(4,dtype=np.float32)
        initial[2,3] = .04
        data = dict(initial=initial,k=np.array([[50,0,32],[0,50,24],[0,0,1]],np.float32),
                    depths=[np.ones((48,64),np.float32)]*6,points=[points]*6)
        for device in (['cpu','cuda:0'] if torch.cuda.is_available() else ['cpu']):
            a = CameraBatch([data],device,min_points=10)
            a.advance(40)
            saved = a.snapshot()
            a.advance(160)
            b = CameraBatch([data],device,min_points=10)
            b.restore(saved)
            b.advance(160)
            torch.testing.assert_close(a.param,b.param,rtol=0,atol=0)
            aa,bb = a.finish()[0],b.finish()[0]
            np.testing.assert_array_equal(aa[0],bb[0])
            self.assertTrue(aa[1]['accepted'])
            self.assertLess(aa[1]['final_holdout_loss'],.002)
            c = CameraBatch([data,data],device,min_points=10)
            c.advance(160)
            for result in c.finish():
                np.testing.assert_allclose(aa[0],result[0],rtol=1e-5,atol=1e-5)
                self.assertTrue(result[1]['accepted'])

    def test_unobservable_camera_does_not_poison_other_camera(self):
        from recam_refine.batched import CameraBatch
        rng = np.random.default_rng(42)
        p = rng.uniform([-.2,-.2,1.],[.2,.2,1.],(100,3)).astype(np.float32)
        valid = dict(initial=np.eye(4),k=np.array([[50,0,32],[0,50,24],[0,0,1]]),
                     depths=[np.ones((48,64),np.float32)]*4,points=[p]*4)
        invalid = {**valid,'depths':[np.zeros((48,64),np.float32)]*4}
        fit = CameraBatch([valid,invalid],'cpu',min_points=10)
        fit.advance(4)
        results = fit.finish()
        self.assertTrue(results[0][1]['accepted'])
        self.assertFalse(results[1][1]['accepted'])
        self.assertFalse(results[1][1]['can_continue'])


if __name__=='__main__':
    unittest.main()
