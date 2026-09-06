# SPDX-License-Identifier: Apache-2.0
"""Fixed-shape PointWorld objective and resumable, CUDA-graph Adam batches.

Each camera has independent parameters, moments, visibility and best iterate.
The scalar reference objective remains the authority for final acceptance.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .common import check_transform, require
from .pointworld import depth_loss, BATCHED_BACKEND_VERSION

BACKEND_VERSION = BATCHED_BACKEND_VERSION


def pose_matrices(p):
    x,y,z,r,pitch,yaw = p.unbind(-1)
    cr,sr,cp,sp,cy,sy = r.cos(),r.sin(),pitch.cos(),pitch.sin(),yaw.cos(),yaw.sin()
    zero,one = x*0,x*0+1
    return torch.stack((cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr,x,
                        sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr,y,
                        -sp,cp*sr,cp*cr,z,zero,zero,zero,one),-1).reshape(-1,4,4)


class ProjectedDepth:
    """One fixed-size row per camera/frame; exact first-hit 0.5 px dedup.

    Original point order defines the first hit. Invalid/padded points never
    enter the loss, its denominator, or its gradient. No host scalar reads.
    """
    def __init__(self, points, depths, intrinsics):
        self.points,self.depths,self.k = points,depths,intrinsics
        self.rows,self.n = points.shape[:2]
        self.h,self.w = depths.shape[-2:]
        self.columns = 2*self.w+1
        self.cells = self.columns*(2*self.h+1)
        self.order = torch.arange(self.n,device=points.device,dtype=torch.int32)[None].expand(self.rows,-1)
        self.first = torch.empty((self.rows,self.cells+1),device=points.device,dtype=torch.int32)

    def __call__(self, matrices):
        xyz = torch.bmm(self.points,matrices[:,:3,:3].transpose(1,2))+matrices[:,None,:3,3]
        uvw = torch.bmm(xyz,self.k.transpose(1,2))
        uv = uvw[:,:,:2]/(uvw[:,:,2:]+1e-8)
        inside = (xyz[:,:,2]>0)&torch.isfinite(uv).all(-1)&(uv[:,:,0]>=0)&(uv[:,:,0]<self.w)&(uv[:,:,1]>=0)&(uv[:,:,1]<self.h)
        safe_uv = torch.where(inside[:,:,None],uv,0.)
        q = torch.round(safe_uv*2).long()
        cell = torch.where(inside,q[:,:,1]*self.columns+q[:,:,0],self.cells)
        self.first.fill_(self.n)
        self.first.scatter_reduce_(1,cell,torch.where(inside,self.order,self.n),reduce='amin',include_self=True)
        first_hit = inside&(self.first.gather(1,cell)==self.order)
        grid = torch.stack((2*safe_uv[:,:,0]/(self.w-1)-1,2*safe_uv[:,:,1]/(self.h-1)-1),-1)
        observed = torch.nn.functional.grid_sample(self.depths[:,None],grid[:,None],align_corners=True).reshape(self.rows,self.n)
        valid = first_hit&(observed>=.3)&(observed<=2.)
        count = valid.sum(-1)
        residual = torch.where(valid,(observed-xyz[:,:,2]).abs(),0.)
        return residual.sum(-1)/count.clamp_min(1),count


class CameraBatch:
    """Same-budget independent fits, with state preserved across retry/restart."""
    def __init__(self, inputs, device='cuda:0', min_points=1000, use_graph=True):
        self.inputs,self.device,self.min_points = inputs,torch.device(device),min_points
        self.m = len(inputs)
        require(self.m>0,'Empty camera batch')
        torch.set_num_threads(2)
        if self.device.type=='cuda':
            torch.backends.cuda.matmul.allow_tf32 = False
        self.initial = torch.as_tensor(np.stack([np.linalg.inv(x['initial']) for x in inputs]),dtype=torch.float32,device=self.device)
        self.ks = [torch.as_tensor(x['k'],dtype=torch.float32,device=self.device) for x in inputs]
        self.depths = [[torch.as_tensor(d,dtype=torch.float32,device=self.device) for d in x['depths']] for x in inputs]
        self.points = [[torch.as_tensor(p,dtype=torch.float32,device=self.device) for p in x['points']] for x in inputs]
        self.train,self.holdout,self.initial_train,self.initial_test = [],[],[],[]
        # Scalar reference determines frame selection and the acceptance baseline.
        with torch.no_grad():
            for c in range(self.m):
                visible = [f for f in range(len(self.depths[c])) if self._reference(c,self.initial[c],[f])[1]]
                train,holdout = visible[::2],visible[1::2]
                self.train.append(train if len(visible)>=4 else [])
                self.holdout.append(holdout if len(visible)>=4 else [])
                self.initial_train.append(self._reference(c,self.initial[c],self.train[-1])[0])
                self.initial_test.append(self._reference(c,self.initial[c],self.holdout[-1])[0])
        self.f = max(1,max(map(len,self.train)))
        pp,dd,kk,mask = [],[],[],[]
        for c in range(self.m):
            for f in range(self.f):
                index = self.train[c][f] if f<len(self.train[c]) else 0
                pp.append(self.points[c][index])
                dd.append(self.depths[c][index])
                kk.append(self.ks[c])
                mask.append(f<len(self.train[c]))
        self.frame_mask = torch.tensor(mask,device=self.device).reshape(self.m,self.f)
        self.objective = ProjectedDepth(torch.stack(pp),torch.stack(dd),torch.stack(kk))
        self.param = torch.zeros((self.m,6),device=self.device,requires_grad=True)
        self.scale = torch.tensor([.01]*3+[np.deg2rad(.05)]*3,dtype=torch.float32,device=self.device)
        self.running = self.frame_mask.any(-1)
        self.best = self.initial.clone()
        self.best_loss = torch.tensor(self.initial_train,device=self.device,dtype=torch.float32)
        self.optimizer = torch.optim.Adam([self.param],lr=.05,eps=1e-6,foreach=False,capturable=self.device.type=='cuda')
        self.iteration = 0
        self.graph = None
        # Allocate Adam state and gradient buffers before graph capture.
        snapshot = self.snapshot(include_optimizer=False)
        self._step()
        self.restore(snapshot)
        if use_graph and self.device.type=='cuda':
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._step()
            torch.cuda.current_stream(self.device).wait_stream(stream)
            self.restore(snapshot)
            try:
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph,stream=stream):
                    self._step()
                torch.cuda.current_stream(self.device).wait_stream(stream)
            except torch.cuda.OutOfMemoryError:
                raise
            except RuntimeError as exc:
                if 'captur' not in str(exc).lower():
                    raise
                self.graph = None
                print(f'{self.device}: CUDA graph capture unavailable; using eager batches: {exc}',flush=True)
            self.restore(snapshot)

    def _reference(self,c,matrix,frames):
        if not frames:
            return float('inf'),False
        terms = [depth_loss(self.points[c][f],matrix,self.depths[c][f],self.ks[c]) for f in frames]
        if any(n<self.min_points for _,n in terms):
            return float('inf'),False
        return float(torch.stack([loss for loss,_ in terms]).mean()),True

    def _step(self):
        self.optimizer.zero_grad(set_to_none=False)
        matrices = pose_matrices(self.param*self.scale)@self.initial
        repeated = matrices[:,None].expand(-1,self.f,-1,-1).reshape(-1,4,4)
        losses,counts = self.objective(repeated)
        losses,counts = losses.reshape(self.m,self.f),counts.reshape(self.m,self.f)
        per_camera = torch.where(self.frame_mask,losses,0.).sum(-1)/self.frame_mask.sum(-1).clamp_min(1)
        usable = ((counts>=self.min_points)|~self.frame_mask).all(-1)&torch.isfinite(per_camera)
        with torch.no_grad():
            self.running.logical_and_(usable)
            improved = self.running&(per_camera.detach()<self.best_loss)
            self.best.copy_(torch.where(improved[:,None,None],matrices.detach(),self.best))
            self.best_loss.copy_(torch.where(improved,per_camera.detach(),self.best_loss))
        # Sum cameras, mean frames: another camera never rescales this gradient.
        torch.where(self.running,per_camera,0.).sum().backward()
        self.optimizer.step()
        with torch.no_grad():
            self.param[:,:3].clamp_(-20,20)
            self.param[:,3:].clamp_(-400,400)

    def advance(self, target):
        require(target>=self.iteration,'Cannot rewind optimizer state')
        for _ in range(target-self.iteration):
            if self.graph is None:
                self._step()
            else:
                self.graph.replay()
        if self.device.type=='cuda':
            torch.cuda.synchronize(self.device)
        self.iteration = target

    def snapshot(self,include_optimizer=True):
        state = dict(iteration=self.iteration,param=self.param.detach().cpu().clone(),best=self.best.cpu().clone(),
                     best_loss=self.best_loss.cpu().clone(),running=self.running.cpu().clone())
        if include_optimizer:
            state['optimizer'] = {k:v.detach().cpu().clone() for k,v in self.optimizer.state[self.param].items()}
        return state

    def restore(self,state):
        with torch.no_grad():
            for name in ('param','best','best_loss','running'):
                getattr(self,name).copy_(state[name])
            current = self.optimizer.state[self.param]
            for k,v in current.items():
                if 'optimizer' in state:
                    v.copy_(state['optimizer'][k])
                else:
                    v.zero_()
        self.iteration = state['iteration']

    def finish(self):
        results = []
        running = self.running.cpu().tolist()
        with torch.no_grad():
            for c,x in enumerate(self.inputs):
                train,train_valid = self._reference(c,self.best[c],self.train[c])
                test,test_valid = self._reference(c,self.best[c],self.holdout[c])
                result = np.linalg.inv(self.best[c].cpu().numpy())
                shift = float(np.linalg.norm(result[:3,3]-x['initial'][:3,3]))
                angle = float(np.rad2deg(Rotation.from_matrix(result[:3,:3]@x['initial'][:3,:3].T).magnitude()))
                accepted = bool(train_valid and test_valid and test<.1 and test<=self.initial_test[c]+1e-4
                                and train<=self.initial_train[c] and shift<=.30 and angle<=25)
                metric = dict(initial_train_loss=self.initial_train[c],final_train_loss=train,
                    initial_holdout_loss=self.initial_test[c],final_holdout_loss=test,
                    translation_change_m=shift,rotation_change_deg=angle,train_frames=self.train[c],
                    holdout_frames=self.holdout[c],accepted=accepted,iterations=self.iteration,
                    backend=BACKEND_VERSION,cuda_graph=self.graph is not None,can_continue=bool(running[c]))
                # Strict JSON reports: unavailable quantities are null, never Infinity.
                metric = {k:(None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in metric.items()}
                results.append((check_transform(result,'batch candidate'),metric))
        return results
