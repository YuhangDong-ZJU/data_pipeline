"""Scheduling, bounded recovery and input identity tests without real writes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from recam_refine import calibration as c
from recam_refine.common import RefineError, read_json

try:
    import torch
except ImportError:
    torch = None


class SchedulerTests(unittest.TestCase):
    def test_backend_resume_preserves_legacy_protocol(self):
        args = argparse.Namespace(devices='0,1',refine_backend='auto')
        self.assertEqual(c.select_backend(args), 'batched')
        self.assertEqual(c.select_backend(args,{'iterations':2000}), 'reference')
        self.assertEqual(c.select_backend(args,{'backend':'batched','backend_version':c.BATCHED_BACKEND_VERSION}), 'batched')
        args.refine_backend = 'batched'
        with self.assertRaises(RefineError):
            c.select_backend(args,{'iterations':2000})
        args.devices = 'cpu'
        with self.assertRaises(RefineError):
            c.select_backend(args)

    def test_dynamic_retry_prefetch_and_complete_camera_pairs(self):
        local = threading.local()
        calls = []
        def fake_process(tasks,next_tasks):
            tag = [c.task_key(t) for t in tasks]
            if getattr(local,'reserved',None):
                self.assertEqual(local.reserved,tag)
            local.reserved = [c.task_key(t) for t in next_tasks]
            calls.extend(tag)
            time.sleep(.003 if tasks[0]['job']['episode_index']%2 else .02)
            results = []
            for t in tasks:
                retry = t['cam'] == 2 and t['iteration'] == 0
                results.append(dict(task=t,retry=retry,pose=np.eye(4).tolist(),metric={'accepted':not retry},frames=[0,1]))
            return dict(results=results,device='fake',elapsed_seconds=.01,cameras=len(tasks),retry_cameras=sum(r['retry'] for r in results))
        def pool(**kwargs):
            return ThreadPoolExecutor(max_workers=1)
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            pending = [(dict(episode_index=i,source={'source_episode_id':str(i)}),work/f'cameras/episode_{i:06d}.json') for i in range(11)]
            args = argparse.Namespace(iterations=2000,gpu_batch_size=3)
            with patch.object(c,'ProcessPoolExecutor',pool),patch.object(c,'load_state',return_value=None), \
                 patch.object(c,'worker_capacity',return_value=dict(device='fake',batch_size=3,free_vram_gib=7)), \
                 patch.object(c,'process_batch',fake_process):
                errors = c._run_batched(work,{},pending,work/'fake.urdf',work,args,['cuda:0','cuda:1'])
            self.assertEqual(errors,[])
            self.assertEqual(len(calls),33)
            self.assertEqual(len(set(calls)),33)
            for job,path in pending:
                result = read_json(path)
                self.assertEqual(result['source_episode_id'],str(job['episode_index']))
                self.assertTrue(all(m['accepted'] for m in result['metrics']))
            self.assertEqual(read_json(work/'calibration_timing.json')['episodes'],11)

    @unittest.skipIf(torch is None,'Torch runtime required')
    def test_atomic_state_roundtrip_and_regrouping(self):
        from recam_refine.batched import CameraBatch
        rng = np.random.default_rng(42)
        points = rng.uniform([-.2,-.2,1],[.2,.2,1],(100,3)).astype(np.float32)
        data = dict(initial=np.eye(4),k=np.array([[50,0,32],[0,50,24],[0,0,1]]),
                    points=[points]*4,depths=[np.ones((48,64),np.float32)]*4)
        fit = CameraBatch([data,data],'cpu',min_points=10)
        fit.advance(5)
        state = fit.snapshot()
        split = [c.split_state(state,i) for i in (1,0)]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'checkpoint.pt'
            c.save_state(path,dict(fingerprint='test',state=split[0],attempts=[]))
            restored = c.load_state(path)
            self.assertEqual(restored['fingerprint'],'test')
            joined = c.join_states([restored['state'],split[1]])
            fit.restore(joined)
            fit.advance(10)
            self.assertTrue(all(m['accepted'] for _,m in fit.finish()))
            split[1]['iteration'] += 1
            with self.assertRaises(RefineError):
                c.join_states(split)

    @unittest.skipIf(torch is None,'Torch runtime required')
    def test_oom_splits_without_losing_cameras(self):
        def fake_fit(items):
            if len(items)>1:
                raise torch.cuda.OutOfMemoryError('synthetic VRAM pressure')
            return [dict(task=items[0]['task'],retry=False)]
        items = [dict(task=dict(job={'episode_index':i},cam=1,iteration=0),saved=None) for i in range(5)]
        with tempfile.TemporaryDirectory() as td,patch.object(c,'_worker',dict(device='fake',work=td)), \
             patch.object(c,'_fit_group',fake_fit),patch.object(c,'load_state',return_value=None):
            results = c.fit_with_backoff(items)
            self.assertEqual([r['task']['job']['episode_index'] for r in results],list(range(5)))

    @unittest.skipIf(torch is None,'Torch runtime required')
    def test_boundary_checkpoint_is_evaluated_before_retry(self):
        advanced = []
        state = dict(iteration=2000,param=torch.ones((1,6)),best=torch.eye(4)[None],
                     best_loss=torch.ones(1),running=torch.ones(1,dtype=torch.bool),
                     optimizer={'step':torch.tensor(2000.),'exp_avg':torch.ones((1,6)),
                                'exp_avg_sq':torch.ones((1,6))})
        class FakeBatch:
            def __init__(self,*args,**kwargs):
                self.iteration = 0
            def restore(self,saved):
                self.iteration = saved['iteration']
                torch.testing.assert_close(saved['optimizer']['exp_avg'],torch.ones((1,6)))
            def advance(self,target):
                advanced.append((self.iteration,target))
                self.iteration = target
            def snapshot(self):
                return {**state,'iteration':self.iteration}
            def finish(self):
                return [(np.eye(4),dict(accepted=self.iteration==6000,can_continue=True))]
        with tempfile.TemporaryDirectory() as td:
            item = dict(task=dict(job={'episode_index':0},cam=1,iteration=2000),
                        data={'initial':np.eye(4)},frames=[0,1],fingerprint='fixed',
                        saved=dict(state=state,attempts=[]))
            worker = dict(device='cpu',work=Path(td),iterations=2000,graphs=False)
            with patch.object(c,'_worker',worker),patch('recam_refine.batched.CameraBatch',FakeBatch):
                result = c._fit_group([item])[0]
                self.assertTrue(result['retry'])
                self.assertEqual(advanced,[])
                item['saved'] = c.load_state(c.state_path(td,item['task']))
                result = c._fit_group([item])[0]
                self.assertFalse(result['retry'])
                self.assertEqual(advanced,[(2000,3000),(3000,4000),(4000,5000),(5000,6000)])

    @unittest.skipIf(torch is None,'Torch runtime required')
    def test_changed_checkpoint_input_is_rejected(self):
        from recam_refine.tests.test_refine import fixture
        from recam_refine.common import media_path
        from PIL import Image
        class FakeRobot:
            def points(self,*args):
                return np.ones((10,3),np.float32)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)/'droid'
            info = fixture(root)
            # Replace only disposable fixture PNGs with tiny deterministic inputs.
            for t in range(4):
                path = media_path(root,info,0,'observation.images.depth_01',t)
                path.parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray(np.full((8,8),1000,np.uint16)).save(path)
            task = dict(cam=1,iteration=0,job=dict(episode_index=0,length=4,source={'source_episode_id':'fixture'},
                        initial_intrinsics=np.tile(np.eye(3),(3,1,1)).tolist(),
                        initial_extrinsics=np.tile(np.eye(4),(3,1,1)).tolist(),depths={'1':{'record':None}}))
            worker = dict(root=root,info=info,robot=FakeRobot(),work=Path(td)/'work',iterations=2000)
            with patch.object(c,'_worker',worker):
                prepared = c.prepare_tasks([task])[0]
                self.assertNotIn('error',prepared)
                c.save_state(c.state_path(worker['work'],task),dict(fingerprint=prepared['fingerprint'],state={'iteration':0}))
                self.assertNotIn('error',c.prepare_tasks([task])[0])
                c.save_state(c.state_path(worker['work'],task),dict(fingerprint='legacy-payload-hash',state={'iteration':0}))
                self.assertNotIn('error',c.prepare_tasks([task])[0])
                c.save_state(c.state_path(worker['work'],task),dict(fingerprint=prepared['fingerprint'],state={'iteration':0}))
                task['job']['initial_extrinsics'][1][0][3] = .001
                self.assertIn('Checkpoint inputs changed',c.prepare_tasks([task])[0]['error'])


if __name__ == '__main__':
    unittest.main()
