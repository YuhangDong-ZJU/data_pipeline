"""Multi-worker orchestration on disposable shared data; fitting is isolated."""
from __future__ import annotations

import copy
import errno
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from recam_refine.common import RefineError, read_json, sha256, write_json, require
from recam_refine.pointworld import POINTWORLD_COMMIT, URDF_RELATIVE
from recam_refine.shards import (PLAN, READY, load_plan, merge_shards, part_jobs, plan_shards,
                                 refine_shard, sampled_frames, worker_locks, input_hashes)
from recam_refine.steps import MARKERS, locked_step, run_step, transfer_only
from recam_refine.tests.test_steps import all_files, manual_fixture


def fake_assets(root):
    path = root/URDF_RELATIVE
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('<robot name="fixture"/>')
    write_json(root/'assets.sha256.json',{URDF_RELATIVE:sha256(path)})
    return path


def fake_calibration(root,info,pending,urdf,work,args,backend):
    # Test orchestration only. Real loss/gradient/optimizer tests live separately.
    for job,path in pending:
        if job.get('calibration_inputs'):
            require(input_hashes(root,info,job)==job['calibration_inputs'],'Changed fixture input')
        write_json(path,dict(episode_index=job['episode_index'],source_episode_id=job['source']['source_episode_id'],
            source='pointworld_method_droid_initialization',pointworld_commit=POINTWORLD_COMMIT,
            camera_to_base=np.asarray(job['initial_extrinsics'])[1:].tolist(),sample_frame_indices=sampled_frames(job),
            metrics=[dict(accepted=False,status='official_initial_retained',reason='synthetic unobservable robot')]*2))


def fixture(tmp,pending=(0,1),num_shards=2,devices='cpu'):
    args,droid,archives = manual_fixture(tmp)
    for i in pending:
        (args.pointworld_cameras/f'lab+{i}_cameras.json').unlink()
    transfer_only(args)
    for stage in ('unpack','align','overlap'):
        args.step = stage
        run_step(args)
    args.num_shards,args.shard_id = num_shards,0
    args.devices = devices
    args.worker_work_dir = tmp/'worker0'
    args.refine_backend,args.gpu_batch_size,args.no_cuda_graphs = 'auto',0,False
    with patch('recam_refine.shards.prepare_assets',fake_assets), patch('recam_refine.shards.input_hashes', side_effect=AssertionError('New plan must not pre-read depth')):
        plan_shards(args.root,args.work_dir,args)
    return args,droid,archives


def worker(args,shard_id):
    result = copy.deepcopy(args)
    result.shard_id = shard_id
    result.worker_work_dir = args.work_dir.parent/f'worker{shard_id}'
    return result


class ShardTests(unittest.TestCase):
    def test_network_directory_fallback_keeps_contention_errors(self):
        import fcntl
        from recam_refine.common import acquire_directory_lock, validate_lock_mount
        with tempfile.TemporaryDirectory() as td:
            fd = os.open(td,os.O_RDONLY|os.O_DIRECTORY)
            try:
                with patch('recam_refine.common.validate_lock_mount',return_value='nfs4'), \
                     patch.object(fcntl,'flock',side_effect=OSError(errno.EBADF,'NFS directory is read only')):
                    acquire_directory_lock(fd,fcntl.LOCK_EX)
                with patch('recam_refine.common.validate_lock_mount',return_value='nfs4'), \
                     patch.object(fcntl,'flock',side_effect=BlockingIOError(errno.EAGAIN,'busy')),self.assertRaises(BlockingIOError):
                    acquire_directory_lock(fd,fcntl.LOCK_EX)
                with patch('recam_refine.common.lock_mount',return_value=('nfs4',{'local_lock=all'})),self.assertRaises(RefineError):
                    validate_lock_mount(Path(td))
            finally:
                os.close(fd)

    def test_disjoint_shards_resume_merge_and_apply_boundary(self):
        with tempfile.TemporaryDirectory() as td:
            args,droid,_ = fixture(Path(td))
            before = all_files(droid)
            plan = load_plan(args.root,args.work_dir)
            ids = [i for p in plan['shards'] for i in p['episodes']]
            self.assertEqual(sorted(ids),[0,1])
            self.assertEqual(len(set(ids)),len(ids))
            self.assertFalse((args.work_dir/MARKERS['refine']).exists())
            with self.assertRaises(RefineError):
                merge_shards(args.root,args.work_dir,args)
            for i in range(2):
                with patch('recam_refine.calibration.run_calibrations',fake_calibration):
                    refine_shard(args.root,args.work_dir,worker(args,i))
                self.assertFalse((args.work_dir/MARKERS['refine']).exists())
            with patch('recam_refine.calibration.run_calibrations',side_effect=AssertionError('must not recompute')):
                refine_shard(args.root,args.work_dir,worker(args,0))
            self.assertEqual(all_files(droid),before)
            merge_shards(args.root,args.work_dir,args)
            receipt = read_json(args.work_dir/MARKERS['refine'])
            self.assertEqual(len(receipt['candidate_sha256']),2)
            self.assertEqual(receipt['retained_official_cameras'],4)
            self.assertEqual(all_files(droid),before)
            merge_shards(args.root,args.work_dir,args)
            args.step = 'apply'
            run_step(args)
            self.assertTrue((args.work_dir/MARKERS['apply']).exists())
            args.step = 'cleanup'
            with self.assertRaises(RefineError):
                run_step(args)

    def test_readers_coexist_duplicate_shard_and_writers_are_excluded(self):
        with tempfile.TemporaryDirectory() as td:
            args,_,_ = fixture(Path(td))
            with worker_locks(args.root,args.work_dir,Path(td)/'worker0',0):
                with worker_locks(args.root,args.work_dir,Path(td)/'worker1',1):
                    with self.assertRaises(RuntimeError):
                        with locked_step(args.root,args.work_dir):
                            pass
                    with self.assertRaises(RuntimeError):
                        with worker_locks(args.root,args.work_dir,Path(td)/'duplicate',0):
                            pass

    def test_changed_input_wrong_plan_and_extra_results_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            args,droid,_ = fixture(Path(td))
            plan = load_plan(args.root,args.work_dir)
            for i in (0,1):
                with patch('recam_refine.calibration.run_calibrations',fake_calibration):
                    refine_shard(args.root,args.work_dir,worker(args,i))
            complete = args.work_dir/'shards/results/shard-00000/COMPLETE.json'
            original_receipt = read_json(complete)
            write_json(complete,{**original_receipt,'plan_id':'other-plan'})
            with self.assertRaises(RefineError):
                merge_shards(args.root,args.work_dir,args)
            write_json(complete,original_receipt)
            extra = complete.parent/'cameras/episode_999999.json'
            write_json(extra,{})
            with self.assertRaises(RefineError):
                merge_shards(args.root,args.work_dir,args)
            self.assertFalse((args.work_dir/MARKERS['refine']).exists())
            self.assertFalse(list((args.work_dir/'cameras').glob('*.json')))

    def test_changed_plan_and_single_mode_mixing_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            args,_,_ = fixture(Path(td))
            ready_hash = sha256(args.work_dir/READY)
            plan_shards(args.root,args.work_dir,args)
            self.assertEqual(sha256(args.work_dir/READY),ready_hash)
            args.num_shards = 3
            with self.assertRaises(RefineError):
                plan_shards(args.root,args.work_dir,args)
            args.step = 'refine'
            with self.assertRaises(RefineError):
                run_step(args)
            value = read_json(args.work_dir/PLAN)
            value['pending_episodes'] += 1
            write_json(args.work_dir/PLAN,value)
            with self.assertRaises(RefineError):
                load_plan(args.root,args.work_dir)

    def test_reference_worker_checks_planned_input_bytes_before_fitting(self):
        from recam_refine.pipeline import calibrate_one
        with tempfile.TemporaryDirectory() as td:
            args,droid,_ = fixture(Path(td))
            plan = load_plan(args.root,args.work_dir)
            job = part_jobs(args.work_dir,plan['shards'][0])[0]
            job['calibration_inputs'] = dict(parquet_sha256='wrong')
            with self.assertRaisesRegex(RefineError,'Shard Parquet changed'):
                calibrate_one((droid,read_json(args.work_dir/'training_info.json'),job,Path(td)/'candidate.json',
                               args.work_dir/'pointworld'/URDF_RELATIVE,'cpu',1))

    def test_partial_publication_and_merge_recover_without_recomputing(self):
        with tempfile.TemporaryDirectory() as td:
            args,droid,_ = fixture(Path(td),pending=(1,))
            before = all_files(droid)
            for i in (0,1):
                with patch('recam_refine.calibration.run_calibrations',fake_calibration):
                    refine_shard(args.root,args.work_dir,worker(args,i))
            plan = load_plan(args.root,args.work_dir)
            part = next(p for p in plan['shards'] if p['episodes'])
            shared = args.work_dir/f'shards/results/shard-{part["shard_id"]:05d}'
            (shared/'COMPLETE.json').unlink()
            replacement = worker(args,part['shard_id'])
            replacement.worker_work_dir = Path(td)/'replacement-machine'
            with patch('recam_refine.calibration.run_calibrations',side_effect=AssertionError('published candidate must be reused')):
                refine_shard(args.root,args.work_dir,replacement)
            # Simulate interruption after the first coordinator candidate write.
            from recam_refine import shards
            real_write = shards.atomic_bytes
            count = 0
            def interrupted(path,data):
                nonlocal count
                real_write(path,data)
                count += 1
                if count==1:
                    raise RuntimeError('simulated process interruption')
            with patch.object(shards,'atomic_bytes',interrupted),self.assertRaises(RuntimeError):
                merge_shards(args.root,args.work_dir,args)
            self.assertFalse((args.work_dir/MARKERS['refine']).exists())
            merge_shards(args.root,args.work_dir,args)
            self.assertEqual(all_files(droid),before)
            self.assertEqual(read_json(args.work_dir/MARKERS['refine'])['episodes'],2)


if __name__=='__main__':
    unittest.main()
