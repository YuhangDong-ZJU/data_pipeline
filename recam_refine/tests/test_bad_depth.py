import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from recam_refine.bad_depth import BadDepthImage, read_depth, excluded_candidate
from recam_refine.common import read_json, read_jsonl, write_json, values
from recam_refine.quarantine import survivor_order


class BadDepthTests(unittest.TestCase):
    def test_retry_only_failed_decode_and_include_filename(self):
        stream = io.BytesIO()
        Image.fromarray(np.full((8,8),1000,np.uint16)).save(stream,format='PNG')
        with patch.object(Path,'read_bytes',side_effect=[b'broken',stream.getvalue()]) as read:
            self.assertTrue(np.all(read_depth('/tmp/frame.png')==1))
            self.assertEqual(read.call_count,2)
        with patch.object(Path,'read_bytes',return_value=b'broken') as read:
            with self.assertRaisesRegex(BadDepthImage,'frame.png'):
                read_depth('/tmp/frame.png')
            self.assertEqual(read.call_count,2)
        with patch.object(Path,'read_bytes',side_effect=PermissionError('mount')) as read:
            with self.assertRaises(PermissionError):
                read_depth('/tmp/frame.png')
            self.assertEqual(read.call_count,1)

    def test_tail_holes_keep_survivors_unique(self):
        for bad in ({2,5},{8,9},{0,9},{0,1,8,9}):
            order = survivor_order(10,bad)
            self.assertEqual(set(order),set(range(10))-bad)
            self.assertEqual(len(order),10-len(bad))

    def test_bad_jobs_complete_but_other_errors_remain_fatal(self):
        import argparse
        from recam_refine import calibration as c
        from recam_refine.common import RefineError
        with tempfile.TemporaryDirectory() as td:
            work=Path(td)
            job=dict(episode_index=7,source={'source_episode_id':'lab+7'})
            args=argparse.Namespace(devices='cpu',iterations=1)
            bad=dict(episode_index=7,camera=1,error='bad.png',bad_depth=True)
            with patch.object(c,'_run_reference',return_value=[bad]):
                c.run_calibrations(work,{},[(job,work/'7.json')],None,work,args,'reference')
            self.assertTrue(read_json(work/'7.json')['excluded_bad_depth'])
            with patch.object(c,'_run_reference',return_value=[dict(bad,bad_depth=False)]):
                with self.assertRaises(RefineError):
                    c.run_calibrations(work,{},[(job,work/'7.json')],None,work,args,'reference')

    def test_two_shards_resume_exclude_apply_check_cleanup_repack(self):
        import pyarrow.parquet as pq
        from recam_refine.tests.test_shards import fixture,worker,fake_calibration
        from recam_refine.tests.test_steps import geometry_pass
        from recam_refine.shards import refine_shard,merge_shards,load_plan
        from recam_refine.steps import run_step,MARKERS
        from recam_refine import quarantine
        with tempfile.TemporaryDirectory() as td:
            args,droid,_ = fixture(Path(td))
            (droid/'images/chunk-000/observation.images.depth_01/episode_000000/frame_000000.png').write_bytes(b'broken')
            original = read_json(args.work_dir/'plan.json')
            def fitting(root,info,pending,urdf,work,aa,backend):
                for job,path in pending:
                    if job['episode_index']==0:
                        write_json(path,excluded_candidate(job,[dict(episode_index=0,camera=1,bad_depth=True,error='broken.png')]))
                    else:
                        fake_calibration(root,info,[(job,path)],urdf,work,aa,backend)
            for i in range(2):
                with patch('recam_refine.calibration.run_calibrations',fitting):
                    refine_shard(args.root,args.work_dir,worker(args,i))
            with patch('recam_refine.calibration.run_calibrations',side_effect=AssertionError('Repeated GPU work')):
                for i in range(2):
                    refine_shard(args.root,args.work_dir,worker(args,i))
            merge_shards(args.root,args.work_dir,args)
            self.assertEqual([x['episode_index'] for x in read_json(args.work_dir/'excluded_bad_depth.json')],[0])
            args.step='apply'
            move=quarantine.relocate
            calls=[]
            def interrupted(src,dst):
                move(src,dst)
                calls.append(src)
                if len(calls)==1:
                    raise RuntimeError('Interrupted after move before receipt')
            with patch.object(quarantine,'relocate',interrupted):
                with self.assertRaisesRegex(RuntimeError,'Interrupted'):
                    run_step(args)
            run_step(args)
            self.assertEqual(read_json(args.work_dir/'plan.json'),original)
            episodes=read_jsonl(droid/'meta/episodes.jsonl')
            self.assertEqual(len(episodes),1)
            self.assertEqual(episodes[0]['episode_index'],0)
            self.assertEqual(episodes[0]['source_episode_index'],1)
            table=pq.read_table(droid/'data/chunk-000/episode_000000.parquet')
            self.assertTrue(np.all(values(table['episode_index'])==0))
            self.assertTrue(np.array_equal(values(table['index']).ravel(),np.arange(len(table))))
            self.assertFalse((droid/'data/chunk-000/episode_000001.parquet').exists())
            self.assertTrue((args.work_dir/'bad_depth_exclusion/episodes/data/chunk-000/episode_000000.parquet').exists())
            from recam_refine import audit
            import argparse
            audit_args=argparse.Namespace(root=args.root,work_dir=args.work_dir,report_dir=args.work_dir/'test_audit',
                frames=4,image_frames=0,workers=1,device='cpu',fit=False,iterations=1,episodes=['all'],
                episode_manifest=None,pointworld_cameras=args.pointworld_cameras,candidate_dir=None,depth_metadata=[])
            seen=[]
            def audit_worker(task):
                seen.append(task)
                return dict(episode_index=0,source_episode_id='lab+1',summary={},assessment={},evaluation_frames=[],images=[])
            with patch.object(audit,'prepare_assets',return_value=None),patch.object(audit,'_init_audit_worker'), \
                 patch.object(audit,'_audit_worker',audit_worker),patch.object(audit,'write_html'), \
                 patch('recam_refine.audit_plots.comparison_plots'):
                audit._run_audit_locked(audit_args)
            self.assertEqual(len(seen),1)
            self.assertEqual(seen[0][2]['source_episode_id'],'lab+1')
            self.assertEqual(seen[0][3]['episode_index'],0)
            self.assertEqual(seen[0][6],args.work_dir/'bad_depth_exclusion/cameras')
            with patch.object(quarantine,'relocate',side_effect=AssertionError('Repeated move')):
                run_step(args)
            for stage in ('check','cleanup'):
                args.step=stage
                args.episodes_per_shard=250
                with patch('recam_refine.audit._run_audit_locked',geometry_pass):
                    run_step(args)
            from recam_refine.repack import run_repack
            run_repack(args)
            with patch('recam_refine.repack.verify_existing_tar',side_effect=AssertionError('Repeated TAR read')):
                run_repack(args)
            self.assertTrue((args.work_dir/MARKERS['check']).exists())


if __name__=='__main__':
    unittest.main()
