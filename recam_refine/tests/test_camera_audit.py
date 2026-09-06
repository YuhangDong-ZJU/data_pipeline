from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from recam_refine.audit import assess_comparison, evaluation_frames, run_audit
from recam_refine.common import RefineError, read_json, sha256, write_json, write_jsonl
from recam_refine.geometry_metrics import (aggregate_depth, aggregate_matches, cloud_matches,
                                           robot_mask, scene_cloud)
from recam_refine.pointworld import CalibrationRejected, refine_camera_with_retry


class AuditGeometryTests(unittest.TestCase):
    def test_failed_fit_retry_keeps_threshold_and_observability_failures(self):
        pose = np.eye(4)
        failed = CalibrationRejected('holdout .12 > .10', candidate=pose, metrics={'accepted':False})
        with patch('recam_refine.pointworld.refine_camera', side_effect=[failed, (pose, {'accepted':True})]) as fit:
            _, metrics = refine_camera_with_retry(pose, None, None, None)
            self.assertEqual([call.args[-1] for call in fit.call_args_list],[2000,6000])
            self.assertEqual(len(metrics['attempts']),2)
        with patch('recam_refine.pointworld.refine_camera', side_effect=CalibrationRejected('not visible')) as fit:
            with self.assertRaises(CalibrationRejected):
                refine_camera_with_retry(pose,None,None,None)
            self.assertEqual(fit.call_count,1)

    def test_small_depth_loss_cannot_hide_bad_f1(self):
        def metric(loss, f1):
            return dict(depth=dict(point_weighted_m=loss, frame_mean_m=loss),
                        two_view=dict(f1_5mm=f1, f1_20mm=f1),
                        fixed_mask_two_view=dict(f1_5mm=f1, f1_20mm=f1))
        result = assess_comparison(dict(droid_initial=metric(.07,.8),recam_candidate=metric(.06,.3)))
        self.assertTrue(result['recam_candidate']['depth_under_release_0_10'])
        self.assertEqual(result['recam_candidate']['status'],'needs_review')
        self.assertIn('two_view.f1_5mm',result['recam_candidate']['regressions'])

    def test_f1_thresholds_and_count_weighting(self):
        a = np.array([[0., 0., 1.], [.1, 0., 1.]])
        b = a + [0., 0., .007]
        result = aggregate_matches([cloud_matches(a, b)])
        self.assertEqual(result["f1_5mm"], 0.)
        self.assertEqual(result["f1_20mm"], 1.)
        # A long bad frame must not get the same weight as one matching point.
        rows = [dict(points_a=1, points_b=1, hits_a=[1, 1], hits_b=[1, 1]),
                dict(points_a=9, points_b=9, hits_a=[0, 0], hits_b=[0, 0])]
        self.assertAlmostEqual(aggregate_matches(rows)["f1_5mm"], .1)
        self.assertFalse(aggregate_matches([cloud_matches(a, a[:0])])["available"])

    def test_depth_paper_and_release_aggregations_are_distinct(self):
        result = aggregate_depth([dict(loss_m=.1, robot_points=100), dict(loss_m=.02, robot_points=900)])
        self.assertAlmostEqual(result["frame_mean_m"], .06)
        self.assertAlmostEqual(result["point_weighted_m"], .028)
        self.assertIsNone(aggregate_depth([])["point_weighted_m"])

    def test_mask_lifting_and_known_pose(self):
        vertices = np.array([[-.1,-.1,1.],[.1,-.1,1.],[.1,.1,1.],[-.1,.1,1.]])
        faces = np.array([[0,1,2],[0,2,3]])
        k = np.array([[100.,0.,50.],[0.,100.,50.],[0.,0.,1.]])
        pose = np.eye(4)
        mask = robot_mask(vertices, faces, pose, k, (100,100))
        self.assertTrue(mask[50,50])
        self.assertFalse(mask[20,20])
        points, pixels = scene_cloud(np.ones((100,100)), k, pose, mask,
                                    bounds=(np.array([-1,-1,-1]), np.array([1,1,2])))
        self.assertEqual(len(points), 10000 - int(mask.sum()))
        np.testing.assert_allclose(points[:, :2], (pixels - 50) / 100.)
        vertices[0,2] = -1
        clipped = robot_mask(vertices, faces, pose, k, (100,100))
        self.assertEqual(clipped.shape, (100,100))
        self.assertTrue(clipped.any())
        self.assertFalse(clipped.all())

    def test_evaluation_excludes_fit_and_selection_frames(self):
        excluded = np.linspace(0,99,16,dtype=int).tolist()
        frames = evaluation_frames(100, excluded, 24)
        self.assertEqual(len(frames), 24)
        self.assertFalse(set(frames) & set(excluded))
        self.assertEqual(frames, evaluation_frames(100, excluded, 24))
        with self.assertRaises(RefineError):
            evaluation_frames(6, [0,1,2], 24)

    def test_read_only_audit_generates_real_artifacts(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("GPU lock installs PyTorch; CPU-only CI tests the metric primitives")
        from test_refine import fixture
        class SmallRobot:
            def __init__(self, *args, **kwargs):
                pass
            def geometry(self, *args):
                return (np.array([[-.1,-.1,1.],[.1,-.1,1.],[.1,.1,1.],[-.1,.1,1.]]),
                        np.array([[0,1,2],[0,2,3]]))
            def points(self, *args):
                xy = np.stack(np.meshgrid(np.linspace(-.1,.1,40),np.linspace(-.1,.1,40)),-1).reshape(-1,2)
                return np.column_stack([xy,np.ones(len(xy))]).astype(np.float32)
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            root, work, out = temp/'droid', temp/'work', temp/'report'
            fixture(root)
            manifest = temp/'manifest.jsonl'
            write_jsonl(manifest,[dict(episode_index=0,source_episode_id='audit+0',camera_serials={'external_1':'a','external_2':'b'})])
            camera_dir = temp/'published'
            write_json(camera_dir/'audit+0_cameras.json',dict(uuid='audit+0',optimization_success=True,
                optimization_summary={'final_loss':.05},a={'optimized_extrinsics':np.eye(4).tolist()},
                b={'optimized_extrinsics':np.eye(4).tolist()}))
            before = {str(p):sha256(p) for p in root.rglob('*') if p.is_file()}
            args = argparse.Namespace(root=root,work_dir=work,report_dir=out,episode_manifest=manifest,
                pointworld_cameras=camera_dir,candidate_dir=None,depth_metadata=[],episodes=['0'],
                frames=4,iterations=1,fit=False,device='cpu')
            with patch('recam_refine.audit.Robot', SmallRobot), patch('recam_refine.audit.prepare_assets', return_value=None):
                run_audit(args)
            self.assertEqual(before,{str(p):sha256(p) for p in root.rglob('*') if p.is_file()})
            result = read_json(out/'episode_000000/metrics.json')
            self.assertTrue(result['inputs_unchanged'])
            self.assertTrue((out/'index.html').exists())
            self.assertTrue(all((out/'episode_000000'/name).exists() for name in result['images']))
            self.assertEqual(len(result['evaluation_frames']),4)
            self.assertEqual(result['summary']['droid_initial'],result['summary']['pointworld_release'])
            # A completed report is replayed only after verifying its inputs.
            report_hash = sha256(out/'episode_000000/metrics.json')
            with patch('recam_refine.audit.Robot', SmallRobot), patch('recam_refine.audit.prepare_assets', return_value=None):
                run_audit(args)
            self.assertEqual(report_hash,sha256(out/'episode_000000/metrics.json'))


if __name__ == '__main__':
    unittest.main()
