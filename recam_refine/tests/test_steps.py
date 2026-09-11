"""Stage boundaries and recovery, using disposable datasets only."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from recam_refine.archives import unpack_archive
from recam_refine.common import RefineError, read_json, sha256, values, write_json, write_jsonl
from recam_refine.steps import MARKERS, run_step, transfer_only
from recam_refine.tests.test_refine import fixture


def source_depth(source, rows):
    for row in rows:
        i, n = row['episode_index'], row['length']
        for cam, serial in ((1, 'a'), (2, 'b')):
            decoded = n-2 if i==1 and cam==1 else n
            stream = f'chunk-{i//1000:03d}/observation.images.depth_{cam:02d}'
            directory = source/f'images/{stream}/episode_{i:06d}'
            directory.mkdir(parents=True)
            for f in range(n):
                Image.fromarray(np.full((720,1280),1000 if f<decoded else 0,np.uint16)).save(directory/f'frame_{f:06d}.png')
            write_json(source/f'annotations/foundation_stereo_depth/{stream}/episode_{i:06d}.json',
                dict(source=dict(episode_index=i,source_episode_id=row['source_episode_id'],camera_role=f'external_{cam}',
                    camera_serial=serial,frame_count=n,decoded_frame_count=decoded,tail_missing_count=n-decoded,
                    missing_frame_indices=list(range(decoded,n)),timestamps_ms=[float(f) if f<decoded else None for f in range(n)]),
                     inference={'method':'FoundationStereo'},calibration={'intrinsic':[[1000,0,640],[0,1000,360],[0,0,1]]}))


def arguments(tmp):
    return argparse.Namespace(root=tmp/'recam_lerobot',work_dir=tmp/'work',depth_output=tmp/'source',depth_chunks='0',
        episode_manifest=tmp/'manifest.jsonl',pointworld_cameras=tmp/'cameras',depth_metadata=[],
        workers=2,devices='cpu',iterations=1,audit_frames=4)


def stage(args, name):
    args.step = name
    run_step(args)


def geometry_pass(args):
    # Orchestration test only: geometry arithmetic/report generation are tested
    # independently in test_camera_audit, with real robot/depth sample validation.
    write_json(args.report_dir/'summary.json',{'test_fixture':True})
    (args.report_dir/'QUALITY_REVIEW_REQUIRED.json').unlink(missing_ok=True)
    return 0


def all_files(root):
    return {str(p.relative_to(root)):sha256(p) for p in root.rglob('*') if p.is_file()}


def manual_fixture(tmp):
    args = arguments(tmp)
    droid = args.root/'real_world/droid'
    simulation = args.root/'simulation/test_sim'
    info = fixture(droid)
    fixture(simulation,False)
    archives = []
    for subset,cam in ((droid,1),(simulation,0)):
        camera = subset/f'images/chunk-000/observation.images.depth_{cam:02d}'
        archive = camera/'episodes-000000-000001.tar'
        with tarfile.open(archive,'w') as tar:
            for p in sorted(camera.glob('episode_*/*.png')):
                tar.add(p,arcname=p.relative_to(subset).as_posix())
        if subset==simulation:
            for p in camera.glob('episode_*/*.png'):
                p.unlink()
        archives.append(archive)
    # Source metric depth is HD; stale DROID TARs contain the old low-res data.
    for cam in (1,2):
        info['features'][f'observation.images.depth_{cam:02d}']['shape'] = [720,1280,1]
    write_json(droid/'meta/info.json',info)
    rows = [dict(episode_index=i,length=8,source_episode_id=f'lab+{i}',tasks=['test task'],
                 camera_serials={'external_1':'a','external_2':'b'}) for i in range(2)]
    write_jsonl(args.episode_manifest,rows)
    source_depth(args.depth_output,rows)
    c2w = np.eye(4)
    c2w[0,3] = .02
    for row in rows:
        uuid = row['source_episode_id']
        write_json(args.pointworld_cameras/f'{uuid}_cameras.json',dict(uuid=uuid,optimization_success=True,
            optimization_summary={'final_loss':.04},a={'optimized_extrinsics':np.linalg.inv(c2w).tolist()},
            b={'optimized_extrinsics':np.linalg.inv(c2w).tolist()}))
    (droid/'logs').mkdir()
    (droid/'logs/run.log').write_text('preserve this log outside the training dataset')
    return args,droid,archives


class ManualStepsTests(unittest.TestCase):
    def test_transfer_preserves_bytes_and_defers_png_decode_to_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = arguments(Path(tmp))
            droid = args.root/'real_world/droid'
            rows = [dict(episode_index=0,length=2,source_episode_id='lab+0',
                         camera_serials={'external_1':'a','external_2':'b'})]
            write_json(droid/'meta/info.json',dict(codebase_version='v2.1',chunks_size=1000))
            write_jsonl(droid/'meta/episodes.jsonl',[dict(episode_index=0,length=2)])
            write_jsonl(args.episode_manifest,rows)
            source_depth(args.depth_output,rows)
            bad = args.depth_output/'images/chunk-000/observation.images.depth_02/episode_000000/frame_000001.png'
            bad.write_bytes(b'corrupt PNG')
            transfer_only(args)
            copied = droid/bad.relative_to(args.depth_output)
            self.assertEqual(copied.read_bytes(), b'corrupt PNG')
            from recam_refine.archives import checked_png_array
            with self.assertRaises(Exception):
                checked_png_array(copied)
            expected = all_files(droid)
            self.assertEqual(read_json(args.work_dir/MARKERS['transfer'])['png_files'],4)
            self.assertFalse((args.work_dir/MARKERS['unpack']).exists())
            self.assertFalse(args.pointworld_cameras.exists())
            transfer_only(args)
            self.assertEqual(expected,all_files(droid))

    def test_superseded_archive_does_not_resurrect_old_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'dataset'
            directory = root/'images/chunk-000/observation.images.depth_01/episode_000000'
            directory.mkdir(parents=True)
            paths = [directory/f'frame_{i:06d}.png' for i in range(2)]
            for p in paths:
                Image.fromarray(np.full((8,8),300,np.uint16)).save(p)
            archive = directory.parent/'episodes-000000-000000.tar'
            with tarfile.open(archive,'w') as tar:
                for p in paths:
                    tar.add(p,arcname=p.relative_to(root).as_posix())
            paths[1].unlink()
            Image.fromarray(np.full((8,8),1000,np.uint16)).save(paths[0])
            authority = {directory.relative_to(root).as_posix():{paths[0].name:sha256(paths[0])}}
            result = unpack_archive(archive,root,Path(tmp)/'receipts',authority)
            self.assertEqual(result['superseded_by_metric_depth'],2)
            self.assertFalse(paths[1].exists())
            self.assertEqual(sha256(paths[0]),authority[directory.relative_to(root).as_posix()][paths[0].name])
            self.assertTrue(archive.exists())
            Image.fromarray(np.full((8,8),2000,np.uint16)).save(paths[0])
            with self.assertRaisesRegex(RefineError,'Extracted/migrated file changed'):
                unpack_archive(archive,root,Path(tmp)/'receipts',authority)

    def test_manual_lifecycle_resume_and_cleanup_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,droid,archives = manual_fixture(Path(tmp))
            original = pq.read_table(droid/'data/chunk-000/episode_000001.parquet')
            with self.assertRaisesRegex(RefineError,'Run transfer first'):
                stage(args,'unpack')
            transfer_only(args)
            self.assertEqual(read_json(droid/'meta/info.json')['total_frames'],14)
            stage(args,'unpack')
            self.assertTrue(all(p.exists() for p in archives))
            self.assertEqual(read_json(droid/'meta/info.json')['total_frames'],14)
            stage(args,'align')
            self.assertEqual(read_json(droid/'meta/info.json')['total_frames'],12)
            aligned = pq.read_table(droid/'data/chunk-000/episode_000001.parquet')
            np.testing.assert_array_equal(values(aligned['observation.camera.extrinsics']),values(original['observation.camera.extrinsics'])[:6])
            self.assertFalse((args.work_dir/'cameras').exists())
            # Simulate a crash after a Parquet write and before the stage marker.
            aligned_files = all_files(droid)
            (args.work_dir/MARKERS['align']).unlink()
            (args.work_dir/'aligned/episode_000001.json').unlink()
            stage(args,'align')
            self.assertEqual(aligned_files,all_files(droid))
            with self.assertRaisesRegex(RefineError,'Run overlap first'):
                stage(args,'refine')
            stage(args,'overlap')
            self.assertEqual(read_json(args.work_dir/'pointworld_overlap.json')['pointworld_release'],2)
            stage(args,'refine')
            self.assertEqual(aligned_files,all_files(droid))
            self.assertFalse((args.work_dir/MARKERS['apply']).exists())
            stage(args,'apply')
            applied = pq.read_table(droid/'data/chunk-000/episode_000001.parquet')
            np.testing.assert_allclose(values(applied['observation.camera.extrinsics'])[:,1:,0,3],.02)
            np.testing.assert_array_equal(values(applied['observation.camera.extrinsics'])[:,0],values(original['observation.camera.extrinsics'])[:6,0])
            for column in ('timestamp','observation.states.joint_state','observation.states.gripper_state'):
                self.assertTrue(applied[column].equals(original[column].slice(0,6)))
            final_files = all_files(droid)
            stage(args,'apply')
            self.assertEqual(final_files,all_files(droid))
            with self.assertRaisesRegex(RefineError,'Run check first'):
                stage(args,'cleanup')
            with patch('recam_refine.audit._run_audit_locked',side_effect=geometry_pass):
                stage(args,'check')
            self.assertTrue(all(p.exists() for p in archives))
            self.assertTrue((droid/'logs/run.log').exists())
            video = droid/'videos/chunk-000/observation.images.rgb_01/episode_000000.mp4'
            stat = video.stat()
            os.utime(video,ns=(stat.st_atime_ns,stat.st_mtime_ns+1_000_000))
            with self.assertRaisesRegex(RefineError,'Training files changed after check'):
                stage(args,'cleanup')
            self.assertTrue(all(p.exists() for p in archives))
            with patch('recam_refine.audit._run_audit_locked',return_value=2):
                with self.assertRaisesRegex(RefineError,'Geometry requires review'):
                    stage(args,'check')
            self.assertFalse((args.work_dir/MARKERS['check']).exists())
            with patch('recam_refine.audit._run_audit_locked',side_effect=geometry_pass):
                stage(args,'check')
            stage(args,'cleanup')
            self.assertFalse(archives[0].exists())
            self.assertTrue(archives[1].exists())
            self.assertFalse((droid/'logs/run.log').exists())
            self.assertTrue((args.work_dir/'auxiliary/real_world/droid/logs/run.log').exists())
            self.assertTrue((args.work_dir/'SUCCESS.json').exists())
            self.assertTrue(read_json(args.work_dir/'SUCCESS.json')['full_decode'])
            stage(args,'cleanup')

    def test_rejects_mixing_automatic_and_manual_workdirs(self):
        from recam_refine.pipeline import run
        with tempfile.TemporaryDirectory() as tmp:
            args = arguments(Path(tmp))
            (args.root/'real_world/droid').mkdir(parents=True)
            write_json(args.work_dir/'configuration.json',{})
            with self.assertRaisesRegex(RefineError,'automatic run'):
                stage(args,'unpack')
            (args.work_dir/'configuration.json').unlink()
            write_json(args.work_dir/'manual_workflow.json',{'version':1,'root':str(args.root)})
            with self.assertRaisesRegex(RefineError,'manual'):
                run(args)


if __name__=='__main__':
    unittest.main()
