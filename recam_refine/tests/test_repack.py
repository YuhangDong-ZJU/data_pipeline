"""TAR publication, round trips and refusal gates on disposable datasets."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from recam_refine.archives import unpack_archive
from recam_refine.common import RefineError, read_json, sha256, write_json
from recam_refine.repack import SUCCESS, pack_one, run_repack
from recam_refine.steps import MARKERS, transfer_only
from recam_refine.tests.test_steps import all_files, geometry_pass, manual_fixture, stage


def completed_fixture(tmp):
    args, droid, _ = manual_fixture(tmp)
    transfer_only(args)
    for name in ('unpack', 'align', 'overlap', 'refine', 'apply'):
        stage(args, name)
    # Exercise the actual full media checks and cleanup. Geometry pass/failure
    # orchestration is isolated from optimizer numerics, tested elsewhere.
    with patch('recam_refine.audit._run_audit_locked', side_effect=geometry_pass):
        stage(args, 'check')
    stage(args, 'cleanup')
    args.episodes_per_shard = 1
    return args, droid


class RepackTests(unittest.TestCase):
    def test_verified_round_trip_resume_and_archive_corruption(self):
        with tempfile.TemporaryDirectory() as td:
            args, droid = completed_fixture(Path(td))
            before = all_files(args.root)
            run_repack(args)
            record = read_json(args.work_dir / SUCCESS)
            self.assertTrue(record['complete'] and record['source_pngs_retained'])
            self.assertEqual(len(record['archives']), 4)
            self.assertEqual(sum(r['files'] for r in record['archives']), 24)
            for relative, digest in before.items():
                self.assertEqual(sha256(args.root / relative), digest)
            restored = Path(td) / 'restored'
            for item in record['archives']:
                self.assertIsNone(item['sha256'])
                unpack_archive(droid / item['path'], droid, Path(td) / 'receipts')
                # Use a separate extraction root, preserving the original TAR layout.
                archive_copy = restored / item['path']
                archive_copy.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(droid / item['path'], archive_copy)
                unpack_archive(archive_copy, restored, Path(td) / 'restored_receipts')
            expected = {p.relative_to(droid).as_posix(): sha256(p) for p in (droid / 'images').rglob('*.png')}
            actual = {p.relative_to(restored).as_posix(): sha256(p) for p in restored.rglob('*.png')}
            self.assertEqual(expected, actual)
            tar = droid / record['archives'][0]['path']
            original_time = tar.stat().st_mtime_ns
            with patch('recam_refine.repack.verify_existing_tar', side_effect=AssertionError('repeated completed TAR read')):
                run_repack(args)
            self.assertEqual(tar.stat().st_mtime_ns, original_time)
            self.assertTrue(all(r['status'] == 'reused' for r in read_json(args.work_dir / SUCCESS)['archives']))
            with tar.open('r+b') as f:
                f.seek(512)
                f.write(b'corrupt PNG bytes')
            with self.assertRaisesRegex(RefineError, 'TAR changed|TAR content differs'):
                run_repack(args)
            self.assertFalse((args.work_dir / SUCCESS).exists())
            # An existing conflicting TAR without its receipt is not overwritten.
            receipt = hashlib.sha256(record['archives'][0]['path'].encode()).hexdigest() + '.json'
            (args.work_dir / 'repacked_depth' / receipt).unlink()
            corrupted_digest = sha256(tar)
            with self.assertRaisesRegex(RefineError, 'TAR changed|TAR content differs'):
                run_repack(args)
            self.assertEqual(sha256(tar), corrupted_digest)
            for relative, digest in before.items():
                self.assertEqual(sha256(args.root / relative), digest)

    def test_check_cleanup_review_and_source_change_gates(self):
        with tempfile.TemporaryDirectory() as td:
            args, droid = completed_fixture(Path(td))
            for stage_name in ('check', 'cleanup'):
                path = args.work_dir / MARKERS[stage_name]
                saved = path.read_bytes()
                path.unlink()
                with self.assertRaisesRegex(RefineError, f'Run {stage_name} successfully'):
                    run_repack(args)
                path.write_bytes(saved)
            review = args.work_dir / 'camera_audit/QUALITY_REVIEW_REQUIRED.json'
            write_json(review, {'needs_review': True})
            with self.assertRaisesRegex(RefineError, 'requires review'):
                run_repack(args)
            review.unlink()
            depth = next((droid / 'images').rglob('*.png'))
            depth.write_bytes(b'changed after full check')
            with self.assertRaisesRegex(RefineError, 'Training files changed'):
                run_repack(args)
            self.assertFalse(list((droid / 'images').rglob('*.tar')))
            self.assertFalse((args.work_dir / SUCCESS).exists())

    def test_streaming_large_pngs_and_complete_archive_digest(self):
        with tempfile.TemporaryDirectory() as td:
            subset, work = Path(td) / 'droid', Path(td) / 'work'
            relative = 'images/chunk-000/observation.images.depth_01'
            directory = subset / relative / 'episode_000000'
            directory.mkdir(parents=True)
            rng = np.random.default_rng(42)
            for i in range(3):
                pixels = rng.integers(20, 10000, size=(720, 1280), dtype=np.uint16)
                Image.fromarray(pixels).save(directory / f'frame_{i:06d}.png')
            job = dict(path=relative + '/episodes-000000-000000.tar',
                       episodes=[dict(directory=relative + '/episode_000000', length=3)])
            result = pack_one(subset, work, job, 'large-png-test')
            self.assertGreater((subset / result['path']).stat().st_size, 3 * 1024 * 1024)
            self.assertIsNone(result['sha256'])
            self.assertEqual(pack_one(subset, work, job, 'large-png-test')['status'], 'reused')

    def test_failed_verification_does_not_publish_and_crash_can_resume(self):
        from recam_refine import repack
        with tempfile.TemporaryDirectory() as td:
            args, droid = completed_fixture(Path(td))
            args.workers = 1
            before = all_files(args.root)
            partial = droid / 'images/chunk-000/observation.images.depth_01/.episodes-000000-000000.tar.repack-part'
            partial.write_bytes(b'interrupted write from the same packing plan')
            with patch.object(repack.tarfile.TarFile, 'addfile', side_effect=RefineError('injected write failure')):
                with self.assertRaisesRegex(RefineError, 'injected write failure'):
                    run_repack(args)
            self.assertFalse(list((droid / 'images').rglob('*.tar')))
            self.assertFalse(list((droid / 'images').rglob('*.repack-part')))
            self.assertFalse((args.work_dir / SUCCESS).exists())
            original_write = repack.write_json
            def fail_receipt(path, value):
                if path.parent.name == 'repacked_depth':
                    raise OSError('injected crash after TAR publication')
                original_write(path, value)
            with patch.object(repack, 'write_json', side_effect=fail_receipt):
                with self.assertRaisesRegex(OSError, 'injected crash'):
                    run_repack(args)
            self.assertTrue(list((droid / 'images').rglob('*.tar')))
            self.assertFalse((args.work_dir / SUCCESS).exists())
            run_repack(args)
            self.assertTrue(read_json(args.work_dir / SUCCESS)['complete'])
            for relative, digest in before.items():
                self.assertEqual(sha256(args.root / relative), digest)

    def test_wrong_archive_extra_episodes_and_low_disk_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            args, droid = completed_fixture(Path(td))
            camera = droid / 'images/chunk-000/observation.images.depth_01'
            extra = camera / 'episode_999999'
            extra.mkdir()
            with self.assertRaisesRegex(RefineError, 'extra external depth'):
                run_repack(args)
            extra.rmdir()
            old = camera / 'episodes-999998-999999.tar'
            old.write_bytes(b'preserve unknown archive')
            with self.assertRaisesRegex(RefineError, 'Unexpected old depth TARs'):
                run_repack(args)
            self.assertEqual(old.read_bytes(), b'preserve unknown archive')
            old.unlink()
            with patch('recam_refine.repack.shutil.disk_usage', return_value=argparse.Namespace(free=0)):
                with self.assertRaisesRegex(RefineError, 'Insufficient free disk'):
                    run_repack(args)
            self.assertFalse(list((droid / 'images').rglob('*.tar')))


if __name__ == '__main__':
    unittest.main()
