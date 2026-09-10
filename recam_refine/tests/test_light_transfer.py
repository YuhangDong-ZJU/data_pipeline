import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from recam_refine.common import read_json, sha256, write_json
from recam_refine.tests.test_steps import source_depth
from recam_refine.transfer import transfer_depth


class LightTransferTests(unittest.TestCase):
    def exercise(self, legacy=False, interrupted=False, cross=False):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory(dir='/dev/shm' if cross else None) as other:
            base = Path(tmp)
            source, root, work = base/'source', (Path(other) if cross else base)/'dataset', base/'work'
            droid = root/'real_world/droid'
            droid.mkdir(parents=True)
            row = dict(episode_index=0, length=3, source_episode_id='test+0',
                       camera_serials={'external_1':'a', 'external_2':'b'})
            source_depth(source, [row])
            original = {p.relative_to(source/'images'): p.read_bytes() for p in (source/'images').glob('*/*/*/*.png')}
            if legacy:
                for cam in (1, 2):
                    rel = Path(f'images/chunk-000/observation.images.depth_{cam:02d}/episode_000000')
                    write_json(work/'transfer_receipts'/f'episode_000000_{cam}.json', dict(
                        source=str(source/rel), target=str(droid/rel), complete=False,
                        files=[dict(name=p.name, sha256=sha256(p)) for p in sorted((source/rel).glob('*.png'))]))
            if interrupted:
                with patch('recam_refine.transfer.sync_dir', side_effect=RuntimeError('interrupted')):
                    with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                        transfer_depth(root, droid, source, {0:row}, {0}, work)
            if cross:
                self.assertNotEqual(source.stat().st_dev, droid.stat().st_dev)
                transfer_depth(root, droid, source, {0:row}, {0}, work)
            else:
                with patch('recam_refine.transfer.sha256', side_effect=AssertionError('unnecessary full read')):
                    transfer_depth(root, droid, source, {0:row}, {0}, work)
            # Completed receipts do not trigger content rescans, even for legacy hashes.
            with patch('recam_refine.transfer.sha256', side_effect=AssertionError('repeat read')):
                transfer_depth(root, droid, source, {0:row}, {0}, work)
            for rel, data in original.items():
                self.assertEqual((droid/'images'/rel).read_bytes(), data)
            self.assertTrue(all(read_json(p)['complete'] for p in (work/'transfer_receipts').glob('*.json')))

    def test_atomic_rename_and_completed_resume_without_hash_scan(self):
        self.exercise()

    def test_legacy_preflight_receipts(self):
        self.exercise(legacy=True)

    def test_interrupted_directory_rename(self):
        self.exercise(interrupted=True)

    @unittest.skipUnless(os.path.isdir('/dev/shm'), 'Requires Linux second filesystem')
    def test_cross_filesystem_copy(self):
        self.exercise(cross=True)
