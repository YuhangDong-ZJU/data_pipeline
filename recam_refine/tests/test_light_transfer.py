import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from recam_refine.common import read_json, sha256, write_json
from recam_refine.tests.test_steps import source_depth
from recam_refine.transfer import transfer_depth


class LightTransferTests(unittest.TestCase):
    def test_sidecar_cache_reuses_validated_data_and_rejects_changed_timestamps(self):
        from recam_refine.inputs import load_depth_records
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            row = dict(episode_index=0, length=3, source_episode_id='test+0',
                       camera_serials={'external_1':'a', 'external_2':'b'})
            source_depth(base/'source', [row])
            first = load_depth_records([base/'source'], {0:row}, base/'cache')
            with patch.object(Path, 'read_bytes', side_effect=AssertionError('repeated sidecar read')):
                self.assertEqual(first, load_depth_records([base/'source'], {0:row}, base/'cache'))
            path = Path(first[0,1]['path'])
            record = read_json(path)
            record['source']['timestamps_ms'] = [1,1,2]
            write_json(path, record)
            with self.assertRaisesRegex(Exception, 'Unordered timestamps'):
                load_depth_records([base/'source'], {0:row}, base/'cache')

    def test_final_png_decodes_once_without_verify_and_rejects_broken_data(self):
        import numpy as np
        from PIL import Image
        from recam_refine.archives import checked_png_array
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'frame.png'
            expected = np.full((8, 8), 1234, dtype=np.uint16)
            Image.fromarray(expected).save(path)
            reader = Path.read_bytes
            from PIL.PngImagePlugin import PngImageFile
            with patch.object(Path, 'read_bytes', autospec=True, side_effect=reader) as read, \
                    patch.object(PngImageFile, 'verify', side_effect=AssertionError('redundant verify')):
                np.testing.assert_array_equal(checked_png_array(path, (8, 8, 1)), expected)
                self.assertEqual(read.call_count, 1)
            data = bytearray(reader(path))
            data[data.index(b'IDAT') + 4] ^= 1
            path.write_bytes(data)
            with self.assertRaises(Exception):
                checked_png_array(path, (8, 8, 1))

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
                with patch('recam_refine.transfer.sha256', side_effect=AssertionError('unnecessary full read')), \
                        patch('recam_refine.inputs.load_depth_records', side_effect=AssertionError('bulk JSON read in transfer')):
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
