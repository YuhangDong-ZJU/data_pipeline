import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from recam_refine.steps import unpack_only
from recam_refine.common import read_json, write_json, safe_path

class UnpackRangeTests(unittest.TestCase):
    def test_unrelated_transfer_receipts_are_not_read(self):
        with tempfile.TemporaryDirectory() as td:
            root, work = Path(td)/'root', Path(td)/'work'
            subset = root/'real_world/droid'
            directory = subset/'images/chunk-014/observation.images.depth_01'
            directory.mkdir(parents=True)
            with tarfile.open(directory/'depth.tar','w') as tar:
                item = tarfile.TarInfo('episode_014000/frame_000000.png')
                item.size = 3
                tar.addfile(item,io.BytesIO(b'png'))
            original_glob = Path.glob
            def selected_glob(path, pattern):
                self.assertNotEqual(path, work/'transfer_receipts')
                self.assertFalse('chunk-*' in pattern)
                return original_glob(path,pattern)
            with patch('recam_refine.pipeline.discover',return_value=[subset]), patch.object(Path,'glob',selected_glob), patch('recam_refine.steps.read_json',side_effect=AssertionError('No migration JSON reads')):
                unpack_only(root,work,2,{14})
                with patch('recam_refine.archives.tarfile.open',side_effect=AssertionError('Repeated TAR read')), patch('recam_refine.archives.safe_path', wraps=safe_path) as checked_paths:
                    unpack_only(root,work,2,{14})
                    self.assertEqual(checked_paths.call_count, 1)
                    self.assertFalse(str(checked_paths.call_args.args[1]).endswith('.png'))

    def test_ranges_merge_and_repeat_skips_payload(self):
        with tempfile.TemporaryDirectory() as td:
            root, work = Path(td)/'root', Path(td)/'work'
            subset = root/'simulation/libero'
            for chunk in (0, 1):
                directory = subset/f'images/chunk-{chunk:03d}/observation.images.depth_01'
                directory.mkdir(parents=True)
                with tarfile.open(directory/'depth.tar', 'w') as tar:
                    member = tarfile.TarInfo(f'images/chunk-{chunk:03d}/observation.images.depth_01/episode_000000/frame_000000.png')
                    member.size = 4
                    tar.addfile(member, io.BytesIO(b'data'))
            with patch('recam_refine.pipeline.discover', return_value=[subset]):
                first = unpack_only(root,work,2,{0})
                self.assertFalse(first['all_chunks_complete'])
                self.assertEqual(first['archives'],1)
                self.assertFalse(list((subset/'images/chunk-001').rglob('*.png')))
                with patch('recam_refine.archives.tarfile.open', side_effect=AssertionError('Repeated TAR read')):
                    repeated = unpack_only(root,work,2,{0})
                self.assertEqual(repeated['archives'],1)
                final = unpack_only(root,work,2,{1})
                self.assertFalse(final['all_chunks_complete'])
                self.assertFalse((work/'unpacked.json').exists())
                with patch('recam_refine.archives.tarfile.open', side_effect=AssertionError('Repeated TAR read')):
                    final = unpack_only(root,work,2)
                self.assertTrue(final['all_chunks_complete'])
                self.assertEqual(len(read_json(work/'unpacked.json')),2)

    def test_partial_range_does_not_publish_global_success(self):
        from argparse import Namespace
        from contextlib import nullcontext
        from recam_refine.steps import run_step, MARKERS
        with tempfile.TemporaryDirectory() as td:
            root, work = Path(td)/'root', Path(td)/'work'
            work.mkdir()
            (root/'real_world/droid').mkdir(parents=True)
            (work/MARKERS['transfer']).write_text('{}')
            args = Namespace(root=root, work_dir=work, step='unpack', chunk_ids='0-6', workers=3)
            result = dict(all_chunks_complete=False, selected_archives=1, archives=1, total_archives=2)
            with patch('recam_refine.steps.locked_step', return_value=nullcontext()), patch('recam_refine.pipeline.discover', return_value=[]), patch('recam_refine.steps.unpack_only', return_value=result) as unpack:
                run_step(args)
                unpack.assert_called_once_with(root,work,3,set(range(7)))
                self.assertFalse((work/MARKERS['unpack']).exists())
                result['all_chunks_complete'] = True
                run_step(args)
                self.assertTrue((work/MARKERS['unpack']).exists())
