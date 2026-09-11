import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from recam_refine.steps import unpack_only
from recam_refine.common import read_json

class UnpackRangeTests(unittest.TestCase):
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
                self.assertTrue(final['all_chunks_complete'])
                self.assertEqual(len(read_json(work/'unpacked.json')),2)

    def test_partial_range_does_not_publish_global_success(self):
        from argparse import Namespace
        from contextlib import nullcontext
        from recam_refine.steps import run_step, MARKERS
        with tempfile.TemporaryDirectory() as td:
            root, work = Path(td)/'root', Path(td)/'work'
            work.mkdir()
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
