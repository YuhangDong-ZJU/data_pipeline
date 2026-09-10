import tempfile
import unittest
from functools import partial
from pathlib import Path

from recam_refine.parallel import io_map
from recam_refine.pipeline import _cleanup_source


class SourceCleanupTests(unittest.TestCase):
    def test_parallel_cleanup_retains_final_and_backs_up_tail(self):
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            records = []
            for camera in (1, 2):
                src, dst = work / f'source{camera}', work / f'target{camera}'
                src.mkdir()
                dst.mkdir()
                (src / 'frame_000000.png').write_bytes(b'kept')
                (dst / 'frame_000000.png').write_bytes(b'kept')
                (src / 'frame_000001.png').write_bytes(b'tail')
                records.append(dict(source=str(src), target=str(dst), episode_index=4, camera=camera))
            io_map(partial(_cleanup_source, work=work), records, 2, 'cleanup test')
            for rec in records:
                self.assertFalse(Path(rec['source']).exists())
                self.assertEqual((Path(rec['target']) / 'frame_000000.png').read_bytes(), b'kept')
                self.assertEqual((work / 'source_removed_tails/episode_000004' / str(rec['camera']) / 'frame_000001.png').read_bytes(), b'tail')
                _cleanup_source(rec, work)

    def test_mismatch_preserves_source(self):
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            src, dst = work / 'source', work / 'target'
            src.mkdir()
            dst.mkdir()
            (src / 'frame_000000.png').write_bytes(b'good')
            (dst / 'frame_000000.png').write_bytes(b'bad!')
            with self.assertRaisesRegex(Exception, 'Source/final depth differs'):
                _cleanup_source(dict(source=str(src), target=str(dst), episode_index=4, camera=1), work)
            self.assertEqual((src / 'frame_000000.png').read_bytes(), b'good')
