import io
import multiprocessing as mp
import tarfile
import tempfile
import unittest
from pathlib import Path
from argparse import Namespace
from unittest.mock import patch
from recam_refine.steps import locked_step, unpack_claim, run_step, MARKERS
from recam_refine.common import read_json, write_json


def worker(root, work, chunk, start):
    start.wait(10)
    run_step(Namespace(root=root,work_dir=work,step='unpack',chunk_ids=str(chunk),workers=2))


def hold_claim(root, work, directory, ready):
    with locked_step(root,work,shared=True), unpack_claim(work,directory,'cameras') as held:
        assert held
        ready.set()
        import time
        time.sleep(30)


class ConcurrentUnpackTests(unittest.TestCase):
    def setup_data(self, td):
        root,work=Path(td)/'root',Path(td)/'work'
        subset=root/'real_world/droid'
        write_json(subset/'meta/info.json',{'codebase_version':'v2.1'})
        write_json(work/MARKERS['transfer'],{'complete':True})
        for chunk in (0,1):
            directory=subset/f'images/chunk-{chunk:03d}/observation.images.depth_01'
            directory.mkdir(parents=True)
            with tarfile.open(directory/'depth.tar','w') as tar:
                member=tarfile.TarInfo(f'images/chunk-{chunk:03d}/observation.images.depth_01/episode_000000/frame_000000.png')
                member.size=4
                tar.addfile(member,io.BytesIO(b'data'))
        return root,work,subset

    def test_two_processes_merge_without_losing_receipts(self):
        with tempfile.TemporaryDirectory() as td:
            root,work,subset=self.setup_data(td)
            context=mp.get_context('fork')
            start=context.Event()
            children=[context.Process(target=worker,args=(root,work,c,start)) for c in (0,1)]
            for child in children: child.start()
            start.set()
            for child in children:
                child.join(15)
                if child.is_alive(): child.terminate(); child.join()
                self.assertEqual(child.exitcode,0)
            self.assertEqual(len(read_json(work/'unpacked.json')),2)
            self.assertTrue(read_json(work/MARKERS['unpack'])['complete'])
            self.assertEqual(len(list(subset.rglob('*.png'))),2)

    def test_busy_skip_exclusive_block_and_crash_recovery(self):
        with tempfile.TemporaryDirectory() as td:
            root,work,subset=self.setup_data(td)
            context=mp.get_context('fork')
            ready=context.Event()
            child=context.Process(target=hold_claim,args=(root,work,subset/'images/chunk-000/observation.images.depth_01',ready))
            child.start()
            try:
                self.assertTrue(ready.wait(10))
                with self.assertRaisesRegex(RuntimeError,'Another refinement'):
                    with locked_step(root,work): pass
                run_step(Namespace(root=root,work_dir=work,step='unpack',chunk_ids='0',workers=2))
                self.assertFalse((work/MARKERS['unpack']).exists())
                self.assertFalse(list(subset.rglob('*.png')))
            finally:
                child.terminate(); child.join(5)
            run_step(Namespace(root=root,work_dir=work,step='unpack',chunk_ids='0-1',workers=2))
            self.assertTrue((work/MARKERS['unpack']).exists())

    def test_old_per_tar_receipt_survives_missing_global_index(self):
        from recam_refine.archives import unpack_archive
        with tempfile.TemporaryDirectory() as td:
            root,work,subset=self.setup_data(td)
            archive=subset/'images/chunk-000/observation.images.depth_01/depth.tar'
            unpack_archive(archive,subset,work/'archive_receipts')
            run_step(Namespace(root=root,work_dir=work,step='unpack',chunk_ids='1',workers=2))
            self.assertTrue((work/MARKERS['unpack']).exists())
            self.assertEqual(len(read_json(work/'unpacked.json')),2)

    def test_other_host_device_ids_do_not_force_reextraction(self):
        from recam_refine.archives import unpack_archive
        with tempfile.TemporaryDirectory() as td:
            root,work,subset=self.setup_data(td)
            archive=subset/'images/chunk-000/observation.images.depth_01/depth.tar'
            unpack_archive(archive,subset,work/'archive_receipts')
            receipt=next((work/'archive_receipts').glob('*.json'))
            saved=read_json(receipt)
            saved['archive_state'][:2]=[98765,43210]
            for value in saved['target_states'].values():
                if value is not None: value[:2]=[98765,43210]
            write_json(receipt,saved)
            with patch('recam_refine.archives.tarfile.open', side_effect=AssertionError('Unexpected payload read')):
                unpack_archive(archive,subset,work/'archive_receipts')
