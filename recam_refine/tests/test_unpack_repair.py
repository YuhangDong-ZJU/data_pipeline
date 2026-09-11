import io
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from recam_refine.archives import unpack_archive


class UnpackRepairTests(unittest.TestCase):
    def fixture(self, root):
        subset = root/'simulation/libero'
        directory = subset/'images/chunk-002/observation.images.depth_00/episode_002657'
        directory.mkdir(parents=True)
        archive = directory.parent/'episodes-002500-002749.tar'
        with tarfile.open(archive,'w') as tar:
            for i in range(3):
                member = tarfile.TarInfo(f'episode_002657/frame_{i:06d}.png')
                member.size = 8
                tar.addfile(member,io.BytesIO(bytes([i])*8))
        unpack_archive(archive,subset,root/'receipts')
        return archive,subset,directory

    def test_permission_and_time_changes_do_not_read_tar(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            archive,subset,directory = self.fixture(root)
            frame = directory/'frame_000000.png'
            os.utime(frame,(10,10))
            frame.chmod(0o600)
            with patch('recam_refine.archives.tarfile.open',side_effect=AssertionError('Repeated TAR read')):
                unpack_archive(archive,subset,root/'receipts')

    def test_only_missing_or_wrong_size_members_are_restored(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            archive,subset,directory = self.fixture(root)
            good = directory/'frame_000000.png'
            good_state = good.stat()
            (directory/'frame_000001.png').unlink()
            (directory/'frame_000002.png').write_bytes(b'bad')
            extracted = []
            original = tarfile.TarFile.extractfile
            def observe(tar,member):
                extracted.append(member.name)
                return original(tar,member)
            with patch.object(tarfile.TarFile,'extractfile',observe):
                unpack_archive(archive,subset,root/'receipts')
            self.assertEqual(len(extracted),2)
            self.assertFalse(any('frame_000000' in name for name in extracted))
            self.assertEqual(good.stat().st_mtime_ns,good_state.st_mtime_ns)
            for i in range(3):
                self.assertEqual((directory/f'frame_{i:06d}.png').read_bytes(),bytes([i])*8)
            with patch('recam_refine.archives.tarfile.open',side_effect=AssertionError('Repeated TAR read')):
                unpack_archive(archive,subset,root/'receipts')
