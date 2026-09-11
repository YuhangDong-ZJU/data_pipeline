import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from recam_refine.common import Journal, RefineError, copy_checked_size


class JournalIOTests(unittest.TestCase):
    def test_backup_copies_once_without_payload_hash(self):
        with tempfile.TemporaryDirectory() as td:
            root, work = Path(td)/'data', Path(td)/'work'
            root.mkdir()
            src = root/'depth.png'
            src.write_bytes(b'original')
            with patch('recam_refine.common.sha256',side_effect=AssertionError('Payload hash')):
                journal = Journal(root,work)
                saved = journal.backup(src)
                self.assertEqual(saved.read_bytes(),b'original')
                with patch('recam_refine.common.shutil.copy2',side_effect=AssertionError('Repeated backup')):
                    journal.backup(src)

    def test_truncated_copy_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            src,dst = Path(td)/'src',Path(td)/'dst'
            src.write_bytes(b'original')
            with patch('recam_refine.common.shutil.copy2',side_effect=lambda a,b:b.write_bytes(b'bad')):
                with self.assertRaisesRegex(RefineError,'Copy incomplete'):
                    copy_checked_size(src,dst)
            self.assertEqual(src.read_bytes(),b'original')

    def test_conflicting_retirement_does_not_delete_source(self):
        with tempfile.TemporaryDirectory() as td:
            root,work = Path(td)/'data',Path(td)/'work'
            root.mkdir()
            src = root/'log.txt'
            src.write_bytes(b'one')
            dst = work/'retired/log.txt'
            dst.parent.mkdir(parents=True)
            dst.write_bytes(b'two')
            with patch('recam_refine.common.sha256',side_effect=AssertionError('Payload hash')):
                with self.assertRaisesRegex(RefineError,'Retirement conflict'):
                    Journal(root,work).retire(src)
            self.assertEqual(src.read_bytes(),b'one')
