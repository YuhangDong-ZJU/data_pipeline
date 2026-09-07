"""Real POSIX leases protect a shared environment from concurrent mutation."""
from pathlib import Path
import tempfile
import unittest

from recam_refine.bootstrap import runtime_lock


class RuntimeLeaseTests(unittest.TestCase):
    def test_workers_share_runtime_and_block_installation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)/'runtime'
            with runtime_lock(root):
                pass
            with runtime_lock(root,True),runtime_lock(root,True):
                with self.assertRaisesRegex(RuntimeError,'Runtime is in use'):
                    with runtime_lock(root):
                        self.fail('Installer entered while workers were reading')
            with runtime_lock(root):
                pass

    def test_installer_blocks_worker_even_through_symlink(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)/'runtime'
            alias = Path(td)/'alias'
            with runtime_lock(root):
                alias.symlink_to(root,target_is_directory=True)
                with self.assertRaisesRegex(RuntimeError,'Runtime is in use'):
                    with runtime_lock(alias,True):
                        self.fail('Worker read a partially installed runtime')

    def test_missing_prepared_runtime_is_not_created(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)/'missing'
            with self.assertRaisesRegex(RuntimeError,'Runtime not prepared'):
                with runtime_lock(root,True):
                    self.fail('Missing environment was accepted')
            self.assertFalse(root.exists())
