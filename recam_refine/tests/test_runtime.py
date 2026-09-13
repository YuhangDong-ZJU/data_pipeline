"""Runtime readiness without filesystem locks."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from recam_refine.bootstrap import runtime_lock

class RuntimeLeaseTests(unittest.TestCase):
    def test_processing_modules_do_not_import_file_lock_primitives(self):
        import ast
        import re
        package = Path(__file__).resolve().parents[1]
        for path in package.rglob('*.py'):
            if 'tests' in path.relative_to(package).parts:
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
                if isinstance(node,ast.Import):
                    self.assertFalse(any(name.name=='fcntl' for name in node.names),str(path))
                if isinstance(node,ast.ImportFrom):
                    self.assertNotEqual(node.module,'fcntl',str(path))
        for path in package.rglob('*.sh'):
            self.assertIsNone(re.search(r'\bflock\b',path.read_text()),str(path))

    def test_context_never_locks_or_creates_lock_files(self):
        with tempfile.TemporaryDirectory() as td, patch('fcntl.flock',side_effect=AssertionError('File lock')):
            root=Path(td)/'runtime'
            with runtime_lock(root):
                (root/'env/bin').mkdir(parents=True)
                (root/'env/bin/python').touch()
            with runtime_lock(root,True),runtime_lock(root):
                pass
            self.assertFalse((root/'bootstrap.lock').exists())

    def test_missing_prepared_runtime_is_not_created(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'missing'
            with self.assertRaisesRegex(RuntimeError,'Runtime not prepared'):
                with runtime_lock(root,True):
                    pass
            self.assertFalse(root.exists())
