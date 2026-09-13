import errno
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from recam_refine.environment import select, freeze_gpu


class GPULockCompatibilityTests(unittest.TestCase):
    def test_no_install_selection_never_acquires_environment_lock(self):
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            snapshot = dict(prefix=str(work/'env'))
            report = dict(python_version='3.11.11',packages={'torch':'2.8.0+cu129'})
            with patch('recam_refine.environment.candidates',return_value=[sys.executable]), \
                 patch('recam_refine.dependencies.inventory',return_value=snapshot), \
                 patch('recam_refine.dependencies.environment_lease',side_effect=AssertionError('Unexpected environment lock')), \
                 patch('recam_refine.bootstrap.runtime_lock',side_effect=AssertionError('Unexpected runtime lock')), \
                 patch('recam_refine.environment.subprocess.run',return_value=subprocess.CompletedProcess([],0,json.dumps(report),'') ):
                python,_,fds = select(work,'gpu',work/'cache',install_missing=False)
                self.assertEqual(python,sys.executable)
                self.assertEqual(fds,())

    def test_environment_record_survives_enosys(self):
        with tempfile.TemporaryDirectory() as td, patch('fcntl.flock',side_effect=OSError(errno.ENOSYS,'unsupported')):
            work = Path(td)
            report = dict(python_version='3.11.11',packages={'torch':'2.8.0+cu129'})
            freeze_gpu(work,report)
            freeze_gpu(work,report)
            with self.assertRaises(RuntimeError):
                freeze_gpu(work,{**report,'packages':{'torch':'other'}})

    def test_both_shards_start_when_flock_is_unimplemented(self):
        from recam_refine.shards import worker_locks
        with tempfile.TemporaryDirectory() as td:
            root,work = Path(td)/'data',Path(td)/'work'
            root.mkdir(); work.mkdir()
            for code in (errno.ENOSYS,errno.EOPNOTSUPP):
                with patch('fcntl.flock',side_effect=OSError(code,'unsupported')):
                    with worker_locks(root,work,Path(td)/'worker0',0) as first:
                        with worker_locks(root,work,Path(td)/'worker1',1) as second:
                            self.assertNotEqual(first,second)
