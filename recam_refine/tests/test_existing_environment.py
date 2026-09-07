"""Existing environments are selected explicitly and never silently replaced."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from recam_refine.environment import candidates, environment, freeze_gpu, select


class ExistingEnvironmentTests(unittest.TestCase):
    def test_explicit_python_preserves_venv_symlink_and_skips_conda(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            target = root/'base-python'
            target.touch()
            alias = root/'env/bin/python'
            alias.parent.mkdir(parents=True)
            alias.symlink_to(target)
            with patch('recam_refine.environment.conda_executable', side_effect=AssertionError('Conda queried')):
                self.assertEqual(candidates(root, 'base', python=str(alias)), [str(alias)])

    def test_conda_names_and_explicit_missing_environment(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            response = subprocess.CompletedProcess([], 0, json.dumps({'envs': [
                str(root/'envs/recam_data_pipeline'), str(root/'envs/droid_normals')]}), '')
            with patch.dict(os.environ, {}, clear=True), \
                 patch('recam_refine.environment.conda_executable', return_value='/conda'), \
                 patch('recam_refine.environment.subprocess.run', return_value=response) as run:
                found = candidates(root, 'gpu', conda_env='droid_normals')
                self.assertEqual(found, [str(root/'envs/droid_normals/bin/python')])
                self.assertEqual(run.call_args.args[0], ['/conda', 'env', 'list', '--json'])
                with self.assertRaisesRegex(RuntimeError, 'nothing was installed'):
                    candidates(root, 'base', conda_env='absent')

    def test_missing_explicit_python_never_falls_back_or_installs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with patch('recam_refine.environment.subprocess.run', side_effect=AssertionError('Executed subprocess')):
                with self.assertRaisesRegex(RuntimeError, 'No installation or upgrade'):
                    select(root, 'base', root/'cache', python=str(root/'missing'))
            self.assertEqual(list(root.iterdir()), [])

    def test_gpu_workers_may_use_different_prefixes_but_not_versions(self):
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            first = dict(python='/env-a/bin/python', python_version='3.10.20', packages={'torch': '2.8.0+cu129'})
            freeze_gpu(work, first)
            freeze_gpu(work, {**first, 'python': '/env-b/bin/python'})
            original = (work/'existing_gpu_environment.json').read_bytes()
            with self.assertRaisesRegex(RuntimeError, 'changed between workers'):
                freeze_gpu(work, {**first, 'packages': {'torch': '2.8.0+cu128'}})
            self.assertEqual((work/'existing_gpu_environment.json').read_bytes(), original)

    def test_network_configuration_survives_and_caches_are_separate(self):
        with patch.dict(os.environ, {'HTTPS_PROXY': 'http://proxy:123', 'REQUESTS_CA_BUNDLE': '/site/ca.pem',
                                     'LD_LIBRARY_PATH': '/site/lib', 'PYTHONPATH': '/other/python'}):
            env = environment('/installed/bin/python', Path('/worker/cache'))
        self.assertEqual(env['HTTPS_PROXY'], 'http://proxy:123')
        self.assertEqual(env['REQUESTS_CA_BUNDLE'], '/site/ca.pem')
        self.assertEqual(env['LD_LIBRARY_PATH'], '/site/lib')
        self.assertNotIn('PYTHONPATH', env)
        self.assertEqual(env['PYTHONDONTWRITEBYTECODE'], '1')
        self.assertEqual(env['CUDA_CACHE_PATH'], '/worker/cache/cuda')
