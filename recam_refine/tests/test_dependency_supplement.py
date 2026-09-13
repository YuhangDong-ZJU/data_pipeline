"""Additive installation gates protect installed packages and shared workers."""
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from recam_refine.dependencies import (environment_lease, fetch_wheel, installer_environment,
    missing_requirements, pip_command, plan_requirements, supplement, validate_plan)
from recam_refine.environment import profile_packages


class SupplementTests(unittest.TestCase):
    def test_missing_only_pins_preserve_existing_versions(self):
        snapshot = dict(python_version=[3, 10], packages={'numpy': '2.2.6', 'torch': '2.8.0+cu129'})
        wanted = profile_packages('prepare-gpu')
        missing = missing_requirements(snapshot, wanted, 'prepare-gpu')
        self.assertNotIn('torch', missing)
        self.assertNotIn('numpy', missing)
        pins = plan_requirements(snapshot, wanted, 'prepare-gpu')
        self.assertIn('numpy==2.2.6', pins)
        self.assertIn('torch==2.8.0+cu129', pins)
        self.assertIn('pycollada==0.9.2', pins)

    def test_existing_torch_and_python_never_replaced(self):
        for version in ('2.0.1', '2.8.0+cu128', '2.9.0'):
            with self.assertRaisesRegex(RuntimeError, 'preserved'):
                missing_requirements(dict(python_version=[3, 10], packages={'torch': version}),
                                     profile_packages('gpu'), 'gpu')
        with self.assertRaisesRegex(RuntimeError, 'not be replaced'):
            missing_requirements(dict(python_version=[3, 12], packages={}), profile_packages('base'), 'base')
        missing_requirements(dict(python_version=[3, 10], packages={'torch': '2.8.0+cu129'}),
                             profile_packages('cpu'), 'cpu')

    def test_plan_rejects_installed_packages_unhashed_and_source_archives(self):
        def row(name='trimesh', url='https://example.org/trimesh-4.6.4-py3-none-any.whl', sha='a'*64):
            return dict(metadata=dict(name=name, version='4.6.4'), download_info=dict(url=url,
                        archive_info=dict(hashes=dict(sha256=sha))))
        self.assertEqual(len(validate_plan({'install': [row()]}, {})), 1)
        for rows, installed in (([row('torch')], {'torch': '2.8.0+cu129'}),
                                ([row(url='https://example.org/a.tar.gz')], {}),
                                ([row(sha='')], {}), ([row(), row()], {})):
            with self.assertRaises(RuntimeError):
                validate_plan({'install': rows}, installed)

    def test_hash_failure_never_publishes_wheel_and_valid_cache_is_offline(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            source, dest = folder/'source.whl', folder/'wheel.whl'
            source.write_bytes(b'fixture wheel')
            with self.assertRaisesRegex(RuntimeError, 'mismatch'):
                fetch_wheel(source.as_uri(), 'a'*64, dest)
            self.assertFalse(dest.exists())
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            fetch_wheel(source.as_uri(), digest, dest)
            with patch('urllib.request.urlopen', side_effect=AssertionError('Network used')):
                fetch_wheel(source.as_uri(), digest, dest)

    def test_pip_routing_cannot_mutate_another_prefix(self):
        result = installer_environment(dict(PIP_TARGET='/other', PIP_USER='1', PIP_PREFIX='/other',
            PIP_UPGRADE='1', PIP_TRUSTED_HOST='bad', PIP_INDEX_URL='https://mirror/simple',
            HTTPS_PROXY='http://site-proxy', REQUESTS_CA_BUNDLE='/site/ca.pem'), Path('/work'))
        for name in ('PIP_TARGET', 'PIP_USER', 'PIP_PREFIX', 'PIP_UPGRADE', 'PIP_TRUSTED_HOST'):
            self.assertNotIn(name, result)
        self.assertEqual(result['PIP_INDEX_URL'], 'https://mirror/simple')
        self.assertEqual(result['REQUESTS_CA_BUNDLE'], '/site/ca.pem')
        self.assertEqual(result['PIP_CONFIG_FILE'], os.devnull)

    def test_standalone_pip_does_not_install_or_upgrade_environment_pip(self):
        with patch('recam_refine.dependencies.fetch_wheel') as download:
            command = pip_command('/env/bin/python', {'packages': {}}, Path('/work'))
        self.assertEqual(download.call_count, 1)
        self.assertEqual(command[0], '/env/bin/python')
        self.assertEqual(command[-1], '/work/pip-25.2-py3-none-any.whl')
        self.assertNotIn('install', command)

    def test_environment_context_never_uses_file_locks(self):
        with tempfile.TemporaryDirectory() as td, patch('fcntl.flock',side_effect=AssertionError('File lock')):
            work = Path(td)
            with environment_lease(work, '/env'), environment_lease(work, '/env', exclusive=True):
                pass
            self.assertEqual(list(work.iterdir()),[])

    def test_system_python_refused_before_pip_or_download(self):
        with patch('recam_refine.dependencies.pip_command', side_effect=AssertionError('Installer called')):
            with self.assertRaisesRegex(RuntimeError, 'system Python'):
                supplement('/usr/bin/python3', dict(prefix='/usr', base_prefix='/usr'),
                           {}, 'base', Path('/tmp/cache'), {})
