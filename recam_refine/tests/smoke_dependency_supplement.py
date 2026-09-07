"""Real CLI: install missing geometry wheels offline into a disposable existing venv."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from recam_refine.dependencies import (PIP_SHA256, PIP_URL, fetch_wheel, inventory,
    installer_environment, pip_command, plan_requirements)
from recam_refine.environment import environment, profile_packages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inherit', action='store_true', help='Inherit this interpreter\'s existing packages read-only')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix='recam_supplement_') as td:
        folder = Path(td)
        work = folder / 'work'
        cache = work / 'runtime_cache'
        cache.mkdir(parents=True)
        prefix = folder / 'existing_env'
        subprocess.run([sys.executable, '-m', 'venv', '--without-pip',
                        *(['--system-site-packages'] if args.inherit else []), str(prefix)], check=True)
        python = str(prefix / 'bin/python')
        env = environment(python, cache)
        pip_env = installer_environment(env, cache)
        before = inventory(python, env)
        pip = pip_command(python, before, cache)
        wanted = profile_packages('geometry')
        absent = {'trimesh', 'pycollada', 'zstandard'}
        if not args.inherit:
            baseline = {k: v for k, v in wanted.items() if k not in absent}
            subprocess.run([*pip, 'install', '--only-binary=:all:',
                            *plan_requirements(before, baseline, 'geometry')], env=pip_env, check=True)
        before = inventory(python, env, wanted)
        assert absent.isdisjoint(before['packages']), before
        wheelhouse = folder / 'offline_wheels'
        wheelhouse.mkdir()
        subprocess.run([*pip, 'download', '--only-binary=:all:', '-d', str(wheelhouse),
                        'trimesh==4.6.4', 'pycollada==0.9.2', 'zstandard==0.23.0'], env=pip_env, check=True)
        fetch_wheel(PIP_URL, PIP_SHA256, cache / 'pip-25.2-py3-none-any.whl')
        offline = dict(env, HTTPS_PROXY='http://127.0.0.1:1', HTTP_PROXY='http://127.0.0.1:1',
                       HF_HUB_OFFLINE='1', CUDA_VISIBLE_DEVICES='', PIP_NO_INDEX='1',
                       PIP_FIND_LINKS=str(wheelhouse), PIP_RETRIES='0')
        command = [sys.executable, '-m', 'recam_refine.environment', str(work),
                   '--python', python, '--profile', 'geometry']
        denied = subprocess.run([*command, '--check-only'], cwd=repo, env=offline,
                                capture_output=True, text=True, timeout=180)
        assert denied.returncode != 0 and 'No installation' in denied.stderr, denied
        assert inventory(python, env)['packages'] == before['packages']
        subprocess.run(command, cwd=repo, env=offline, check=True, timeout=240)
        after = inventory(python, env)
        assert all(after['packages'].get(n) == v for n, v in before['packages'].items())
        assert absent <= after['packages'].keys()
        report = json.loads((cache / 'environment_geometry.json').read_text())
        assert len(report['supplement']['packages']) >= 3
        # A complete environment no longer needs pip, the wheel cache or network.
        offline['PIP_FIND_LINKS'] = str(folder / 'no_wheels_here')
        subprocess.run(command, cwd=repo, env=offline, check=True, timeout=180)
        assert inventory(python, env)['packages'] == after['packages']
        report = json.loads((cache / 'environment_geometry.json').read_text())
        assert 'supplement' not in report
        if not args.inherit:
            assert 'pip' not in after['packages'], 'Standalone installer modified environment pip'
        print('Supplement CLI passed: check-only preserves env, missing wheels added offline, '
              'all previous versions preserved, geometry probe passed, repeat uses no installer/network.')


if __name__ == '__main__':
    main()
