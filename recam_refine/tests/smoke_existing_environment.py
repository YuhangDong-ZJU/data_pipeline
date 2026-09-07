"""Run the real shell entry using an existing Python, with networking disabled."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

from recam_refine.common import read_json
from recam_refine.steps import MARKERS
from recam_refine.tests.test_steps import all_files, manual_fixture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--python', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix='recam_reuse_cli_') as td:
        fixture, droid, archives = manual_fixture(Path(td))
        env = dict(os.environ, HTTPS_PROXY='http://127.0.0.1:1', HTTP_PROXY='http://127.0.0.1:1',
                   HF_HUB_OFFLINE='1', CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
        def shell(stage, *extra, python=args.python, success=True):
            command = ['bash', 'recam_refine/run_step.sh', stage, str(fixture.root), str(fixture.work_dir),
                       '--python', str(python), *extra]
            result = subprocess.run(command, cwd=repo, env=env, capture_output=True, text=True, timeout=180)
            print(result.stdout, flush=True)
            assert (result.returncode == 0) == success, (result.returncode, result.stdout, result.stderr)
        before = all_files(droid)
        shell('transfer', '--depth-output', str(fixture.depth_output), '--depth-chunks', '0',
              '--episode-manifest', str(fixture.episode_manifest), python=Path(td)/'absent', success=False)
        assert all_files(droid) == before
        assert not (fixture.work_dir/MARKERS['transfer']).exists()
        shell('transfer', '--depth-output', str(fixture.depth_output), '--depth-chunks', '0',
              '--episode-manifest', str(fixture.episode_manifest))
        shell('unpack', '--prepared-runtime')  # Existing scripts can retain this verification-only flag.
        assert all(p.exists() for p in archives)
        shell('align', '--episode-manifest', str(fixture.episode_manifest), '--workers', '2')
        assert read_json(droid/'meta/info.json')['total_frames'] == 12
        assert not (fixture.work_dir/'runtime').exists(), 'A runtime was installed during reuse'
        assert (fixture.work_dir/'runtime_cache/environment_base.json').is_file()
        print('Existing environment smoke passed: no network/install, real transfer/unpack/align, failure preserves data.')


if __name__ == '__main__':
    main()
