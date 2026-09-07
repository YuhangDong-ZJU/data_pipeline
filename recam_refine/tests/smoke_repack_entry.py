"""Linux Bash repack entry with a prepared CPU runtime and disposable data."""
import argparse
from pathlib import Path
import subprocess
import tempfile

from recam_refine.common import read_json, sha256
from recam_refine.repack import SUCCESS
from recam_refine.steps import MARKERS
from recam_refine.tests.test_repack import completed_fixture
from recam_refine.tests.test_steps import all_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime-work-dir', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix='recam_repack_cli_') as td:
        case, _ = completed_fixture(Path(td))
        (case.work_dir / 'runtime').symlink_to((args.runtime_work_dir / 'runtime').resolve(), target_is_directory=True)
        before = all_files(case.root)
        cmd = ['bash', 'recam_refine/run_step.sh', 'repack', str(case.root), str(case.work_dir),
               '--workers', '2', '--episodes-per-shard', '1', '--prepared-runtime']
        marker = case.work_dir / MARKERS['cleanup']
        saved = marker.read_bytes()
        marker.unlink()
        assert subprocess.run(cmd, cwd=repo).returncode != 0
        assert not (case.work_dir / SUCCESS).exists()
        marker.write_bytes(saved)
        for _ in range(2):
            assert subprocess.run(cmd, cwd=repo).returncode == 0
            assert read_json(case.work_dir / SUCCESS)['complete']
            assert len(read_json(case.work_dir / SUCCESS)['archives']) == 4
        assert not (case.work_dir / 'REPACK_FAILED.json').exists()
        for relative, digest in before.items():
            assert sha256(case.root / relative) == digest
        print('Repack Bash/CLI smoke passed: cleanup gate, verified TARs, resume, PNG retention.', flush=True)


if __name__ == '__main__':
    main()
