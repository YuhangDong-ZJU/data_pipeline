"""Delivered shell commands on disposable data and independent worker paths."""
import argparse
from pathlib import Path
import subprocess
import tempfile

from recam_refine.steps import MARKERS
from recam_refine.tests.test_shards import fixture
from recam_refine.tests.test_steps import all_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime-work-dir',type=Path,required=True)
    args = parser.parse_args()
    runtime = (args.runtime_work_dir/'runtime').resolve()
    repo = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix='recam_shard_cli_') as td:
        case,droid,_ = fixture(Path(td),pending=())
        (case.work_dir/'runtime').symlink_to(runtime,target_is_directory=True)
        before = all_files(droid)
        def shell(stage,*extra,success=True):
            code = subprocess.run(['bash','recam_refine/run_step.sh',stage,str(case.root),str(case.work_dir),*extra],cwd=repo).returncode
            assert (code==0)==success,(stage,code)
        shell('shard-plan','--num-shards','2','--devices','cpu','--iterations','1')
        shell('shard-merge',success=False)
        for shard_id in (0,1):
            local = Path(td)/f'worker{shard_id}'
            local.mkdir(exist_ok=True)
            (local/'runtime').symlink_to(runtime,target_is_directory=True)
            if shard_id==0:
                shell('shard-refine','--shard-id','0','--worker-work-dir',str(local),'--devices','cpu',
                      '--iterations','2',success=False)
            shell('shard-refine','--shard-id',str(shard_id),'--worker-work-dir',str(local),'--devices','cpu')
            assert not (case.work_dir/MARKERS['refine']).exists()
            assert (local/'step_shard-refine.log').exists()
            assert not (case.work_dir/'step_shard-refine.log').exists()
        shell('shard-status')
        shell('shard-merge')
        assert (case.work_dir/MARKERS['refine']).exists()
        assert all_files(droid)==before
        shell('apply','--workers','2')
        shell('shard-status')
        shell('cleanup',success=False)
        print('Shard Bash/CLI smoke passed: per-worker runtime/logs, complete coverage, explicit merge/apply, checked cleanup.',flush=True)


if __name__=='__main__':
    main()
