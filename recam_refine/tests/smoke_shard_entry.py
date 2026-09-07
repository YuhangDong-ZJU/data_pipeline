"""Delivered shell commands on disposable data and independent worker paths."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import subprocess
import tempfile

from recam_refine.steps import MARKERS
from recam_refine.tests.test_shards import fixture
from recam_refine.tests.test_steps import all_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime-work-dir',type=Path,required=True)
    parser.add_argument('--shared-runtime',action='store_true')
    parser.add_argument('--test-devices',default='cpu',help='Optional CUDA IDs for shared-runtime smoke, e.g. 0,1')
    args = parser.parse_args()
    runtime = (args.runtime_work_dir/'runtime').resolve()
    if args.test_devices!='cpu' and not args.shared_runtime:
        parser.error('--test-devices requires --shared-runtime')
    repo = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix='recam_shard_cli_') as td:
        case,droid,_ = fixture(Path(td),pending=(),devices=args.test_devices)
        (case.work_dir/'runtime').symlink_to(runtime,target_is_directory=True)
        before = all_files(droid)
        def shell(stage,*extra,success=True):
            code = subprocess.run(['bash','recam_refine/run_step.sh',stage,str(case.root),str(case.work_dir),*extra],cwd=repo).returncode
            assert (code==0)==success,(stage,code)
        shell('shard-plan','--num-shards','2','--devices',args.test_devices,'--iterations','1')
        shell('shard-merge',success=False)
        if args.shared_runtime:
            shared = Path(td)/'shared_env'
            shared.mkdir()
            (shared/'runtime').symlink_to(runtime,target_is_directory=True)
            # Two concurrent workers use exactly one prepared environment;
            # existing worker runtime contents must remain untouched.
            with ExitStack() as stack:
                processes = []
                for shard_id in (0,1):
                    local = Path(td)/f'worker{shard_id}'
                    (local/'runtime').mkdir(parents=True)
                    (local/'runtime/preserve.txt').write_text('existing worker environment')
                    log = stack.enter_context((Path(td)/f'worker{shard_id}.log').open('w+'))
                    device = 'cpu' if args.test_devices=='cpu' else args.test_devices.split(',')[shard_id%len(args.test_devices.split(','))]
                    cmd = ['bash','recam_refine/run_step.sh','shard-refine',str(case.root),str(case.work_dir),
                           '--shard-id',str(shard_id),'--worker-work-dir',str(local),'--devices',device,
                           '--runtime-work-dir',str(shared)]
                    # Shared selection itself enforces verify-only, even when
                    # the user omits the redundant --prepared-runtime flag.
                    processes.append((subprocess.Popen(cmd,cwd=repo,stdout=log,stderr=subprocess.STDOUT),log,local))
                for process,log,local in processes:
                    code = process.wait(timeout=180)
                    log.seek(0)
                    output = log.read()
                    print(output,flush=True)
                    assert code==0,(code,output)
                    assert (local/'step_shard-refine.log').is_file()
                    assert (local/'runtime_cache/cuda').is_dir()
                    assert (local/'runtime/preserve.txt').read_text()=='existing worker environment'
                    assert sorted(p.name for p in (local/'runtime').iterdir())==['preserve.txt']
            assert not (case.work_dir/'step_shard-refine.log').exists()
            assert not (shared/'step_shard-refine.log').exists()
            assert not (case.work_dir/MARKERS['refine']).exists()
            # Reuse of the coordinator as a GPU environment is rejected before
            # it can replace CPU Torch, as are missing runtime paths.
            local = Path(td)/'worker0'
            for invalid in (case.work_dir,local,Path(td)/'not_prepared'):
                shell('shard-refine','--shard-id','0','--worker-work-dir',str(local),'--devices',args.test_devices,
                      '--runtime-work-dir',str(invalid),success=False)
        for shard_id in (() if args.shared_runtime else (0,1)):
            local = Path(td)/f'worker{shard_id}'
            local.mkdir(exist_ok=True)
            (local/'runtime').symlink_to(runtime,target_is_directory=True)
            if shard_id==0:
                shell('shard-refine','--shard-id','0','--worker-work-dir',str(local),'--devices','cpu',
                      '--iterations','2',success=False)
            shell('shard-refine','--shard-id',str(shard_id),'--worker-work-dir',str(local),'--devices','cpu','--prepared-runtime')
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
        print('Shard Bash/CLI smoke passed: selected runtime, independent logs/cache, complete coverage, explicit merge/apply, checked cleanup.',flush=True)


if __name__=='__main__':
    main()
