"""Prepared runtime entry supports concurrent readers without file locks."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile
import time

from recam_refine.bootstrap import runtime_lock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime-work-dir',type=Path,required=True)
    parser.add_argument('--reuse-env',action='store_true')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    root = (args.runtime_work_dir/'runtime').resolve()
    env = dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    with tempfile.TemporaryDirectory(prefix='recam_runtime_lease_') as td:
        ready,gate = Path(td)/'ready',Path(td)/'finish'
        cmd = ['python3','recam_refine/bootstrap.py',str(args.runtime_work_dir),'--verify-only','--exec']
        if args.reuse_env:
            cmd = ['python3','-m','recam_refine.environment',str(args.runtime_work_dir),
                   '--python',str(root/'env/bin/python'),'--exec']
        child = "import pathlib,sys,time\npathlib.Path(sys.argv[1]).touch()\nwhile not pathlib.Path(sys.argv[2]).exists(): time.sleep(.05)"
        process = subprocess.Popen([*cmd,'-c',child,str(ready),str(gate)],cwd=repo,env=env)
        try:
            deadline = time.monotonic()+60
            while not ready.exists():
                assert process.poll() is None,'Launcher exited before its child was ready'
                assert time.monotonic()<deadline,'Launcher timeout'
                time.sleep(.05)
            # A second reader can verify and execute while the first is alive.
            subprocess.run([*cmd,'-c',"print('Concurrent runtime reader passed')"],cwd=repo,env=env,check=True,timeout=60)
            assert process.poll() is None
        finally:
            gate.touch()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=15)
        assert process.returncode==0
        with runtime_lock(root):
            pass
    print('Runtime CLI passed: concurrent prepared readers without file locking.')


if __name__=='__main__':
    main()
