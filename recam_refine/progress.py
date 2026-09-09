"""Standard-library progress reporting and a signal-aware stage heartbeat."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time


def phase(name, completed=None, total=None, detail=None):
    """Called by the coordinator only, not by parallel workers."""
    state = dict(phase=name, completed=completed, total=total, detail=detail,
                 updated_at=time.time())
    path = os.environ.get('RECAM_PROGRESS_STATE')
    if path:
        target = Path(path)
        temporary = target.with_suffix('.part')
        temporary.write_text(json.dumps(state), encoding='utf-8')
        os.replace(temporary, target)
    count = f' {completed}/{total}' if completed is not None and total is not None else ''
    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {name}{count}' +
          (f' | {detail}' if detail else ''), flush=True)


def run(command, label, log, interval=30, capture=False):
    if interval <= 0:
        raise ValueError('Heartbeat interval must be positive')
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(prefix='.progress-', suffix='.json', dir=log.parent)
    os.close(fd)
    state = Path(filename)
    state.write_text(json.dumps({'phase':'stage execution (see detailed output)'}), encoding='utf-8')
    started = time.monotonic()
    stop = threading.Event()
    env = dict(os.environ, RECAM_PROGRESS_CHILD='1', RECAM_PROGRESS_STATE=filename,
               PYTHONUNBUFFERED='1')
    def emit(message):
        line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] [{label}] {message}'
        print(line, flush=True)
        with log.open('a', encoding='utf-8') as stream:
            stream.write(line+'\n')
    process = None
    previous = {}
    def forward(signum, frame):
        if process is not None and process.poll() is None:
            try:
                if os.name == 'posix':
                    os.killpg(process.pid, signum)
                else:
                    process.terminate()
            except ProcessLookupError:
                pass
    def heartbeat():
        while not stop.wait(interval):
            if process.poll() is not None:
                return
            try:
                current = json.loads(state.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                current = {'phase':'waiting for stage status'}
            count = ''
            if current.get('total') is not None:
                count = f' completed={current.get("completed")}/{current["total"]}'
            emit(f'RUNNING pid={process.pid} elapsed={time.monotonic()-started:.0f}s '
                 f'phase={current["phase"]}{count} '
                 f'detail={current.get("detail") or "-"} '
                 '(process alive; does not prove work is advancing)')
    thread = None
    try:
        emit(f'START; heartbeat every {interval:g}s; detailed progress follows')
        process = subprocess.Popen(command, env=env, start_new_session=os.name=='posix',
                                   stdout=subprocess.PIPE if capture else None,
                                   stderr=subprocess.STDOUT if capture else None,
                                   text=True, encoding='utf-8', errors='replace')
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, forward)
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        if capture:
            for line in process.stdout:
                print(line, end='', flush=True)
                with log.open('a', encoding='utf-8') as stream:
                    stream.write(line)
            process.stdout.close()
        result = process.wait()
        stop.set()
        thread.join()
        code = result if result >= 0 else 128-result
        emit(f'{"SUCCESS" if code == 0 else "FAILED"} exit_code={code} elapsed={time.monotonic()-started:.1f}s')
        return code
    finally:
        stop.set()
        if thread:
            thread.join()
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        state.unlink(missing_ok=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label',required=True)
    p.add_argument('--log',required=True)
    p.add_argument('--interval',type=float,default=30)
    p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args()
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    if not command:
        p.error('Command required')
    return run(command,a.label,a.log,a.interval)


if __name__=='__main__':
    sys.exit(main())
