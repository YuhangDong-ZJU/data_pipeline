"""Counted stage progress, elapsed time and signal-aware execution."""
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

_STARTED = time.time()


def phase(name, completed=0, total=1, detail=None):
    """Called by the coordinator only, not by parallel workers."""
    state = dict(phase=name, completed=completed, total=total, detail=detail,
                 updated_at=time.time())
    path = os.environ.get('RECAM_PROGRESS_STATE')
    if path:
        target = Path(path)
        temporary = target.with_suffix('.part')
        temporary.write_text(json.dumps(state), encoding='utf-8')
        os.replace(temporary, target)
    elapsed = max(0, time.time()-float(os.environ.get('RECAM_PROGRESS_STARTED', _STARTED)))
    count = f' {completed}/{total}'
    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] RUNNING elapsed={elapsed:.0f}s phase={name} progress={count.strip()}' +
          (f' | {detail}' if detail else ''), flush=True)


def tracked(items, name, total=None):
    """Count processed items, including checked resume entries; publish at most once/s."""
    total = len(items) if total is None else total
    phase(name, 0, total)
    updated = time.monotonic()
    for count, item in enumerate(items, 1):
        yield item
        now = time.monotonic()
        if now-updated >= 1 or count == total:
            phase(name, count, total)
            updated = now


def mapped(pool, function, items, name):
    """Report completed futures while preserving input order in the returned list."""
    from concurrent.futures import as_completed
    futures = {pool.submit(function, item): i for i, item in enumerate(items)}
    results = [None]*len(items)
    for future in tracked(as_completed(futures), name, len(futures)):
        results[futures[future]] = future.result()
    return results


def run(command, label, log, interval=30, capture=False):
    if interval <= 0:
        raise ValueError('Heartbeat interval must be positive')
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(prefix='.progress-', suffix='.json', dir=log.parent)
    os.close(fd)
    state = Path(filename)
    state.write_text(json.dumps({'phase':'准备（准备任务）','completed':0,'total':1}), encoding='utf-8')
    started = time.monotonic()
    stop = threading.Event()
    env = dict(os.environ, RECAM_PROGRESS_CHILD='1', RECAM_PROGRESS_STATE=filename,
               PYTHONUNBUFFERED='1', RECAM_PROGRESS_STARTED=str(time.time()))
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
            emit(f'RUNNING elapsed={time.monotonic()-started:.0f}s '
                 f'phase={current["phase"]} progress={current.get("completed",0)}/{current.get("total",1)}' +
                 (f' | {current["detail"]}' if current.get('detail') else ''))
    thread = None
    try:
        emit('RUNNING elapsed=0s phase=准备（准备任务） progress=0/1')
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
