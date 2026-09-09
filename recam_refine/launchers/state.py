"""Standard-library checks for the four machine-level entry scripts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

PATHS = ('RECAM_ROOT', 'RECAM_WORK', 'DEPTH_OUTPUT', 'WORKER_A', 'WORKER_B')
OPTIONS = ('ALIGN_WORKERS', 'CHECK_WORKERS', 'REPACK_WORKERS', 'AUDIT_FRAMES',
           'GPU_DEVICES', 'GPU_BATCH_SIZE', 'EPISODES_PER_TAR')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configuration():
    return {k: str(Path(os.environ[k]).resolve()) for k in PATHS} | {
        k: os.environ[k] for k in OPTIONS}


def code_digest():
    repo = Path(os.environ['REPO_DIR'])
    files = sorted([*repo.glob('run_*.sh'),
                    *(p for p in (repo/'recam_refine').rglob('*')
                      if p.suffix in ('.py', '.sh') and 'tests' not in p.parts)])
    return hashlib.sha256(json.dumps([(p.relative_to(repo).as_posix(), digest(p))
                                     for p in files]).encode()).hexdigest()


def validate_paths():
    config = configuration()
    paths = [Path(config[k]) for k in PATHS]
    if not (paths[0]/'real_world/droid').is_dir():
        raise ValueError('RECAM_ROOT/real_world/droid does not exist')
    for i, a in enumerate(paths):
        for b in paths[i+1:]:
            if a == b or a in b.parents or b in a.parents:
                raise ValueError(f'Data, depth output, work and worker directories must be separate: {a}; {b}')
    for key in OPTIONS:
        if key == 'GPU_DEVICES':
            devices = config[key].split(',')
            if not all(v.isdigit() for v in devices) or len(set(devices)) != len(devices):
                raise ValueError('GPU_DEVICES must contain distinct GPU indices')
        elif int(config[key]) < (0 if key == 'GPU_BATCH_SIZE' else 1):
            raise ValueError(f'Invalid {key}')


def marker(name):
    path = Path(os.environ['RECAM_WORK'])/name
    if not path.is_file():
        return False
    value = read(path)
    if not isinstance(value, dict):
        raise ValueError(f'Invalid completion record: {path}')
    if value.get('complete') is False:
        raise ValueError(f'Record explicitly reports incomplete work: {path}')
    if value.get('root') and Path(value['root']).resolve() != Path(os.environ['RECAM_ROOT']).resolve():
        raise ValueError(f'Completion record belongs to another dataset: {path}')
    return True


def scan_report():
    work = Path(os.environ['RECAM_WORK'])
    reports = sorted(work.glob('depth_timestamp_scan*.json'),
                     key=lambda p: p.stat().st_mtime_ns, reverse=True)
    if not reports:
        return None
    # Never fall back to an older report when the newest report is unsuitable.
    p = reports[0]
    data = read(p)
    expected = Path(os.environ['DEPTH_OUTPUT']).resolve()/'annotations/foundation_stereo_depth'
    if Path(data.get('root', '')).resolve() != expected:
        raise ValueError(f'Latest report has a different depth root: {p}')
    if set(data.get('by_chunk', {})) != {f'chunk-{i:03d}' for i in range(2,14)}:
        raise ValueError(f'Latest report must cover chunks 2-13: {p}')
    if (data.get('affected_episode_ids') != ['006795'] or
            data.get('duplicate_final_retry_episode_ids') != ['006795']):
        raise ValueError(f'Expected only the previously confirmed episode 6795; inspect report: {p}')
    return p.resolve()


def shard_done(shard):
    work = Path(os.environ['RECAM_WORK'])
    plan = read(work/'shards/plan.json')
    part = plan['shards'][shard]
    directory = work/f'shards/results/shard-{shard:05d}'
    if not (directory/'COMPLETE.json').exists():
        return False
    receipt = read(directory/'COMPLETE.json')
    if (receipt.get('complete') is not True or receipt['plan_id'] != plan['plan_id'] or
            receipt['shard_id'] != shard or receipt['manifest_sha256'] != part['sha256'] or
            digest(work/part['path']) != part['sha256']):
        raise ValueError('Completed shard identity or manifest changed')
    expected = {f'episode_{i:06d}.json' for i in part['episodes']}
    if expected != set(receipt['candidate_sha256']) or expected != {p.name for p in (directory/'cameras').glob('*.json')}:
        raise ValueError('Completed shard candidate coverage changed')
    for name, checksum in receipt['candidate_sha256'].items():
        if digest(directory/'cameras'/name) != checksum:
            raise ValueError(f'Completed shard candidate changed: {name}')
    return True


def ready(save=False):
    work = Path(os.environ['RECAM_WORK'])
    target = work/'launchers/PREPARE_READY.json'
    plan = read(work/'shards/plan.json')
    if ([p['shard_id'] for p in plan['shards']] != [0,1] or
            plan['settings']['backend'] != 'batched'):
        raise ValueError('Machine entry scripts require exactly shards 0/1 with the batched backend')
    if read(work/'SHARD_PLAN_READY.json')['plan_sha256'] != digest(work/'shards/plan.json'):
        raise ValueError('Shard plan does not match its completion record')
    if save:
        python = read(work/'runtime_cache/environment_cpu.json')['python']
        gpu = read(work/'runtime_cache/environment_prepare-gpu.json')['python']
        if Path(python).resolve() != Path(gpu).resolve():
            raise ValueError('CPU and GPU preparation must use the selected shared interpreter')
        value = dict(config=configuration(), code_sha256=code_digest(),
                     python=str(Path(python).resolve()),
                     plan_sha256=digest(work/'SHARD_PLAN_READY.json'))
        part = target.with_suffix('.part')
        part.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
        part.replace(target)
    value = read(target)
    if value['config'] != configuration():
        raise ValueError('Shared paths/options changed after preparation; restore the original configuration')
    if value['code_sha256'] != code_digest():
        raise ValueError('Processing code changed after preparation; do not update code during this run')
    if value['plan_sha256'] != digest(work/'SHARD_PLAN_READY.json'):
        raise ValueError('The fixed shard plan changed')
    if not Path(value['python']).is_file():
        raise ValueError(f"Prepared shared Python is not visible on this host: {value['python']}")
    return value['python']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('paths','marker','scan-report','ready','save-ready','python','shard-done'))
    p.add_argument('value', nargs='?')
    args = p.parse_args()
    if args.action == 'paths':
        validate_paths()
    elif args.action == 'marker':
        return 0 if marker(args.value) else 3
    elif args.action == 'shard-done':
        return 0 if shard_done(int(args.value)) else 3
    elif args.action == 'scan-report':
        report = scan_report()
        if report is None:
            return 3
        print(report)
    elif args.action == 'python':
        print(read(Path(os.environ['RECAM_WORK'])/f'runtime_cache/environment_{args.value}.json')['python'])
    else:
        print(ready(save=args.action == 'save-ready'))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
