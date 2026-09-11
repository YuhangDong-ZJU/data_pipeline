"""Publish external-depth TARs after cleanup, retaining training PNGs."""
from __future__ import annotations

from .progress import tracked
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import stat
import tarfile

from .archives import frame_files, member_path
from .common import media_path, read_json, read_jsonl, require, safe_path, sha256, sync_dir, write_json
from .pipeline import discover
from .steps import MARKERS, freeze_settings, locked_step


SUCCESS = 'REPACK_SUCCESS.json'
BLOCK = 1024 * 1024
CAMERAS = ('observation.images.depth_01', 'observation.images.depth_02')


def receipt_file(work, relative):
    return work / 'repacked_depth' / (hashlib.sha256(relative.encode()).hexdigest() + '.json')


def snapshot(path):
    value = path.stat()
    require(stat.S_ISREG(value.st_mode) and not path.is_symlink(), f'Not a regular file: {path}')
    return (value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def checked_dataset(root, work):
    for stage in ('check', 'cleanup'):
        path = work / MARKERS[stage]
        require(path.is_file(), f'Run {stage} successfully before repack: missing {path.name}')
        record = read_json(path)
        require(record.get('complete') is True and record.get('full_decode') is True,
                f'Invalid {stage} completion record: {path}')
    check = read_json(work / MARKERS['check'])
    require(read_json(work / MARKERS['cleanup']).get('root') == str(root), 'Cleanup belongs to another dataset')
    require(not (work / 'camera_audit/QUALITY_REVIEW_REQUIRED.json').exists(), 'Geometry still requires review')
    require(sha256(work / 'camera_audit/summary.json') == check['geometry_summary_sha256'],
            'Geometry report changed after check')
    return {stage: sha256(work / MARKERS[stage]) for stage in ('check', 'cleanup')}


def source_files(subset, job):
    for entry in job['episodes']:
        directory = safe_path(subset, entry['directory'])
        files = frame_files(directory)
        require(len(files) == entry['length'], f'Depth/meta frame count differs: {directory}')
        for path in files:
            yield path


def packing_plan(subset, episodes_per_shard):
    info = read_json(subset / 'meta/info.json')
    rows = read_jsonl(subset / 'meta/episodes.jsonl')
    require(rows and [r['episode_index'] for r in rows] == list(range(info['total_episodes'])),
            'DROID episode catalogue is not complete and ordered')
    require(sum(r['length'] for r in rows) == info['total_frames'], 'DROID metadata frame totals differ')
    declared = {k for k in info['features'] if k.startswith('observation.images.depth_')}
    require(declared == set(CAMERAS) and all(info['features'][k]['dtype'] == 'image' for k in CAMERAS),
            'Expected exactly external depth_01/02 PNG features after cleanup')
    groups, directories = {}, set()
    for row in rows:
        i, n = row['episode_index'], row['length']
        require(n > 0, f'Empty episode: {i}')
        for camera in CAMERAS:
            directory = media_path(subset, info, i, camera, 0).parent
            relative = directory.relative_to(subset).as_posix()
            expected = f'images/chunk-{i // info["chunks_size"]:03d}/{camera}/episode_{i:06d}'
            require(relative == expected, f'Unsupported publishing image path: {relative}')
            require(relative not in directories, f'Duplicate depth directory: {relative}')
            directories.add(relative)
            groups.setdefault(directory.parent.relative_to(subset).as_posix(), []).append(
                dict(episode_index=i, directory=relative, length=n))
    actual = {p.relative_to(subset).as_posix() for p in (subset / 'images').glob('chunk-*/*/episode_*')
              if p.parent.name in CAMERAS}
    require(actual == directories, 'Missing or extra external depth episode directories')
    jobs = []
    for parent, episodes in sorted(groups.items()):
        for start in range(0, len(episodes), episodes_per_shard):
            batch = episodes[start:start + episodes_per_shard]
            first, last = batch[0]['episode_index'], batch[-1]['episode_index']
            relative = f'{parent}/episodes-{first:06d}-{last:06d}.tar'
            job = dict(path=relative, episodes=batch)
            jobs.append(job)
    expected = {job['path'] for job in jobs}
    existing = {p.relative_to(subset).as_posix() for p in (subset / 'images').glob('chunk-*/*/*.tar')
                if p.parent.name in CAMERAS}
    require(existing <= expected, 'Unexpected old depth TARs; preserve them and resolve before repack')
    return jobs


def expected_members(job):
    return {f"{e['directory']}/frame_{f:06d}.png" for e in job['episodes'] for f in range(e['length'])}


def verify_existing_tar(path, subset, job):
    """Exceptional recovery only: compare an unreceipted archive without hashing."""
    seen = set()
    expected = expected_members(job)
    with tarfile.open(path, 'r:') as tar:
        for member in tar:
            name = member_path(member.name, job['path'])
            require(member.isfile() and member.name == name and name in expected and name not in seen,
                    f'Unexpected TAR member: {member.name}')
            source = safe_path(subset, name)
            require(snapshot(source)[0] == member.size, f'TAR member size differs: {name}')
            with tar.extractfile(member) as incoming, source.open('rb') as original:
                while True:
                    a, b = incoming.read(BLOCK), original.read(BLOCK)
                    require(a == b, f'TAR content differs: {name}')
                    if not a:
                        break
            seen.add(name)
    require(seen == expected, f'Missing TAR members: {path}')


def pack_one(subset, work, job, plan_id):
    target = safe_path(subset, job['path'])
    receipt_path = receipt_file(work, job['path'])
    expected = expected_members(job)
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        require(receipt['plan_id'] == plan_id and receipt['path'] == job['path'] and receipt['complete'] is True,
                f'TAR receipt belongs to another plan: {target}')
        require(expected == set(receipt['members']), f'TAR receipt coverage differs: {target}')
        current = snapshot(target)
        state = receipt.get('file_state', receipt.get('verified_state'))
        if state is None:
            verify_existing_tar(target, subset, job)
            receipt['file_state'] = list(current)
            write_json(receipt_path, receipt)
        else:
            require(state == list(current), f'TAR changed after packing: {target}')
        print(f'SKIPPED 打包：已完成且归档未变化 {target}', flush=True)
        return dict(path=job['path'], sha256=receipt.get('sha256'), files=len(expected),
                    status='reused', state=current, content_hashes=False)
    entries = {}
    part = target.with_name('.' + target.name + '.repack-part')
    require(not part.is_symlink(), f'Linked temporary TAR: {part}')
    try:
        if target.exists():
            verify_existing_tar(target, subset, job)
            entries = {name:dict(size=snapshot(safe_path(subset, name))[0]) for name in expected}
            status = 'recovered'
        else:
            payload = 0
            with part.open('wb') as destination:
                with tarfile.open(fileobj=destination, mode='w', format=tarfile.USTAR_FORMAT,
                                  copybufsize=BLOCK) as archive:
                    for path in source_files(subset, job):
                        before = snapshot(path)
                        name = path.relative_to(subset).as_posix()
                        member = tarfile.TarInfo(name)
                        member.size, member.mode, member.mtime = before[0], 0o644, 0
                        with path.open('rb') as source:
                            archive.addfile(member, source)
                        require(snapshot(path) == before, f'PNG changed while packing: {path}')
                        entries[name] = dict(size=before[0])
                        payload += 512 + ((before[0] + 511) // 512) * 512
                destination.flush()
                os.fsync(destination.fileno())
            expected_size = ((payload + 1024 + 10239) // 10240) * 10240
            require(set(entries) == expected and snapshot(part)[0] == expected_size, f'Incomplete TAR write: {part}')
            require(not target.exists(), f'TAR appeared while packing; refusing to overwrite: {target}')
            os.replace(part, target)
            sync_dir(target.parent)
            status = 'created'
        write_json(receipt_path, dict(plan_id=plan_id, path=job['path'], complete=True,
                                     sha256=None, members=entries, file_state=list(snapshot(target)),
                                     content_hashes=False))
        return dict(path=job['path'], sha256=None, files=len(entries), status=status,
                    state=snapshot(target), content_hashes=False)
    finally:
        part.unlink(missing_ok=True)


def run_repack(args):
    root, work = args.root.resolve(), args.work_dir.resolve()
    require(args.workers > 0 and args.episodes_per_shard > 0, 'workers and episodes-per-shard must be positive')
    with locked_step(root, work):
        (work / SUCCESS).unlink(missing_ok=True)
        checked = checked_dataset(root, work)
        subset = root / 'real_world/droid'
        jobs = packing_plan(subset, args.episodes_per_shard)
        settings = dict(schema=2, root=str(root), checked=checked, episodes_per_shard=args.episodes_per_shard, jobs=jobs)
        previous = work/'step_settings/repack.json'
        if previous.exists():
            saved = read_json(previous)
            normalized = {**saved, 'schema':2, 'jobs':[{k:v for k,v in j.items() if k != 'tar_bytes'} for j in saved['jobs']]}
            require(normalized == settings, 'repack settings changed; resume with the original arguments')
            settings = saved
        freeze_settings(work, 'repack', settings)
        plan_id = hashlib.sha256(json.dumps(settings, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        # Reclaim only this frozen plan's reserved partial files after a crash.
        # Final TARs and source PNGs are never removed here.
        for job in jobs:
            target = safe_path(subset, job['path'])
            part = target.with_name('.' + target.name + '.repack-part')
            require(not part.is_symlink(), f'Linked temporary TAR: {part}')
            if part.exists():
                snapshot(part)
                part.unlink()
                sync_dir(part.parent)
        print(f'Repack: {len(jobs)} TARs; source PNGs retained; no content hashes', flush=True)
        results = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(pack_one, subset, work, job, plan_id) for job in jobs]
            try:
                for future in tracked(as_completed(futures),'打包：TAR',len(futures)):
                    result = future.result()
                    results.append(result)
                    print(f'Repack {len(results)}/{len(jobs)} {result["status"]}: {result["path"]}', flush=True)
            except Exception:
                for future in futures:
                    future.cancel()
                raise
        require({stage:sha256(work/MARKERS[stage]) for stage in ('check','cleanup')} == checked,
                'Check/cleanup records changed while packing')
        for result in results:
            require(snapshot(safe_path(subset, result['path'])) == result.pop('state'),
                    f'TAR changed before final publication: {result["path"]}')
        write_json(work / SUCCESS, dict(complete=True, root=str(root), plan_id=plan_id, checked=checked,
                                       source_pngs_retained=True, archives=sorted(results, key=lambda r: r['path'])))
        (work / 'REPACK_FAILED.json').unlink(missing_ok=True)
        print(f'REPACK COMPLETE: {work / SUCCESS}; TAR writes completed; source PNGs retained; content hashes not computed.', flush=True)
