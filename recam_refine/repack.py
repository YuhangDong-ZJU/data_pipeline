"""Publish verified external-depth TARs after cleanup, retaining training PNGs."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import shutil
import stat
import tarfile

from .archives import frame_files, member_path
from .common import media_path, read_json, read_jsonl, require, safe_path, sha256, sync_dir, write_json
from .pipeline import discover
from .steps import MARKERS, freeze_settings, locked_step, training_signature


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
    print('Repack: checking final training file state', flush=True)
    require(training_signature(root, discover(root)) == check['training_signature'],
            'Training files changed after check; do not repack an unverified dataset')
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
            # PNG payloads are already compressed. Account for TAR headers,
            # 512-byte padding and the final 10240-byte TAR record.
            payload = sum(512 + ((snapshot(p)[0] + 511) // 512) * 512 for p in source_files(subset, job))
            job['tar_bytes'] = ((payload + 1024 + 10239) // 10240) * 10240
            jobs.append(job)
    expected = {job['path'] for job in jobs}
    existing = {p.relative_to(subset).as_posix() for p in (subset / 'images').glob('chunk-*/*/*.tar')
                if p.parent.name in CAMERAS}
    require(existing <= expected, 'Unexpected old depth TARs; preserve them and resolve before repack')
    return jobs


class HashReader:
    def __init__(self, source):
        self.source = source
        self.hash = hashlib.sha256()
        self.size = 0

    def read(self, size=-1):
        data = self.source.read(size)
        self.hash.update(data)
        self.size += len(data)
        return data


def verify_tar(path, entries, relative):
    """Read every member and every archive byte; never extract to the dataset."""
    before = snapshot(path)
    seen = set()
    with path.open('rb') as source:
        reader = HashReader(source)
        with tarfile.open(fileobj=reader, mode='r|', bufsize=BLOCK) as archive:
            for member in archive:
                require(member.isfile(), f'Unexpected non-file TAR member: {member.name}')
                name = member_path(member.name, relative)
                require(member.name == name and name in entries and name not in seen,
                        f'Unexpected or duplicate TAR member: {member.name}')
                require(member.size == entries[name]['size'], f'TAR member size differs: {name}')
                digest, size = hashlib.sha256(), 0
                with archive.extractfile(member) as content:
                    for block in iter(lambda: content.read(BLOCK), b''):
                        digest.update(block)
                        size += len(block)
                require(size == member.size and digest.hexdigest() == entries[name]['sha256'],
                        f'TAR member SHA-256 differs: {name}')
                seen.add(name)
        # The streaming TAR reader can stop at its end marker before EOF.
        # Drain the underlying reader to hash the entire archive, including padding.
        for _ in iter(lambda: reader.read(BLOCK), b''):
            pass
    require(seen == set(entries), f'Missing TAR members: {path}')
    require(snapshot(path) == before and reader.size == before[0], f'TAR changed while verifying: {path}')
    return reader.hash.hexdigest()


def pack_one(subset, work, job, plan_id):
    target = safe_path(subset, job['path'])
    receipt_path = receipt_file(work, job['path'])
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        require(receipt['plan_id'] == plan_id and receipt['path'] == job['path'] and receipt['complete'] is True,
                f'TAR receipt belongs to another plan: {target}')
        require(target.is_file(), f'Completed TAR is missing: {target}')
        expected_names = {p.relative_to(subset).as_posix() for p in source_files(subset, job)}
        require(expected_names == set(receipt['members']), f'TAR receipt coverage differs: {target}')
        digest = verify_tar(target, receipt['members'], job['path'])
        require(digest == receipt['sha256'], f'TAR changed after packing: {target}')
        return dict(path=job['path'], sha256=digest, files=len(expected_names), status='verified', state=snapshot(target))

    entries = {}
    part = target.with_name('.' + target.name + '.repack-part')
    require(not part.is_symlink(), f'Linked temporary TAR: {part}')
    try:
        if target.exists():
            # Recover a crash after atomic publication but before the receipt.
            # Existing archives are accepted only after comparing every PNG byte.
            for path in source_files(subset, job):
                before = snapshot(path)
                digest = sha256(path)
                require(snapshot(path) == before, f'PNG changed while hashing: {path}')
                entries[path.relative_to(subset).as_posix()] = dict(size=before[0], sha256=digest)
            digest = verify_tar(target, entries, job['path'])
            status = 'recovered'
        else:
            with part.open('wb') as destination:
                with tarfile.open(fileobj=destination, mode='w', format=tarfile.USTAR_FORMAT,
                                  copybufsize=BLOCK) as archive:
                    for path in source_files(subset, job):
                        before = snapshot(path)
                        name = path.relative_to(subset).as_posix()
                        member = tarfile.TarInfo(name)
                        member.size, member.mode, member.mtime = before[0], 0o644, 0
                        with path.open('rb') as source:
                            reader = HashReader(source)
                            archive.addfile(member, reader)
                        require(reader.size == before[0] and snapshot(path) == before,
                                f'PNG changed while packing: {path}')
                        entries[name] = dict(size=reader.size, sha256=reader.hash.hexdigest())
                destination.flush()
                os.fsync(destination.fileno())
            digest = verify_tar(part, entries, job['path'])
            require(not target.exists(), f'TAR appeared while packing; refusing to overwrite: {target}')
            os.replace(part, target)
            sync_dir(target.parent)
            status = 'created'
        write_json(receipt_path, dict(plan_id=plan_id, path=job['path'], complete=True,
                                     sha256=digest, members=entries))
        return dict(path=job['path'], sha256=digest, files=len(entries), status=status, state=snapshot(target))
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
        settings = dict(schema=1, root=str(root), checked=checked, episodes_per_shard=args.episodes_per_shard, jobs=jobs)
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
        required = sum(job['tar_bytes'] for job in jobs if not safe_path(subset, job['path']).exists())
        manifests = sum(4096 + 320 * sum(e['length'] for e in job['episodes'])
                        for job in jobs if not receipt_file(work, job['path']).exists())
        margin = 64 * 1024 * 1024
        if subset.stat().st_dev == work.stat().st_dev:
            require(shutil.disk_usage(subset).free >= required + manifests + margin,
                    f'Insufficient free disk space for TARs and manifests: need {(required + manifests + margin) / 1024**3:.2f} GiB')
        else:
            require(shutil.disk_usage(subset).free >= required + margin,
                    f'Insufficient free disk space for new TARs: need {(required + margin) / 1024**3:.2f} GiB')
            require(shutil.disk_usage(work).free >= manifests + margin,
                    f'Insufficient free disk space in work directory for manifests: need {(manifests + margin) / 1024**3:.2f} GiB')
        print(f'Repack: {len(jobs)} TARs; {sum(e["length"] for j in jobs for e in j["episodes"])} PNGs; '
              f'new TAR space {required / 1024**3:.2f} GiB; source PNGs retained', flush=True)
        results = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(pack_one, subset, work, job, plan_id) for job in jobs]
            try:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    print(f'Repack {len(results)}/{len(jobs)} {result["status"]}: {result["path"]}', flush=True)
            except Exception:
                for future in futures:
                    future.cancel()
                raise
        require(checked_dataset(root, work) == checked, 'Check/cleanup records changed while packing')
        for result in results:
            require(snapshot(safe_path(subset, result['path'])) == result.pop('state'),
                    f'TAR changed before final publication: {result["path"]}')
        write_json(work / SUCCESS, dict(complete=True, root=str(root), plan_id=plan_id, checked=checked,
                                       source_pngs_retained=True, archives=sorted(results, key=lambda r: r['path'])))
        (work / 'REPACK_FAILED.json').unlink(missing_ok=True)
        print(f'REPACK COMPLETE: {work / SUCCESS}; all TAR member hashes verified; source PNGs retained.', flush=True)
