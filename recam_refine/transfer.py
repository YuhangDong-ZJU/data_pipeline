"""Incremental atomic moves and verified copies, without a global preflight."""
import hashlib
import os
from pathlib import Path

from .common import Journal, read_json, require, safe_path, sha256, sync_dir, write_json
from .parallel import io_map, Counter
from functools import partial


def matches(path, entry, content=False):
    if not path.is_file() or path.is_symlink():
        return False
    if 'size' in entry and path.stat().st_size != entry['size']:
        return False
    return not (content and entry.get('sha256')) or sha256(path) == entry['sha256']


def _names(directory):
    return sorted(p.name for p in directory.iterdir() if not p.name.endswith('.refine-part'))


def _prepare_stream(task):
    i, cam, row, source, droid, work = task
    original = int(row.get("source_episode_index", i))
    src = safe_path(source, f'images/chunk-{original//1000:03d}/observation.images.depth_{cam:02d}/episode_{original:06d}')
    dst = safe_path(droid, f'images/chunk-{i//1000:03d}/observation.images.depth_{cam:02d}/episode_{i:06d}')
    receipt_path = work/'transfer_receipts'/f'episode_{i:06d}_{cam}.json'
    existing_receipt = receipt_path.exists()
    if existing_receipt:
        receipt = read_json(receipt_path)
        require(receipt['source'] == str(src) and receipt['target'] == str(dst), 'Transfer receipt path changed')
        names = [f'frame_{f:06d}.png' for f in range(len(receipt['files']))]
        require([e['name'] for e in receipt['files']] == names, f'Transfer receipt frame count/names changed: {receipt_path}')
    else:
        names = _names(src)
        require(names and names == [f'frame_{f:06d}.png' for f in range(len(names))],
                f'Missing/noncontiguous or extra depth frames: {src}')
        entries = [dict(name=name) for name in names]
        receipt = dict(version=3, source=str(src), target=str(dst), files=entries, complete=False)
        write_json(receipt_path, receipt)
    require(0 <= int(row['length']) - len(names) <= 2,
            f'Depth frame count differs from source manifest: {src}')
    if dst.exists():
        allowed = set(names)
        require(all(p.name in allowed or p.name.endswith('.refine-part') for p in dst.iterdir()),
                f'Extra target files: {dst}')
    return (i, cam, src, dst, receipt_path, len(receipt["files"]))


def counted(entries, counter):
    for entry in entries:
        yield entry
        counter.tick()


def _move(job, root, work, source_device, counter):
    i, cam, src, dst, receipt_path, count = job
    receipt = read_json(receipt_path)
    journal = Journal(root, work)
    dst.parent.mkdir(parents=True, exist_ok=True)
    same_fs = source_device == dst.parent.stat().st_dev
    renamed = False
    if not receipt['complete'] and src.exists() and same_fs and not dst.exists():
        os.replace(src, dst)
        sync_dir(src.parent)
        sync_dir(dst.parent)
        renamed = True
    dst.mkdir(parents=True, exist_ok=True)
    fast_path = renamed or receipt['complete'] or (same_fs and not src.exists())
    for entry in counted([] if fast_path else receipt["files"], counter):
        p, target = src/entry['name'], dst/entry['name']
        if target.exists() and entry.get('sha256') and matches(target, entry, content=True):
            continue
        if not p.exists():
            # Resume an atomic rename interrupted before the completion receipt.
            require(same_fs and matches(target, entry, content=True), f'Missing source/target: {p}')
            continue
        require(matches(p, entry, content=same_fs), f'Source changed: {p}')
        if same_fs:
            journal.backup(target)
            os.replace(p, target)
            sync_dir(p.parent)
            sync_dir(target.parent)
        else:
            staged = target.with_name('.' + target.name + '.refine-part')
            digest = hashlib.sha256()
            with p.open('rb') as incoming, staged.open('wb') as outgoing:
                while block := incoming.read(1024 * 1024):
                    outgoing.write(block)
                    digest.update(block)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            require(matches(staged, entry) and sha256(staged) == digest.hexdigest(), f'Copy verification failed: {p}')
            require(not entry.get('sha256') or entry['sha256'] == digest.hexdigest(), f'Source changed since receipt: {p}')
            entry['sha256'] = digest.hexdigest()
            # Persist copy evidence before publishing the staged file.
            write_json(receipt_path, receipt)
            journal.replace(target, staged)
    require(_names(dst) == [e['name'] for e in receipt['files']], f'Extra/missing target frames: {dst}')
    if fast_path:
        counter.tick(len(receipt['files']))
    receipt['complete'] = True
    write_json(receipt_path, receipt)
    return dict(source=str(src), target=str(dst), episode_index=i, camera=cam,
                        frame_count=len(receipt['files']), same_filesystem=same_fs)


def transfer_depth(root, droid, source, manifest, chunks, work, workers=8):
    source = Path(source).resolve()
    require(source.is_dir() and not source.is_relative_to(root) and not root.is_relative_to(source),
            'Depth output and dataset must be separate')
    selected = [i for i in sorted(manifest) if int(manifest[i].get('source_episode_index', i)) // 1000 in chunks]
    # Each task owns one camera/episode directory and one receipt.
    require(len({int(manifest[i].get('source_episode_index', i)) for i in selected}) == len(selected),
            'Duplicate source episodes in transfer mapping')
    tasks = [(i, cam, manifest[i], source, droid, work) for i in selected for cam in (1, 2)]
    counter = Counter()
    function = partial(_move, root=root, work=work, source_device=source.stat().st_dev, counter=counter)
    def migrate(task):
        return function(_prepare_stream(task))
    results = io_map(migrate, tasks, workers, '并发迁移（相机序列）',
                     lambda: f'已处理 PNG={counter.value}；并发={workers}')
    require(results, f'No episodes selected in depth chunks {chunks}')
    write_json(work/'depth_transfer.json', results)
