"""Metadata preflight, atomic moves, and verified streaming copies."""
import hashlib
import os
from pathlib import Path

from .archives import check_png, frame_files
from .common import Journal, read_json, require, safe_path, sha256, sync_dir, write_json
from .inputs import load_depth_records
from .progress import tracked


def matches(path, entry, content=False):
    if not path.is_file() or path.is_symlink():
        return False
    if 'size' in entry and path.stat().st_size != entry['size']:
        return False
    return not (content and entry.get('sha256')) or sha256(path) == entry['sha256']


def transfer_depth(root, droid, source, manifest, chunks, work):
    source = Path(source).resolve()
    require(source.is_dir() and not source.is_relative_to(root) and not root.is_relative_to(source),
            'Depth output and dataset must be separate')
    records = load_depth_records([source], manifest)
    journal = Journal(root, work)
    jobs = []
    selected = [i for i in sorted(manifest) if int(manifest[i].get('source_episode_index', i)) // 1000 in chunks]
    for i in tracked(selected, '轻量迁移预检（episode）'):
        original = int(manifest[i].get('source_episode_index', i))
        for cam in (1, 2):
            src = safe_path(source, f'images/chunk-{original//1000:03d}/observation.images.depth_{cam:02d}/episode_{original:06d}')
            dst = safe_path(droid, f'images/chunk-{i//1000:03d}/observation.images.depth_{cam:02d}/episode_{i:06d}')
            require((i, cam) in records, f'Source depth sidecar missing: {i}/{cam}')
            names = [f'frame_{f:06d}.png' for f in range(records[i, cam]['frame_count'])]
            receipt_path = work/'transfer_receipts'/f'episode_{i:06d}_{cam}.json'
            if receipt_path.exists():
                receipt = read_json(receipt_path)
                require(receipt['source'] == str(src) and receipt['target'] == str(dst), 'Transfer receipt path changed')
                require([e['name'] for e in receipt['files']] == names, f'Transfer receipt frame count/names changed: {receipt_path}')
            else:
                files = frame_files(src)
                require([p.name for p in files] == names, f'Incomplete depth output: {src}')
                # Full PNG decoding belongs to final validation. Sample both ends here.
                for p in dict.fromkeys((files[0], files[-1])):
                    check_png(p, (720, 1280, 1))
                entries = [dict(name=p.name, size=p.stat().st_size) for p in files]
                require(all(e['size'] > 0 for e in entries), f'Empty depth file: {src}')
                receipt = dict(version=2, source=str(src), target=str(dst), files=entries, complete=False)
                write_json(receipt_path, receipt)
            if dst.exists():
                allowed = set(names)
                require(all(p.name in allowed or p.name.endswith('.refine-part') for p in dst.iterdir()),
                        f'Extra target files: {dst}')
            for entry in receipt['files']:
                require(matches(dst/entry['name'], entry) or matches(src/entry['name'], entry),
                        f'Missing source/target frame: {src/entry["name"]}')
            jobs.append((i, cam, src, dst, receipt_path, receipt))
    results = []
    for i, cam, src, dst, receipt_path, receipt in tracked(jobs, '迁移深度（相机序列）'):
        dst.parent.mkdir(parents=True, exist_ok=True)
        same_fs = source.stat().st_dev == dst.parent.stat().st_dev
        renamed = False
        if not receipt['complete'] and src.exists() and same_fs and not dst.exists():
            os.replace(src, dst)
            sync_dir(src.parent)
            sync_dir(dst.parent)
            renamed = True
        dst.mkdir(parents=True, exist_ok=True)
        for entry in tracked(receipt['files'], f'迁移 episode_{i:06d}/depth_{cam:02d}（PNG）'):
            p, target = src/entry['name'], dst/entry['name']
            if renamed or receipt['complete']:
                require(matches(target, entry), f'Completed target missing/size changed: {target}')
                continue
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
        require([p.name for p in frame_files(dst)] == [e['name'] for e in receipt['files']], f'Extra/missing target frames: {dst}')
        receipt['complete'] = True
        write_json(receipt_path, receipt)
        results.append(dict(source=str(src), target=str(dst), episode_index=i, camera=cam,
                            frame_count=len(receipt['files']), same_filesystem=same_fs))
    require(results, f'No episodes selected in depth chunks {chunks}')
    write_json(work/'depth_transfer.json', results)
