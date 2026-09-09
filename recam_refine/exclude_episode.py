"""Quarantine confirmed episode 6795 before transfer; preserve source identities.

The last episode fills the hole so LeRobot indices stay contiguous. Run offline.
An immutable plan and per-file backups permit rerunning the same command.
"""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import re
import shutil
import tarfile
import hashlib

import numpy as np
import pyarrow.parquet as pq

from .common import (Journal, require, read_json, read_jsonl, write_json, sha256,
                     parquet_path, set_values, values, acquire_directory_lock,
                     validate_lock_mount)
from .stats import table_stats, aggregate
from .progress import phase


def mapped_path(relative, old, new):
    parts = relative.parts
    return Path(*(f'chunk-{new//1000:03d}' if p == f'chunk-{old//1000:03d}'
                  else p.replace(f'episode_{old:06d}', f'episode_{new:06d}') for p in parts))


def episode_files(droid, episode):
    token = f'episode_{episode:06d}'
    found = []
    # Scan directory names, not millions of PNG filenames outside these episodes.
    for top in droid.iterdir():
        if not top.is_dir() or top.name == 'meta':
            continue
        for directory, dirs, files in os.walk(top):
            parent = Path(directory)
            matching_dirs = [name for name in dirs if name == token]
            for name in matching_dirs:
                base = parent/name
                found.extend(p for p in base.rglob('*') if p.is_file())
            dirs[:] = [name for name in dirs if not name.startswith('episode_')]
            for name in files:
                if re.match(re.escape(token) + r'(?:\.|$)', name):
                    found.append(parent/name)
    require(all(not p.is_symlink() and not any(a.is_symlink() for a in p.parents)
                for p in found), 'Symlinks in episode files are not supported')
    return sorted(found)


def execute(root, work, report):
    phase('排除：读取报告和恢复状态', detail=str(report))
    droid = root/'real_world/droid'
    area = work/'exclude_episode_006795'
    area.mkdir(parents=True, exist_ok=True)
    journal = Journal(root, area)
    plan_path = area/'plan.json'
    done = area/'SUCCESS.json'
    if done.exists():
        saved = read_json(done)
        require(saved['root'] == str(root), 'Dataset root changed')
        require(all(sha256(droid/p) == h for p,h in saved['metadata_hashes'].items()),
                'Dataset advanced after exclusion; do not rerun this preparation step')
        print('Episode 6795 already excluded; no changes.')
        return
    if not plan_path.exists():
        for marker in ('STEP1_DEPTH_TRANSFER_SUCCESS.json', '03_plan.complete.json',
                       'original_droid_episodes.json', 'configuration.json', 'unpacked.json'):
            require(not (work/marker).exists(), f'Exclusion must precede transfer/unpack/align: {marker}')
        require(not list((work/'transfer_receipts').glob('*.json')), 'Transfer receipts exist; cannot change identities')
        scan = read_json(report)
        require(scan['affected_episode_ids'] == ['006795'] and
                scan['duplicate_final_retry_episode_ids'] == ['006795'], 'Expected only confirmed episode 006795')
        anomalies = scan['anomalies']
        require(len(anomalies) == 1 and anomalies[0]['episode_index'] == 6795 and
                anomalies[0]['source_episode_id'] == 'IPRL+w026bb9b+2023-09-22-19h-01m-34s' and
                anomalies[0]['camera_role'] == 'external_1' and
                anomalies[0]['issues'] == ['unordered_timestamps'], 'Unexpected anomaly identity')
        from .scan_depth_timestamps import inspect
        current = inspect(Path(anomalies[0]['path']))
        require(current == anomalies[0], 'Source sidecar changed since scan')
        info = read_json(droid/'meta/info.json')
        episodes = read_jsonl(droid/'meta/episodes.jsonl')
        last = len(episodes)-1
        require(last > 6795 and [e['episode_index'] for e in episodes] == list(range(last+1)), 'Expected original contiguous catalogue')
        require(not any('source_episode_index' in e for e in episodes), 'Dataset already remapped')
        require(info['chunks_size'] == 1000 and info['codebase_version'] == 'v2.1', 'Unsupported layout')
        require(info['splits'] == {'train': f'0:{last+1}'}, 'Only original all-train split supported')
        require(episodes[6795]['length'] == 345, 'Unexpected episode 6795 length')
        allowed = {'info.json','episodes.jsonl','episodes_stats.jsonl','stats.json','tasks.jsonl','cameras.json','coordinates.json'}
        require({p.name for p in (droid/'meta').iterdir()} <= allowed, 'Unknown metadata; inspect before excluding')
        stats = read_jsonl(droid/'meta/episodes_stats.jsonl')
        require([s['episode_index'] for s in stats] == list(range(last+1)), 'Incomplete episode statistics')
        phase('排除：定位两个 episode 的媒体文件')
        paths = {str(i): episode_files(droid,i) for i in (6795,last)}
        for i in (6795,last):
            require(parquet_path(droid,info,i) in paths[str(i)], f'Missing Parquet: {i}')
        # Verify all Parquet inputs before changing anything.
        offset = 0
        phase('排除：检查原始 Parquet 索引',0,len(episodes))
        for number,e in enumerate(episodes,1):
            p = parquet_path(droid,info,e['episode_index'])
            t = pq.read_table(p, columns=['episode_index','index','frame_index'])
            require(len(t) == e['length'] and np.all(values(t['episode_index']) == e['episode_index']), f'Bad Parquet: {p}')
            require(np.array_equal(values(t['index']).ravel(),np.arange(offset,offset+len(t))), f'Bad global index: {p}')
            require(np.array_equal(values(t['frame_index']).ravel(),np.arange(len(t))), f'Bad frame index: {p}')
            offset += len(t)
            if number % 100 == 0 or number == len(episodes):
                phase('排除：检查原始 Parquet 索引',number,len(episodes),str(p))
        require(offset == info['total_frames'], 'Frame total mismatch')
        archives = []
        from .archives import member_path
        for chunk in sorted({6795//1000,last//1000}):
            for archive in sorted((droid/f'images/chunk-{chunk:03d}').glob('*/*.tar')):
                if archive.name.startswith('episode_'):
                    continue
                phase('排除：检查 TAR 成员',detail=str(archive))
                affected = False
                with tarfile.open(archive,'r:*') as tar:
                    for member in tar:
                        if member.isdir():
                            continue
                        require(member.isfile(), f'Unsafe archive: {archive}')
                        rel = Path(member_path(member.name,archive.relative_to(droid).as_posix()))
                        affected |= any(f'episode_{i:06d}' in rel.parts for i in (6795,last))
                if affected:
                    archives.append(dict(path=archive.relative_to(droid).as_posix(),sha256=sha256(archive)))
        hashed = {}
        for k,entries in paths.items():
            hashed[k] = []
            phase('排除：计算备份校验值',0,len(entries),f'episode {k}')
            for number,p in enumerate(entries,1):
                hashed[k].append(dict(path=p.relative_to(droid).as_posix(),sha256=sha256(p)))
                if number % 100 == 0 or number == len(entries):
                    phase('排除：计算备份校验值',number,len(entries),str(p))
        plan = dict(root=str(root), last=last, info=info, episodes=episodes, stats=stats, archives=archives, files=hashed)
        write_json(plan_path, plan)
    plan = read_json(plan_path)
    require(plan['root'] == str(root), 'Exclusion root changed')
    last, info = plan['last'], copy.deepcopy(plan['info'])
    # All media and sidecars from both touched episodes are backed up before replacement.
    for entries in plan['files'].values():
        phase('排除：备份并校验文件',0,len(entries))
        for number,entry in enumerate(entries,1):
            p = droid/entry['path']
            backup = journal.backup(p)
            require(backup.is_file() and sha256(backup) == entry['sha256'], f'Backup mismatch: {p}')
            if number % 100 == 0 or number == len(entries):
                phase('排除：备份并校验文件',number,len(entries),str(p))
    # Remove the bad episode from live paths (backups remain outside the dataset).
    retired = area/'retired.complete.json'
    if not retired.exists():
        phase('排除：移除已备份的异常 episode 文件')
        for entry in plan['files']['6795']:
            p = droid/entry['path']
            if p.exists():
                require(sha256(p) == entry['sha256'], f'Changed bad episode file: {p}')
                p.unlink()
        write_json(retired, dict(complete=True))
    relocated = area/'relocated.complete.json'
    if not relocated.exists():
        entries = plan['files'][str(last)]
        phase('排除：安装补位 episode',0,len(entries))
        for number,entry in enumerate(entries,1):
            rel = Path(entry['path'])
            dst = droid/mapped_path(rel,last,6795)
            src = area/'original'/droid.relative_to(root)/rel
            dst.parent.mkdir(parents=True,exist_ok=True)
            staged = dst.with_name(dst.name+'.exclude-part')
            if rel.suffix == '.tar':
                from .archives import member_path
                with tarfile.open(src,'r:*') as inp, tarfile.open(staged,'w') as out:
                    for member in inp:
                        if member.isdir():
                            continue
                        require(member.isfile(), f'Unsupported archive member: {member.name}')
                        member_rel = Path(member_path(member.name,rel.as_posix()))
                        require(f'episode_{last:06d}' in member_rel.parts, 'Archive spans multiple episodes')
                        member.name = mapped_path(member_rel,last,6795).as_posix()
                        out.addfile(member,inp.extractfile(member))
            else:
                shutil.copy2(src,staged)
                require(sha256(staged) == entry['sha256'], f'Copy mismatch: {dst}')
            os.replace(staged,dst)
            if number % 100 == 0 or number == len(entries):
                phase('排除：安装补位 episode',number,len(entries),str(dst))
        write_json(relocated,dict(complete=True))
    # Original release TARs span many episodes. Remove bad/moved members so unpack
    # cannot resurrect either original identity; materialize the moved PNGs.
    from .archives import member_path, check_png
    for entry in plan['archives']:
        archive=droid/entry['path']
        phase('排除：备份或重写相关 TAR',detail=str(archive))
        receipt=area/'archives'/(hashlib.sha256(entry['path'].encode()).hexdigest()+'.json')
        if receipt.exists():
            require(sha256(archive)==read_json(receipt)['sha256'],'Rewritten TAR changed')
            continue
        backup=journal.backup(archive)
        require(sha256(backup)==entry['sha256'],'Archive backup changed')
        staged=archive.with_name(archive.name+'.exclude-part')
        expected={}
        with tarfile.open(backup,'r:*') as inp, tarfile.open(staged,'w') as out:
            for number,member in enumerate(inp,1):
                if number % 100 == 0:
                    phase('排除：重写当前 TAR（个）',0,1,detail=f'{archive}; 已读取成员={number}')
                if member.isdir():
                    continue
                rel=Path(member_path(member.name,entry['path']))
                if f'episode_{6795:06d}' in rel.parts:
                    continue
                if f'episode_{last:06d}' in rel.parts:
                    target=droid/mapped_path(rel,last,6795)
                    # Existing PNGs from the moved source outrank older archives.
                    if not target.exists():
                        target.parent.mkdir(parents=True,exist_ok=True)
                        temp=target.with_name(target.name+'.exclude-part')
                        with inp.extractfile(member) as src, temp.open('wb') as dst:
                            shutil.copyfileobj(src,dst)
                        check_png(temp)
                        os.replace(temp,target)
                    continue
                require(member.name not in expected,'Duplicate archive member')
                with inp.extractfile(member) as stream:
                    expected[member.name]=hashlib.file_digest(stream,'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hashlib.sha256(stream.read()).hexdigest()
                out.addfile(member,inp.extractfile(member))
        actual={}
        phase('排除：核对重写后的 TAR 内容',detail=str(archive))
        with tarfile.open(staged) as tar:
            for member in tar:
                with tar.extractfile(member) as stream:
                    actual[member.name]=hashlib.sha256(stream.read()).hexdigest()
        require(actual==expected,'Archive rewrite verification failed')
        os.replace(staged,archive)
        write_json(receipt,dict(sha256=sha256(archive)))
    episodes = copy.deepcopy(plan['episodes'][:-1])
    episodes[6795] = dict(plan['episodes'][last],episode_index=6795,source_episode_index=last)
    stats = copy.deepcopy(plan['stats'][:-1])
    stats[6795] = dict(copy.deepcopy(plan['stats'][last]),episode_index=6795)
    offset = 0
    phase('排除：更新全局索引与逐 episode 统计',0,len(episodes))
    for e,s in zip(episodes,stats):
        i,n = e['episode_index'],e['length']
        p = parquet_path(droid,info,i)
        t = pq.read_table(p)
        if not (np.all(values(t['episode_index']) == i) and
                np.array_equal(values(t['index']).ravel(),np.arange(offset,offset+n))):
            t = set_values(t,'episode_index',np.full(n,i))
            t = set_values(t,'index',np.arange(offset,offset+n))
            temp = p.with_name(p.name+'.exclude-part')
            pq.write_table(t,temp)
            require(pq.read_table(temp).equals(t), f'Parquet write verification failed: {p}')
            journal.replace(p,temp)
        require(len(t) == n, f'Length changed: {p}')
        s['stats'].update(table_stats(t.select(['index','episode_index'])))
        offset += n
        if (i+1) % 100 == 0 or i+1 == len(episodes):
            phase('排除：更新全局索引与逐 episode 统计',i+1,len(episodes))
    phase('排除：清理旧编号并写入全局 metadata')
    for entry in plan['files'][str(last)]:
        p = droid/entry['path']
        if p.exists():
            require(sha256(p) == entry['sha256'], f'Last episode changed: {p}')
            p.unlink()
    # Remove empty directories only; never recursively delete live content.
    for top in droid.iterdir():
        if top.is_dir():
            for directory,_,_ in os.walk(top,topdown=False):
                p=Path(directory)
                if p.name.startswith('episode_') and not any(p.iterdir()):
                    p.rmdir()
    info.update(total_episodes=len(episodes),total_frames=offset,
        total_videos=len(episodes)*sum(v['dtype']=='video' for v in info['features'].values()),
        total_images=offset*sum(v['dtype']=='image' for v in info['features'].values()),
        total_chunks=len({e['episode_index']//1000 for e in episodes}),splits={'train':f'0:{len(episodes)}'})
    journal.json(droid/'meta/episodes.jsonl',episodes,lines=True)
    journal.json(droid/'meta/episodes_stats.jsonl',stats,lines=True)
    journal.json(droid/'meta/stats.json',aggregate(stats))
    journal.json(droid/'meta/info.json',info)
    # A failed transfer may have frozen the original catalogue before rejecting its JSON.
    config = work/'step1_configuration.json'
    if config.exists():
        saved = area/'previous_step1_configuration.json'
        if not saved.exists():
            shutil.copy2(config,saved)
        config.unlink()
    write_json(done,dict(root=str(root),excluded_source_episode=6795,replacement_source_episode=last,
        replacement_episode_index=6795,episodes=len(episodes),frames=offset,
        metadata_hashes={p.relative_to(droid).as_posix():sha256(p) for p in (droid/'meta').iterdir()}))
    print(f'EXCLUSION COMPLETE: {len(episodes)} episodes. Backups: {area}')


def main():
    import fcntl
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--work-dir',required=True,type=Path)
    p.add_argument('--scan-report',required=True,type=Path)
    a=p.parse_args()
    root,work=a.root.resolve(),a.work_dir.resolve()
    require(root.is_dir() and not work.is_relative_to(root) and not root.is_relative_to(work),'Work must be outside dataset')
    work.mkdir(parents=True,exist_ok=True)
    validate_lock_mount(work)
    with (work/'run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY)
        try:
            acquire_directory_lock(fd,fcntl.LOCK_EX)
            execute(root,work,a.scan_report)
        finally:
            os.close(fd)


if __name__ == '__main__':
    main()
