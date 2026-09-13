"""Exclude confirmed bad-depth episodes during the existing CPU apply pass.

The immutable GPU plan/results keep their original IDs. Only surviving tail
episodes fill holes, so only a few media directories move. Parquet index updates
share the normal apply pass; no additional media scan or payload hashing.
"""
import copy
import errno
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .common import read_json, read_jsonl, require, write_json, write_jsonl
from .progress import mapped, phase


def survivor_order(count, bad):
    require(bad and bad <= set(range(count)) and len(bad)<count, 'Invalid excluded episode IDs')
    size = count-len(bad)
    donors = iter(sorted(set(range(size,count))-bad, reverse=True))
    return [next(donors) if i in bad else i for i in range(size)]


def episode_entries(droid, i, chunk_size):
    token = f'episode_{i:06d}'
    for top in ('data','images','videos'):
        chunk = droid/top/f'chunk-{i//chunk_size:03d}'
        if not chunk.exists():
            continue
        # Two levels only: Parquet files or camera/episode entries, never PNGs.
        for entry in chunk.iterdir():
            if entry.name == token or entry.name.startswith(token+'.'):
                yield entry
            elif entry.is_dir() and entry.name.startswith('observation.'):
                for child in entry.glob(token+'*'):
                    if child.name == token or child.name.startswith(token+'.'):
                        yield child


def relocate(src, dst):
    src,dst = Path(src),Path(dst)
    if not src.exists():
        require(dst.exists(),f'Missing move source and destination: {src} -> {dst}')
        return
    require(not src.is_symlink(),f'Symlink episode entry: {src}')
    dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists():
        try:
            os.replace(src,dst)
            return
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
        staged = dst.with_name(dst.name+'.quarantine-part')
        if staged.exists():
            if staged.is_dir():
                shutil.rmtree(staged)
            else:
                staged.unlink()
        if src.is_dir():
            shutil.copytree(src,staged)
        else:
            shutil.copy2(src,staged)
        os.replace(staged,dst)
    # A committed cross-filesystem copy can survive interruption before removal.
    if src.is_dir():
        shutil.rmtree(src)
    else:
        src.unlink()


def apply_filtered(root,work,args,jobs,info,excluded):
    from .pipeline import apply_episode, update_metadata
    droid = root/'real_world/droid'
    area = work/'bad_depth_exclusion'
    area.mkdir(parents=True,exist_ok=True)
    done = area/'SUCCESS.json'
    if done.exists():
        print('SKIPPED bad-depth exclusion: already complete',flush=True)
        return read_json(done)
    bad = {v['episode_index'] for v in excluded}
    require([j['episode_index'] for j in jobs]==list(range(len(jobs))), 'Expected contiguous input episodes')
    order = survivor_order(len(jobs),bad)
    snapshot = area/'operations.json'
    if not snapshot.exists():
        catalogue = read_jsonl(droid/'meta/episodes.jsonl')
        require(len(catalogue)==len(jobs),'Episode catalogue changed before exclusion')
        require(info['splits']=={'train':f'0:{len(jobs)}'},'Expected all-train DROID split')
        moves = []
        for i in sorted(bad):
            for src in episode_entries(droid,i,info['chunks_size']):
                moves.append([str(src),str(area/'episodes'/src.relative_to(droid))])
        for new,old in enumerate(order):
            if old==new:
                continue
            for src in episode_entries(droid,old,info['chunks_size']):
                rel = src.relative_to(droid)
                parts = [f'chunk-{new//info["chunks_size"]:03d}' if p==f'chunk-{old//info["chunks_size"]:03d}'
                         else p.replace(f'episode_{old:06d}',f'episode_{new:06d}') for p in rel.parts]
                moves.append([str(src),str(droid.joinpath(*parts))])
        write_json(snapshot,dict(order=order,bad=sorted(bad),catalogue=catalogue,moves=moves))
    saved = read_json(snapshot)
    require(saved['order']==order and saved['bad']==sorted(bad),'Exclusion selection changed')
    tasks,offset,final_jobs,final_catalogue = [],0,[],[]
    for new,old in enumerate(order):
        job = copy.deepcopy(jobs[old])
        job['output_episode_index'] = new
        camera = read_json(work/'cameras'/f'episode_{old:06d}.json')
        tasks.append((str(root),str(droid),info,copy.deepcopy(job),camera,offset,str(work),'applied_filtered'))
        offset += job['length']
        job['episode_index'] = new
        job['source']['episode_index'] = new
        final_jobs.append(job)
        row = dict(saved['catalogue'][old],episode_index=new)
        if new!=old:
            row['source_episode_index'] = row.get('source_episode_index',old)
        final_catalogue.append(row)
        if new!=old:
            camera['episode_index'] = new
            job['candidate_path'] = str(area/'cameras'/f'episode_{new:06d}.json')
            write_json(Path(job['candidate_path']),camera)
    # Existing apply writes the new global and episode indices in the same pass.
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        stats = mapped(pool,apply_episode,tasks,'写回有效 episode')
    for i,stat in enumerate(stats):
        # mapped preserves input order (as does the regular apply path).
        stat['episode_index'] = i
    for n,(src,dst) in enumerate(saved['moves']):
        receipt = area/'moves'/f'{n}.json'
        if not receipt.exists():
            relocate(src,dst)
            write_json(receipt,dict(source=src,target=dst))
        phase('移出异常 episode / 调整尾部编号',n+1,len(saved['moves']))
    final_info = copy.deepcopy(info)
    final_info['splits'] = {'train':f'0:{len(order)}'}
    write_json(area/'original_droid_episodes.json',final_catalogue)
    update_metadata(root,droid,final_info,final_jobs,stats,area,work/'cameras')
    write_json(area/'plan.json',final_jobs)
    write_jsonl(work/'refined_episode_manifest.jsonl',read_jsonl(area/'refined_episode_manifest.jsonl'))
    result = dict(episodes=len(order),frames=offset,excluded_episodes=sorted(bad),
                  renumbered={str(old):new for new,old in enumerate(order) if new!=old})
    write_json(done,result)
    print(f'EXCLUSION COMPLETE: {len(bad)} excluded; {len(order)} retained; GPU results unchanged',flush=True)
    return result
