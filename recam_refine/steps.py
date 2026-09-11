"""Independently executable stages sharing the pipeline recovery journal."""
from __future__ import annotations

from .progress import phase, tracked, mapped
import os
from pathlib import Path
import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor

from .common import (require, read_json, read_jsonl, write_json, sha256, media_path, parquet_path,
                     acquire_directory_lock, validate_lock_mount)
from .inputs import canonical_manifest, download_manifest


def transfer_configuration(root, source, chunks, manifest):
    return dict(root=str(Path(root).resolve()),depth_output=str(Path(source).resolve()),chunks=sorted(chunks),
        identities=[dict(episode_index=i,source_episode_id=r['source_episode_id'],length=int(r['length']),
                         camera_serials={c:str(r['camera_serials'][c]) for c in ('external_1','external_2')})
                    for i,r in sorted(manifest.items()) if int(r.get('source_episode_index',i))//1000 in chunks])


def manual_workflow(root, work):
    exclusion = work/'exclude_episode_006795'
    require(not (exclusion/'plan.json').exists() or (exclusion/'SUCCESS.json').exists(),
            'Episode exclusion was interrupted; resume exclude-6795 before other steps')
    require(not (work/'configuration.json').exists(), 'This work directory belongs to an automatic run; resume it with its original command')
    path = work/'manual_workflow.json'
    value = dict(version=1,root=str(root))
    if path.exists():
        require(read_json(path)==value,'Manual workflow dataset root changed')
    else:
        write_json(path,value)


def transfer_only(args):
    from .progress import phase
    phase('加载迁移模块（任务）', 0, 1)
    import fcntl
    from .pipeline import transfer_depth, parse_chunks
    phase('检查迁移路径与目录锁（任务）', 0, 1)
    root, work, source = args.root.resolve(), args.work_dir.resolve(), args.depth_output.resolve()
    require(root.is_dir(), f'Missing dataset root: {root}')
    require(not work.is_relative_to(root) and not root.is_relative_to(work), 'work-dir must be outside the dataset')
    require(not work.is_relative_to(source) and not source.is_relative_to(work), 'Depth output and work-dir must be separate')
    droid = root/'real_world/droid'
    require(droid.is_dir(), f'Expected {droid}')
    work.mkdir(parents=True,exist_ok=True)
    validate_lock_mount(work)
    with (work/'run.lock').open('a+') as lock:
        fd = os.open(root,os.O_RDONLY|os.O_DIRECTORY)
        try:
            try:
                mode = fcntl.LOCK_EX
                fcntl.flock(lock,mode|fcntl.LOCK_NB)
                acquire_directory_lock(fd,mode)
            except BlockingIOError:
                raise RuntimeError('Another refinement is using this dataset or work directory')
            manual_workflow(root,work)
            phase('检查迁移路径与目录锁（任务）', 1, 1)
            phase('读取并校验 DROID metadata（任务）', 0, 1)
            chunks = parse_chunks(args.depth_chunks)
            info_path, episodes_path = droid/'meta/info.json', droid/'meta/episodes.jsonl'
            info = read_json(info_path)
            require(info.get('codebase_version')=='v2.1' and info['chunks_size']==1000,
                    'Step 1 expects the ReCam LeRobot v2.1 / 1000-episode chunk layout')
            episodes = [e for e in read_jsonl(episodes_path) if int(e.get('source_episode_index',e['episode_index']))//1000 in chunks]
            require(episodes, f'No target episodes in chunks {args.depth_chunks}')
            phase('读取并校验 DROID metadata（任务）', 1, 1, f'选中 {len(episodes)} 个 episode')
            manifest_path = args.episode_manifest or download_manifest(work,sorted({int(e.get('source_episode_index',e['episode_index']))//1000 for e in episodes}))
            manifest = canonical_manifest(manifest_path,episodes)
            config = transfer_configuration(root,source,chunks,manifest)
            config_path = work/'step1_configuration.json'
            if config_path.exists():
                require(read_json(config_path)==config, 'Step 1 paths/chunks/identities changed; use the original command')
            else:
                require(not (work/'03_plan.complete.json').exists(), 'Cannot transfer new depth after alignment has started')
                write_json(config_path,config)
            if (work/'03_plan.complete.json').exists():
                require((work/'STEP1_DEPTH_TRANSFER_SUCCESS.json').exists(), 'Incomplete transfer before alignment')
                print('Step 1 already completed; later stages have started. No data changed.',flush=True)
                return
            transferred = transfer_depth(root,droid,source,manifest,chunks,work,workers=args.workers)
            # transfer_depth already verified copies or atomically renamed directories.
            total = sum(row['frame_count'] for row in transferred)
            result = dict(stage=1,complete=True,episodes=len(episodes),cameras=2*len(episodes),png_files=total,
                          destination=str(droid/'images'),metadata_unchanged=True,
                          source_cleanup='same-filesystem files moved; cross-filesystem originals retained until final checks')
            write_json(work/'02_transfer.complete.json',dict(complete=True))
            write_json(work/'STEP1_DEPTH_TRANSFER_SUCCESS.json',result)
            (work/'STEP1_FAILED.json').unlink(missing_ok=True)
            print(f'STEP 1 COMPLETE: {len(episodes)} episodes / {total} PNGs. Stopped after depth transfer.',flush=True)
        finally:
            os.close(fd)


MARKERS = {'transfer':'STEP1_DEPTH_TRANSFER_SUCCESS.json','unpack':'STEP2_UNPACK_SUCCESS.json',
           'align':'STEP3_ALIGN_SUCCESS.json','overlap':'STEP4_OVERLAP_SUCCESS.json',
           'refine':'STEP4_REFINE_SUCCESS.json','apply':'STEP4_APPLY_SUCCESS.json',
           'check':'STEP5_CHECK_SUCCESS.json','cleanup':'SUCCESS.json'}
PREVIOUS = {'unpack':'transfer','align':'unpack','overlap':'align','refine':'overlap',
            'apply':'refine','check':'apply','cleanup':'check'}


@contextmanager
def locked_step(root,work,shared=False):
    import fcntl
    require(root.is_dir() and (root/'real_world/droid').is_dir(),f'Missing recam_lerobot/real_world/droid: {root}')
    require(not work.is_relative_to(root) and not root.is_relative_to(work),'work-dir must be separate from the dataset')
    work.mkdir(parents=True,exist_ok=True)
    validate_lock_mount(work)
    with (work/'run.lock').open('a+') as lock:
        fd = os.open(root,os.O_RDONLY|os.O_DIRECTORY)
        try:
            try:
                mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
                fcntl.flock(lock,mode|fcntl.LOCK_NB)
                acquire_directory_lock(fd,mode)
            except BlockingIOError:
                raise RuntimeError('Another refinement is using this dataset or work directory')
            manual_workflow(root,work)
            yield
        finally:
            os.close(fd)


def freeze_settings(work,stage,settings):
    path = work/'step_settings'/f'{stage}.json'
    if path.exists():
        require(read_json(path)==settings,f'{stage} settings changed; resume with the original arguments')
    else:
        write_json(path,settings)


def unpack_only(root,work,workers=8,chunks=None):
    from .pipeline import discover
    from .archives import unpack_archive
    droid = root/'real_world/droid'
    from .parallel import io_map, Counter
    groups = {}
    phase('查找数据子集（任务）', 0, 1)
    subsets = discover(root)
    for subset in subsets:
        directories = ([subset/'images'/f'chunk-{i:03d}' for i in sorted(chunks)]
                       if chunks is not None else sorted((subset/'images').glob('chunk-*')))
        for index, directory in enumerate(directories):
            phase('查找 TAR（chunk）', index, len(directories), str(directory))
            for archive in sorted(directory.glob('observation.images.depth_*/*.tar')):
                if subset==droid and archive.parent.name=='observation.images.depth_00':
                    continue
                groups.setdefault((subset, archive.parent), []).append(archive)
        phase('查找 TAR（chunk）', len(directories), len(directories), str(subset))
    counter = Counter()
    total = sum(map(len, groups.values()))
    def extract(group):
        (subset, directory), archives = group
        results = []
        for archive in archives:
            print(f'Unpack {archive}',flush=True)
            results.append(unpack_archive(archive,subset,work/'archive_receipts'))
            counter.tick()
        return results
    batches = io_map(extract, groups.items(), workers, '并发解压（相机目录）',
                     lambda: f'TAR={counter.value}/{total}；并发={workers}')
    receipts = [receipt for batch in batches for receipt in batch]
    if chunks is not None:
        # Disjoint chunk workers only publish existing per-TAR receipts.
        # No shared summary, marker or read-modify-write between machines.
        return dict(archives=len(receipts),selected_archives=len(receipts),total_archives=total,
                    all_chunks_complete=False,range_only=True)
    write_json(work/'unpacked.json',receipts)
    return dict(archives=len(receipts),selected_archives=len(receipts),total_archives=total,
                all_chunks_complete=True,tar_policy='TARs retained until explicit cleanup after successful checks')



def align_only(root,work,args):
    from .pipeline import corrected_droid_info, plan_episode, apply_episode, update_metadata
    from .inputs import load_depth_records
    import numpy as np
    droid = root/'real_world/droid'
    if not (work/'original_droid_episodes.json').exists():
        write_json(work/'original_droid_episodes.json',read_jsonl(droid/'meta/episodes.jsonl'))
    if not (work/'original_droid_info.json').exists():
        write_json(work/'original_droid_info.json',read_json(droid/'meta/info.json'))
    episodes = read_json(work/'original_droid_episodes.json')
    require([e['episode_index'] for e in episodes]==list(range(len(episodes))),'DROID episode catalogue must be complete and ordered')
    info = corrected_droid_info(droid,read_json(work/'original_droid_info.json'))
    manifest_path = args.episode_manifest or download_manifest(work,sorted({int(e.get('source_episode_index',e['episode_index']))//1000 for e in episodes}))
    manifest = canonical_manifest(manifest_path,episodes)
    step1 = read_json(work/'step1_configuration.json')
    require(transfer_configuration(root,step1['depth_output'],set(step1['chunks']),manifest)==step1,'Step 1 episode mapping changed')
    metadata_roots = [droid,Path(step1['depth_output']),*args.depth_metadata]
    records = load_depth_records(metadata_roots,manifest,cache_dir=work/'input_cache')
    for row in step1['identities']:
        for cam in (1, 2):
            require((row['episode_index'], cam) in records,
                    f'Migrated depth sidecar missing before alignment: {row["episode_index"]}/{cam}')
    settings = dict(manifest_sha256=sha256(manifest_path) if Path(manifest_path).is_file() else
                    {p.name:sha256(p) for p in sorted(Path(manifest_path).glob('chunk-*.jsonl'))},
                    sidecars={str(k):v['sha256'] for k,v in records.items()})
    freeze_settings(work,'align',settings)
    if not (work/'03_plan.complete.json').exists():
        grouped = {}
        for key,value in records.items():
            grouped.setdefault(key[0],{})[key] = value
        inputs = [(str(droid),info,e,manifest[e['episode_index']],grouped.get(e['episode_index'],{})) for e in episodes]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = mapped(pool,plan_episode,inputs,'对齐：检查 episode')
        write_json(work/'plan.json',jobs)
        write_json(work/'training_info.json',info)
        write_json(work/'03_plan.complete.json',dict(complete=True))
    jobs, info = read_json(work/'plan.json'), read_json(work/'training_info.json')
    camera_dir = work/'alignment_cameras'
    tasks, offset = [], 0
    for job in tracked(jobs,'相机参数：episode'):
        camera = dict(source='droid_initial_alignment_only',camera_to_base=np.asarray(job['initial_extrinsics'])[1:].tolist())
        write_json(camera_dir/f'episode_{job["episode_index"]:06d}.json',camera)
        tasks.append((str(root),str(droid),info,job,camera,offset,str(work),'aligned'))
        offset += job['length']
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        stats = mapped(pool,apply_episode,tasks,'写回：episode')
    update_metadata(root,droid,info,jobs,stats,work,camera_dir,calibration_pending=True)
    return dict(episodes=len(jobs),frames=offset,dropped_timesteps=sum(j['previous_length']-j['length'] for j in jobs),
                calibration='DROID initial extrinsics retained; refinement remains a separate step')


def camera_inputs(root,work,args):
    from .inputs import download_inputs
    from .pointworld import release_pose
    from collections import Counter
    jobs = read_json(work/'plan.json')
    camera_dir = args.pointworld_cameras
    selection = work/'camera_input_directory.json'
    if camera_dir is None and selection.exists():
        camera_dir = Path(read_json(selection)['path'])
    if camera_dir is None:
        _,camera_dir = download_inputs(work,sorted({j['episode_index']//1000 for j in jobs}))
    require(Path(camera_dir).is_dir(),f'Missing PointWorld camera directory: {camera_dir}')
    write_json(selection,dict(path=str(Path(camera_dir).resolve())))
    entries,counts,details = [],Counter(),[]
    for job in tracked(jobs,'相机参数：episode'):
        poses,status = release_pose(camera_dir,job['source'])
        counts[status] += 1
        entries.append((job,poses,status))
        details.append(dict(episode_index=job['episode_index'],source_episode_id=job['source']['source_episode_id'],
                            camera_serials=job['source']['camera_serials'],status=status))
    write_json(work/'pointworld_overlap.json',dict(counts))
    from .common import write_jsonl
    write_jsonl(work/'pointworld_overlap_episodes.jsonl',details)
    print(f'PointWorld overlap: {dict(counts)}',flush=True)
    return entries,Path(camera_dir)


def refine_only(root,work,args):
    from .calibration import run_calibrations, select_backend
    from .pointworld import prepare_assets, BATCHED_BACKEND_VERSION
    require(not (work/'step_settings/shard-plan.json').exists(),'Sharded refinement is selected; use shard-refine and shard-merge')
    droid = root/'real_world/droid'
    entries,camera_dir = camera_inputs(root,work,args)
    saved_path = work/'step_settings/refine.json'
    saved = read_json(saved_path) if saved_path.exists() else None
    backend = select_backend(args,saved)
    settings = dict(iterations=args.iterations,camera_directory=str(camera_dir.resolve()),
        camera_files={job['source']['source_episode_id']:sha256(camera_dir/(job['source']['source_episode_id']+'_cameras.json'))
                      for job,_,_ in entries if (camera_dir/(job['source']['source_episode_id']+'_cameras.json')).exists()})
    if saved is None or 'backend' in saved:
        settings['backend'] = backend
    if backend == 'batched':
        settings['backend_version'] = BATCHED_BACKEND_VERSION
    freeze_settings(work,'refine',settings)
    jobs,info = read_json(work/'plan.json'),read_json(work/'training_info.json')
    pending = []
    for job,poses,status in entries:
        output = work/'cameras'/f'episode_{job["episode_index"]:06d}.json'
        if output.exists():
            continue
        if poses is not None:
            write_json(output,dict(episode_index=job['episode_index'],source_episode_id=job['source']['source_episode_id'],
                                  source=status,camera_to_base=poses.tolist()))
        else:
            pending.append((job,output))
    if pending:
        urdf = prepare_assets(work/'pointworld')
        run_calibrations(droid,info,pending,urdf,work,args,backend)
    return refinement_result(work,jobs)


def refinement_result(work,jobs):
    retained = []
    for job in tracked(jobs,'相机参数：episode'):
        camera = read_json(work/'cameras'/f'episode_{job["episode_index"]:06d}.json')
        for cam,metric in enumerate(camera.get('metrics',[]),1):
            if metric.get('accepted') is False:
                retained.append(dict(episode_index=job['episode_index'],camera=cam,**metric))
    write_json(work/'retained_official_calibrations.json',retained)
    return dict(episodes=len(jobs),pointworld=read_json(work/'pointworld_overlap.json'),retained_official_cameras=len(retained),
                all_external_calibrations_accepted=not retained,
                candidate_sha256={p.name:sha256(p) for p in sorted((work/'cameras').glob('episode_*.json'))})


def apply_only(root,work,args):
    from .pipeline import apply_episode, update_metadata
    droid = root/'real_world/droid'
    jobs,info = read_json(work/'plan.json'),read_json(work/'training_info.json')
    candidates = read_json(work/MARKERS['refine'])['candidate_sha256']
    require({p.name:sha256(p) for p in sorted((work/'cameras').glob('episode_*.json'))}==candidates,
            'Candidates changed after refinement; refusing to apply unverified replacements')
    tasks,offset = [],0
    for job in tracked(jobs,'相机参数：episode'):
        camera = read_json(work/'cameras'/f'episode_{job["episode_index"]:06d}.json')
        tasks.append((str(root),str(droid),info,job,camera,offset,str(work)))
        offset += job['length']
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        stats = mapped(pool,apply_episode,tasks,'写回：episode')
    update_metadata(root,droid,info,jobs,stats,work)
    return dict(episodes=len(jobs),frames=offset)


def check_only(root,work,args,subsets):
    from .validate import check_subset
    from .audit import _run_audit_locked
    marker = work/MARKERS['check']
    marker.unlink(missing_ok=True)
    summaries = [check_subset(s,work/'checks',args.workers,droid=s==root/'real_world/droid') for s in subsets]
    write_json(work/'checks.json',summaries)
    audit = argparse.Namespace(root=root,work_dir=work,report_dir=work/'camera_audit',episodes=['all'],
        frames=args.audit_frames,image_frames=0,workers=args.workers,device='cpu',fit=False,iterations=args.iterations,
        episode_manifest=None,pointworld_cameras=args.pointworld_cameras or Path(read_json(work/'camera_input_directory.json')['path']),
        candidate_dir=None,depth_metadata=[],fail_on_review=True)
    status = _run_audit_locked(audit)
    require(status==0,'Geometry requires review; see camera_audit/index.html. Cleanup is not enabled')
    return dict(full_decode=True,subsets=summaries,
                geometry_summary_sha256=sha256(work/'camera_audit/summary.json'))


def cleanup_only(root,work,args,subsets):
    from .pipeline import finalize
    check = read_json(work/MARKERS['check'])
    require(not (work/'camera_audit/QUALITY_REVIEW_REQUIRED.json').exists(),'Geometry review is still required')
    require(sha256(work/'camera_audit/summary.json')==check['geometry_summary_sha256'],'Geometry report changed; rerun check')
    finalize(root,root/'real_world/droid',subsets,work)
    # The dataset lock remains held; finalize only removes verified sources,
    # obsolete TARs/modalities and auxiliary files, not declared training files.
    return dict(root=str(root),full_decode=True,subsets=check['subsets'],backup_directory=str(work/'original'),
                **read_json(work/MARKERS['refine']))


def run_step(args):
    from .pipeline import discover
    root,work = args.root.resolve(),args.work_dir.resolve()
    if args.step.startswith('shard-'):
        from .shards import run_shard_step
        return run_shard_step(args)
    if args.step=='status':
        print(json.dumps({stage:(work/name).exists() for stage,name in MARKERS.items()},indent=2),flush=True)
        return
    # Unpack relies on the operator assigning non-overlapping chunks.
    # A no-range unpack is a single-machine completion pass after all workers exit.
    with nullcontext() if args.step == 'unpack' else locked_step(root,work):
        if args.step == 'unpack':
            require(not (work/'configuration.json').exists(), 'This work directory belongs to an automatic run; resume its original command')
            if (work/'manual_workflow.json').exists():
                require(read_json(work/'manual_workflow.json') == dict(version=1,root=str(root)), 'Manual workflow dataset root changed')
            require(root.is_dir() and (root/'real_world/droid').is_dir(), 'Missing DROID root')
            require(not work.is_relative_to(root) and not root.is_relative_to(work), 'work-dir must be separate from dataset')
            require(not (work/'exclude_episode_006795/plan.json').exists() or
                    (work/'exclude_episode_006795/SUCCESS.json').exists(), 'Resume unfinished exclusion first')
        chunk_ids = getattr(args, 'chunk_ids', None)
        require(chunk_ids is None or args.step == 'unpack', '--chunk-ids is only supported for unpack')
        previous = PREVIOUS[args.step]
        require((work/MARKERS[previous]).exists(),f'Run {previous} first; missing {MARKERS[previous]}')
        marker = work/MARKERS[args.step]
        if marker.exists() and args.step!='check':
            print(f'{args.step} already completed. No data changed.',flush=True)
            return
        require(args.step!='check' or not (work/'SUCCESS.json').exists(),
                'Cleanup already completed. Use the read-only check/audit-cameras commands for subsequent audits')
        subsets = discover(root) if args.step != 'unpack' else None
        if args.step=='unpack':
            from .pipeline import parse_chunks
            chunks = parse_chunks(chunk_ids) if chunk_ids is not None else None
            require(chunks is None or (chunks and min(chunks) >= 0), 'Invalid chunk range')
            result = unpack_only(root,work,args.workers,chunks)
            if not result['all_chunks_complete']:
                print(f"UNPACK RANGE COMPLETE: TAR={result['selected_archives']}/{result['total_archives']}. After ALL machines exit, run unpack without --chunk-ids on ONE CPU to finalize.",flush=True)
                return
        elif args.step=='align':
            result = align_only(root,work,args)
        elif args.step=='overlap':
            camera_inputs(root,work,args)
            result = read_json(work/'pointworld_overlap.json')
        elif args.step=='refine':
            result = refine_only(root,work,args)
        elif args.step=='apply':
            result = apply_only(root,work,args)
        elif args.step=='check':
            result = check_only(root,work,args,subsets)
        elif args.step=='cleanup':
            result = cleanup_only(root,work,args,subsets)
        write_json(marker,{**result,'complete':True})
        (work/f'{args.step.upper()}_FAILED.json').unlink(missing_ok=True)
        print(f'{args.step.upper()} COMPLETE: {marker}. Stopped; no later step started.',flush=True)
