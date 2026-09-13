"""Read-only multi-host calibration over a shared dataset and coordinator.

Each worker has a separate checkpoint directory and a fixed disjoint shard.
Run each shard once at a time; completed candidates are reused on restart.
Only merge publishes the normal refinement completion marker.
"""
from __future__ import annotations

from .progress import tracked
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .common import (atomic_bytes, check_transform, media_path, parquet_path, read_json,
                     require, safe_path, sha256, write_json)
from .pointworld import BATCHED_BACKEND_VERSION, POINTWORLD_COMMIT, URDF_RELATIVE, prepare_assets
from .steps import MARKERS, camera_inputs, freeze_settings, locked_step


SCHEMA = 1
READY = 'SHARD_PLAN_READY.json'
PLAN = 'shards/plan.json'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def protocol():
    return dict(schema=SCHEMA,pointworld_commit=POINTWORLD_COMMIT,backend_version=BATCHED_BACKEND_VERSION)


def sampled_frames(job):
    return np.unique(np.linspace(0,job['length']-1,min(16,job['length']),dtype=int)).tolist()


def input_hashes(droid,info,job):
    frames = sampled_frames(job)
    return dict(frames=frames,parquet_sha256=sha256(parquet_path(droid,info,job['episode_index'])),
                depth_sha256={str(cam):[sha256(media_path(droid,info,job['episode_index'],
                                    f'observation.images.depth_{cam:02d}',t)) for t in frames] for cam in (1,2)})


def slim_job(job):
    result = {k:job[k] for k in ('episode_index','length','source','initial_intrinsics','initial_extrinsics')}
    result['depths'] = {str(c):{'record':({'intrinsic':job['depths'][str(c)]['record']['intrinsic']}
                                       if job['depths'][str(c)]['record'] else None)} for c in (1,2)}
    return result


def require_equal_file(path,value):
    if path.exists():
        require(read_json(path)==value,f'Existing file conflicts with this plan: {path}')
    else:
        write_json(path,value)


def plan_shards(root,work,args):
    from .calibration import select_backend
    started = time.perf_counter()
    with locked_step(root,work):
        require((work/MARKERS['overlap']).exists(),'Run overlap before shard-plan')
        require(not (work/'step_settings/refine.json').exists(),
                'Single-machine refinement has already started; resume its original command and preserve its work directory')
        require(not (work/MARKERS['refine']).exists(),'Refinement already completed; no shard plan is needed')
        require(args.num_shards is not None and args.num_shards>0,'Set --num-shards to a positive integer')
        settings = dict(num_shards=args.num_shards,iterations=args.iterations,backend=select_backend(args),protocol=protocol())
        if (work/READY).exists():
            plan = load_plan(root,work)
            require(all(plan['settings'][k]==settings[k] for k in ('num_shards','iterations','backend')), 'Shard settings changed')
            print(f'SHARD PLAN already complete: {plan["pending_episodes"]} episodes / {len(plan["shards"])} shards',flush=True)
            return
        previous_settings = work/'step_settings/shard-plan.json'
        if previous_settings.exists():
            old = read_json(previous_settings)
            require(all(old[k]==settings[k] for k in ('num_shards','iterations','backend')) and
                    all(old['protocol'].get(k)==v for k,v in protocol().items()), 'Shard settings changed')
        else:
            freeze_settings(work,'shard-plan',settings)
        require(not list((work/'cameras').glob('*.json')),'Unmerged candidates already exist in coordinator/cameras')
        entries,camera_dir = camera_inputs(root,work,args)
        jobs,info = read_json(work/'plan.json'),read_json(work/'training_info.json')
        released,pending = {},[]
        for job,poses,status in entries:
            if poses is not None:
                released[str(job['episode_index'])] = dict(episode_index=job['episode_index'],
                    source_episode_id=job['source']['source_episode_id'],source=status,camera_to_base=poses.tolist())
            else:
                pending.append(slim_job(job))
        # Stable UUID permutation spreads laboratories across equally sized shards.
        pending.sort(key=lambda j:(hashlib.sha256(j['source']['source_episode_id'].encode()).hexdigest(),j['episode_index']))
        urdf = prepare_assets(work/'pointworld') if pending else None
        assets = read_json(work/'pointworld/assets.sha256.json') if urdf else {}
        release_path = work/'shards/released.json'
        require_equal_file(release_path,released)
        parts = []
        for shard_id in range(args.num_shards):
            part_jobs = pending[shard_id::args.num_shards]
            relative = f'shards/manifests/shard-{shard_id:05d}.json'
            require_equal_file(work/relative,dict(shard_id=shard_id,jobs=part_jobs))
            parts.append(dict(shard_id=shard_id,path=relative,sha256=sha256(work/relative),
                              episodes=[j['episode_index'] for j in part_jobs]))
        body = dict(schema=SCHEMA,settings=settings,shards=parts,episodes=[j['episode_index'] for j in jobs],
            pending_episodes=len(pending),released_episodes=len(released),released_sha256=sha256(release_path),assets=assets,
            coordinator_sha256={p:sha256(work/p) for p in ('plan.json','training_info.json','pointworld_overlap.json',MARKERS['overlap'])},
            dataset_sha256={p:sha256(root/'real_world/droid'/p) for p in ('meta/info.json','meta/episodes.jsonl','meta/cameras.json')})
        require_equal_file(work/PLAN,{**body,'plan_id':digest(body)})
        write_json(work/READY,dict(plan_id=digest(body),plan_sha256=sha256(work/PLAN),complete=True))
        write_json(work/'shards/plan_timing.json',dict(elapsed_seconds=time.perf_counter()-started,
                   pending_episodes=len(pending),shards=args.num_shards,scope='Plan only; depth is read during optimization'))
        print(f'SHARD PLAN COMPLETE: {len(pending)} optimized + {len(released)} released episodes; {args.num_shards} shards',flush=True)


def load_plan(root,work,check_dataset=True):
    require((work/READY).exists(),'Run shard-plan first; no complete shared plan')
    ready,plan = read_json(work/READY),read_json(work/PLAN)
    require(ready['complete'] and ready['plan_sha256']==sha256(work/PLAN),'Shared plan changed')
    require(plan['schema']==SCHEMA and plan['plan_id']==ready['plan_id']==digest({k:v for k,v in plan.items() if k!='plan_id'}),
            'Invalid shard plan identity')
    require(all(plan['settings']['protocol'].get(k)==v for k,v in protocol().items()),
            'Worker algorithm/schema differs from shard-plan')
    if check_dataset:
        for relative,h in plan['dataset_sha256'].items():
            require(sha256(safe_path(root/'real_world/droid',relative))==h,f'Dataset metadata changed: {relative}')
        for relative,h in plan['coordinator_sha256'].items():
            require(sha256(safe_path(work,relative))==h,f'Coordinator input changed: {relative}')
    return plan


def part_jobs(work,part):
    path = safe_path(work,part['path'])
    require(sha256(path)==part['sha256'],f'Shard manifest changed: {path}')
    value = read_json(path)
    require(value['shard_id']==part['shard_id'] and [j['episode_index'] for j in value['jobs']]==part['episodes'],
            'Shard manifest episode mapping disagrees with plan')
    return value['jobs']


@contextmanager
def worker_locks(root,work,local,shard_id):
    require(root.is_dir() and work.is_dir(),'Shared dataset/coordinator must exist')
    require(not any(a.is_relative_to(b) or b.is_relative_to(a) for a,b in ((root,work),(root,local),(work,local))),
            'worker-work-dir, dataset and coordinator must be separate, non-nested directories')
    local.mkdir(parents=True,exist_ok=True)
    shared = work/f'shards/results/shard-{shard_id:05d}'
    shared.mkdir(parents=True,exist_ok=True)
    yield shared


def provenance(plan,part,job):
    return dict(plan_id=plan['plan_id'],shard_id=part['shard_id'],manifest_sha256=part['sha256'],
                input_sha256=digest(job.get('calibration_inputs', {})))


def validate_candidate(value,job,plan,part,require_provenance=True):
    require(value['episode_index']==job['episode_index'] and value['source_episode_id']==job['source']['source_episode_id'],
            'Candidate episode/UUID mismatch')
    require(value['source']=='pointworld_method_droid_initialization' and value['pointworld_commit']==POINTWORLD_COMMIT,
            'Candidate optimization source differs')
    if require_provenance:
        require(value.get('shard_provenance')==provenance(plan,part,job),'Candidate belongs to different shard/plan/input')
    if value.get('excluded_bad_depth'):
        require(bool(value.get('failures')) and all(e.get('bad_depth') is True and
                e['episode_index']==job['episode_index'] for e in value['failures']),
                'Missing confirmed bad-depth evidence')
        return
    poses = check_transform(value['camera_to_base'],'shard candidate')
    require(poses.shape==(2,4,4) and len(value['metrics'])==2,'Candidate must contain both external cameras')
    require(value['sample_frame_indices']==sampled_frames(job),'Candidate sampling frames changed')
    for cam,metric in enumerate(value['metrics'],1):
        initial = np.asarray(job['initial_extrinsics'][cam])
        if metric.get('accepted') is False:
            require(np.allclose(poses[cam-1],initial,rtol=0,atol=2e-6),'Rejected candidate did not retain official initial pose')
            require(metric.get('status')=='official_initial_retained' and bool(metric.get('reason')),'Missing rejection reason')
            continue
        require(metric.get('accepted') is True,'Missing explicit calibration acceptance')
        keys = ('initial_train_loss','final_train_loss','initial_holdout_loss','final_holdout_loss')
        require(all(isinstance(metric.get(k),(int,float)) and np.isfinite(metric[k]) and metric[k]>=0 for k in keys),
                'Invalid candidate depth metrics')
        require(metric['final_train_loss']<=metric['initial_train_loss'] and metric['final_holdout_loss']<.1 and
                metric['final_holdout_loss']<=metric['initial_holdout_loss']+1e-4,'Candidate fails depth gates')
        shift = np.linalg.norm(poses[cam-1,:3,3]-initial[:3,3])
        angle = np.rad2deg(Rotation.from_matrix(poses[cam-1,:3,:3]@initial[:3,:3].T).magnitude())
        require(shift<=.30 and angle<=25,'Candidate exceeds pose update bounds')
        train,holdout = metric['train_frames'],metric['holdout_frames']
        require(len(train)>=2 and len(holdout)>=2 and len(set(train+holdout))==len(train)+len(holdout) and
                all(isinstance(i,int) and 0<=i<len(sampled_frames(job)) for i in train+holdout),'Invalid fit/holdout frame split')
        budget = plan['settings']['iterations']
        require(metric['iterations'] in ([2000,6000] if budget==2000 else [budget]),'Candidate iteration budget differs')
        if plan['settings']['backend']=='batched':
            require(metric.get('backend')==BATCHED_BACKEND_VERSION,'Candidate backend differs')


def read_results(work,plan,part):
    directory = work/f'shards/results/shard-{part["shard_id"]:05d}'
    require((directory/'COMPLETE.json').exists(),f'Shard {part["shard_id"]} is incomplete; run/resume that shard')
    receipt = read_json(directory/'COMPLETE.json')
    require(receipt['plan_id']==plan['plan_id'] and receipt['shard_id']==part['shard_id'] and
            receipt['manifest_sha256']==part['sha256'] and receipt['complete'] is True,'Shard completion identity differs')
    expected = {f'episode_{i:06d}.json' for i in part['episodes']}
    actual = {p.name for p in (directory/'cameras').glob('*.json')}
    require(actual==expected==set(receipt['candidate_sha256']),'Missing, duplicate or extra shard candidate')
    results = {}
    for job in part_jobs(work,part):
        path = directory/'cameras'/f'episode_{job["episode_index"]:06d}.json'
        require(sha256(path)==receipt['candidate_sha256'][path.name],f'Candidate changed after shard completion: {path}')
        value = read_json(path)
        validate_candidate(value,job,plan,part)
        results[job['episode_index']] = value
    return results


def refine_shard(root,work,args):
    from .calibration import run_calibrations
    require(args.shard_id is not None and args.shard_id>=0,'Set --shard-id (zero based)')
    require(args.worker_work_dir is not None,'Set --worker-work-dir to a separate per-machine checkpoint/runtime directory')
    local = args.worker_work_dir.expanduser().resolve()
    # Read identity before creating a shard directory, then recheck under locks.
    plan = load_plan(root,work)
    require(args.shard_id<len(plan['shards']),'shard-id is outside the fixed plan')
    with worker_locks(root,work,local,args.shard_id) as shared:
        plan = load_plan(root,work)
        part = plan['shards'][args.shard_id]
        jobs = part_jobs(work,part)
        binding = dict(plan_id=plan['plan_id'],shard_id=args.shard_id,manifest_sha256=part['sha256'])
        require_equal_file(local/'shard_binding.json',binding)
        require(not (local/'configuration.json').exists() and not (local/'manual_workflow.json').exists(),
                'worker-work-dir belongs to a different workflow')
        info = read_json(work/'training_info.json')
        started = time.perf_counter()
        context = dict(**binding,host=socket.gethostname(),pid=os.getpid(),worker_work_dir=str(local),
                       started_utc=datetime.now(timezone.utc).isoformat())
        try:
            if (shared/'COMPLETE.json').exists():
                read_results(work,plan,part)
                print(f'Shard {args.shard_id} already completed; no computation or data write',flush=True)
                return
            require(not (work/MARKERS['refine']).exists(),'Candidates already merged; shard computation is closed')
            write_json(shared/'progress.json',{**context,'status':'preparing','total':len(jobs)})
            for relative,h in plan['assets'].items():
                require(sha256(safe_path(work/'pointworld',relative))==h,f'Robot asset changed: {relative}')
            require(not getattr(args,'iterations_explicit',False) or args.iterations==plan['settings']['iterations'],
                    'Worker iterations differ from shard-plan; omit --iterations to inherit the fixed plan budget')
            args.iterations = plan['settings']['iterations']
            backend = plan['settings']['backend']
            require(args.refine_backend in ('auto',backend),'Worker backend differs from plan')
            require(backend!='batched' or 'cpu' not in args.devices.split(','),'This plan requires GPU workers')
            args.calibration_progress_file = shared/'progress.json'
            args.calibration_progress_context = context
            pending,cached = [],[]
            for job in jobs:
                name = f'episode_{job["episode_index"]:06d}.json'
                output,published = local/'cameras'/name,shared/'cameras'/name
                if published.exists():
                    value = read_json(published)
                    validate_candidate(value,job,plan,part)
                    require_equal_file(output,value)
                if output.exists():
                    validate_candidate(read_json(output),job,plan,part,require_provenance=False)
                    cached.append(job)
                else:
                    pending.append((job,output))
            context['cached_episodes'] = len(jobs)-len(pending)
            if pending:
                run_calibrations(root/'real_world/droid',info,pending,work/'pointworld'/URDF_RELATIVE,local,args,backend)
            candidates = {}
            for job in jobs:
                name = f'episode_{job["episode_index"]:06d}.json'
                value = read_json(local/'cameras'/name)
                validate_candidate(value,job,plan,part,require_provenance=False)
                value['shard_provenance'] = provenance(plan,part,job)
                require_equal_file(shared/'cameras'/name,value)
                candidates[name] = sha256(shared/'cameras'/name)
            receipt = dict(**binding,complete=True,candidate_sha256=candidates,episodes=len(jobs),
                           elapsed_seconds=time.perf_counter()-started,host=context['host'],inputs_verified=True)
            write_json(shared/'COMPLETE.json',receipt)
            write_json(shared/'progress.json',{**context,'status':'complete','completed':len(jobs),'total':len(jobs),
                                              'elapsed_seconds':receipt['elapsed_seconds']})
            (shared/'FAILED.json').unlink(missing_ok=True)
            print(f'SHARD {args.shard_id} COMPLETE: {len(jobs)} episodes; candidates only, waiting for shard-merge',flush=True)
        except Exception as exc:
            write_json(shared/'FAILED.json',{**context,'error':str(exc)})
            write_json(shared/'progress.json',{**context,'status':'failed','error':str(exc)})
            raise


def merge_shards(root,work,args):
    from .steps import refinement_result
    started = time.perf_counter()
    with locked_step(root,work):
        if (work/MARKERS['refine']).exists():
            receipt = read_json(work/MARKERS['refine'])
            require(receipt.get('mode')=='sharded','This refinement was not produced by shard-merge')
            require({p.name:sha256(p) for p in (work/'cameras').glob('*.json')}==receipt['candidate_sha256'],
                    'Merged candidates changed')
            print('Shard merge already complete; no data changed',flush=True)
            return
        plan = load_plan(root,work)
        require(sha256(work/'shards/released.json')==plan['released_sha256'],'PointWorld candidate snapshot changed')
        results = {int(k):v for k,v in read_json(work/'shards/released.json').items()}
        known_dirs = {f'shard-{p["shard_id"]:05d}' for p in plan['shards']}
        require({p.name for p in (work/'shards/results').glob('shard-*')}.issubset(known_dirs),'Extra shard result directory')
        info = read_json(work/'training_info.json')
        for part in plan['shards']:
            incoming = read_results(work,plan,part)
            require(not set(results)&set(incoming),'Duplicate episode across shard results')
            results.update(incoming)
        require(set(results)==set(plan['episodes']),'Merged episode coverage is incomplete or contains extra episodes')
        expected = {f'episode_{i:06d}.json':v for i,v in results.items()}
        require({p.name for p in (work/'cameras').glob('*.json')}.issubset(expected),'Unexpected coordinator candidate')
        # Validate every conflict before publishing anything. Atomic writes allow
        # recovery from interruption halfway through this final publication.
        encoded = {name:(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode() for name,v in expected.items()}
        for name,data in tracked(encoded.items(),'合并预检：相机记录'):
            path = work/'cameras'/name
            require(not path.exists() or sha256(path)==hashlib.sha256(data).hexdigest(),f'Merge candidate conflict: {path}')
        for name,data in tracked(encoded.items(),'合并写入：相机记录'):
            if not (work/'cameras'/name).exists():
                atomic_bytes(work/'cameras'/name,data)
        result = refinement_result(work,read_json(work/'plan.json'))
        receipt = {**result,'mode':'sharded','plan_id':plan['plan_id'],'complete':True,
                   'merge_elapsed_seconds':time.perf_counter()-started}
        write_json(work/'SHARD_MERGE_SUCCESS.json',receipt)
        write_json(work/MARKERS['refine'],receipt)
        print('SHARD MERGE COMPLETE: full candidate coverage verified. Stopped before apply.',flush=True)


def shard_status(root,work,args):
    plan = load_plan(root,work,check_dataset=not (work/MARKERS['refine']).exists())
    rows = []
    for part in plan['shards']:
        directory = work/f'shards/results/shard-{part["shard_id"]:05d}'
        progress = read_json(directory/'progress.json') if (directory/'progress.json').exists() else {'status':'not_started'}
        require(progress.get('plan_id',plan['plan_id'])==plan['plan_id'],'Progress belongs to another plan')
        rows.append(dict(shard_id=part['shard_id'],episodes=len(part['episodes']),progress=progress))
    print(json.dumps(dict(plan_id=plan['plan_id'],shards=rows,merged=(work/MARKERS['refine']).exists()),indent=2),flush=True)


def run_shard_step(args):
    root,work = args.root.resolve(),args.work_dir.resolve()
    {'shard-plan':plan_shards,'shard-refine':refine_shard,'shard-merge':merge_shards,'shard-status':shard_status}[args.step](root,work,args)
