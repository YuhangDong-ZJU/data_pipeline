"""Persistent GPU workers, bounded prefetch and resumable camera batches.

Only the work directory is written. Every camera checkpoint is bound to the
actual input bytes, poses, point samples and optimization protocol.
"""
from __future__ import annotations

from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
import gc
import hashlib
import io
import json
from multiprocessing import get_context
import os
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from .common import atomic_bytes, media_path, parquet_path, require, values, write_json
from .pointworld import POINTWORLD_COMMIT, BATCHED_BACKEND_VERSION, Robot


def select_backend(args, saved=None):
    requested = getattr(args, 'refine_backend', 'auto')
    # Legacy runs stay on their original numerical protocol. Scheduling can change.
    previous = saved.get('backend', 'reference') if saved is not None else None
    selected = previous if requested == 'auto' and previous else requested
    if selected == 'auto':
        selected = 'reference' if args.devices == 'cpu' else 'batched'
    require(previous is None or selected == previous,
            'Refinement backend changed; resume this work directory with its original backend')
    if saved is not None and selected == 'batched':
        require(saved.get('backend_version') == BATCHED_BACKEND_VERSION,
                'Batched optimization protocol changed; resume with the code version saved for this run')
    require(selected != 'batched' or 'cpu' not in args.devices.split(','),
            'Batched production refinement requires CUDA; use --refine-backend reference for CPU')
    return selected


def state_path(work, task):
    return Path(work)/'calibration_state'/f'episode_{task["job"]["episode_index"]:06d}_cam{task["cam"]}.pt'


def load_state(path):
    import torch
    return torch.load(path, map_location='cpu', weights_only=True) if path.exists() else None


def save_state(path, value):
    import torch
    stream = io.BytesIO()
    torch.save(value, stream)
    atomic_bytes(path, stream.getvalue())


def split_state(state, index):
    """Extract independent Adam moments; its step is shared only by equal-age rows."""
    return {**{k:state[k][index:index+1].clone() for k in ('param','best','best_loss','running')},
            'iteration':state['iteration'],
            'optimizer':{k:(v.clone() if k == 'step' else v[index:index+1].clone())
                         for k,v in state['optimizer'].items()}}


def join_states(states):
    import torch
    require(len({s['iteration'] for s in states}) == 1, 'Cannot mix different Adam iteration counts')
    require(all(torch.equal(s['optimizer']['step'],states[0]['optimizer']['step']) for s in states),
            'Adam steps disagree with batch age')
    return {**{k:torch.cat([s[k] for s in states]) for k in ('param','best','best_loss','running')},
            'iteration':states[0]['iteration'],
            'optimizer':{k:(states[0]['optimizer'][k].clone() if k == 'step' else
                            torch.cat([s['optimizer'][k] for s in states])) for k in states[0]['optimizer']}}


_worker = None


def init_worker(root, info, urdf, work, device, iterations, batch_size, graphs):
    global _worker
    import torch
    torch.set_num_threads(2)
    torch.cuda.set_device(device)
    free,total = torch.cuda.mem_get_info(device)
    # Keep headroom for graph pools/variable image sizes and other users. OOMs
    # further split the actual batch; total card VRAM never implies free VRAM.
    auto = max(1,min(32,int(free*.55/(384*2**20))))
    _worker = dict(root=Path(root), info=info, robot=Robot(urdf), work=Path(work), device=device,
                   iterations=iterations, batch_size=batch_size or auto, graphs=graphs,
                   loader=ThreadPoolExecutor(max_workers=1), prefetched=None,
                   free_vram_gib=free/2**30, total_vram_gib=total/2**30)


def worker_capacity():
    return {k:_worker[k] for k in ('device','batch_size','free_vram_gib','total_vram_gib')}


def task_key(task):
    return (task['job']['episode_index'],task['cam'],task['iteration'])


def prepare_tasks(tasks):
    from .batched import BACKEND_VERSION
    w = _worker
    episodes, prepared = {},[]
    for task in tasks:
        try:
            job,cam = task['job'],task['cam']
            i = job['episode_index']
            if i not in episodes:
                raw = parquet_path(w['root'],w['info'],i).read_bytes()
                table = pq.read_table(io.BytesIO(raw),use_threads=False).slice(0,job['length'])
                require(len(table) == job['length'],f'Insufficient Parquet rows: {i}')
                joints = values(table['observation.states.joint_state'])
                gripper = values(table['observation.states.gripper_state']).ravel()
                require(np.isfinite(joints).all() and np.isfinite(gripper).all() and
                        np.all((gripper >= 0)&(gripper <= 1)),f'Invalid robot configuration: {i}')
                frames = np.unique(np.linspace(0,job['length']-1,min(16,job['length']),dtype=int)).tolist()
                points = [w['robot'].points(joints[t],gripper[t]) for t in frames]
                episodes[i] = (hashlib.sha256(raw).hexdigest(), frames, points)
            parquet_hash,frames,points = episodes[i]
            k = np.asarray(job['initial_intrinsics'][cam],dtype=np.float64).copy()
            record = job['depths'][str(cam)]['record']
            if record:
                k = np.asarray(record['intrinsic'],dtype=np.float64).copy()
            k[:2] *= .5
            depths,hashes = [],[]
            for t in frames:
                raw = media_path(w['root'],w['info'],i,f'observation.images.depth_{cam:02d}',t).read_bytes()
                hashes.append(hashlib.sha256(raw).hexdigest())
                with Image.open(io.BytesIO(raw)) as im:
                    depths.append(np.asarray(im,dtype=np.float32)[::2,::2]/1000.)
            data = dict(initial=np.asarray(job['initial_extrinsics'][cam]), k=k, depths=depths, points=points)
            identity = dict(backend=BACKEND_VERSION, iterations=w['iterations'], pointworld_commit=POINTWORLD_COMMIT,
                            episode=i, source=job['source'], camera=cam, frames=frames, parquet=parquet_hash,
                            png=hashes, initial=data['initial'].tolist(), intrinsic=k.tolist(),
                            points=[hashlib.sha256(p.tobytes()).hexdigest() for p in points])
            fingerprint = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
            saved = load_state(state_path(w['work'],task))
            if saved:
                require(saved['fingerprint'] == fingerprint, f'Checkpoint inputs changed: episode {i} camera {cam}')
                require(saved['state']['iteration'] == task['iteration'], 'Checkpoint changed during scheduling')
            else:
                require(task['iteration'] == 0, 'Missing optimizer checkpoint')
            prepared.append(dict(task=task, data=data, frames=frames, fingerprint=fingerprint, saved=saved))
        except Exception as exc:
            prepared.append(dict(task=task,error=str(exc)))
    return prepared


def _fit_group(items):
    import torch
    from .batched import CameraBatch
    w = _worker
    solver = None
    try:
        solver = CameraBatch([p['data'] for p in items],w['device'],use_graph=w['graphs'])
        if items[0]['task']['iteration']:
            solver.restore(join_states([p['saved']['state'] for p in items]))
        start_iteration = solver.iteration
        continuation = any(a['iterations'] == 2000 and not a['accepted'] for a in (items[0]['saved'] or {}).get('attempts',[]))
        target = 6000 if w['iterations'] == 2000 and continuation else w['iterations']
        start = time.perf_counter()
        # Atomic optimizer checkpoints every 1000 iterations bound restart loss.
        while solver.iteration < target:
            solver.advance(min(target,((solver.iteration//1000)+1)*1000))
            snapshot = solver.snapshot()
            for c,item in enumerate(items):
                value = dict(fingerprint=item['fingerprint'],state=split_state(snapshot,c),
                             attempts=(item['saved'] or {}).get('attempts',[]))
                save_state(state_path(w['work'],item['task']),value)
        results = solver.finish()
        snapshot = solver.snapshot()
        output = []
        for c,(item,(pose,metric)) in enumerate(zip(items,results)):
            attempts = (item['saved'] or {}).get('attempts',[])+[dict(iterations=target,accepted=metric['accepted'])]
            retry = bool(not metric['accepted'] and metric['can_continue'] and target == 2000 and w['iterations'] == 2000)
            metric.update(attempts=attempts, optimizer_continued=start_iteration>0,
                          batch_cameras=len(items), optimization_seconds_per_batch=time.perf_counter()-start)
            if not metric['accepted'] and not retry:
                pose = item['data']['initial']
                metric.update(status='official_initial_retained',reason='Unobservable robot or calibration quality gate failed')
            result = dict(task=item['task'],retry=retry,pose=pose.tolist(),metric=metric,frames=item['frames'])
            value = dict(fingerprint=item['fingerprint'],state=split_state(snapshot,c),attempts=attempts)
            if not retry:
                value['result'] = {k:v for k,v in result.items() if k != 'task'}
            save_state(state_path(w['work'],item['task']),value)
            output.append(result)
        return output
    finally:
        del solver
        # Autograd/graph objects can contain cycles; collect before an OOM split.
        gc.collect()
        torch.cuda.empty_cache()


def fit_with_backoff(items):
    import torch
    capacity = _worker.get('batch_size',len(items))
    if len(items) > capacity:
        return [r for start in range(0,len(items),capacity) for r in fit_with_backoff(items[start:start+capacity])]
    try:
        return _fit_group(items)
    except torch.cuda.OutOfMemoryError:
        pass
    # Leave the exception handler first: its traceback may retain CUDA tensors.
    gc.collect()
    torch.cuda.empty_cache()
    if len(items) == 1:
        return [dict(task=items[0]['task'],error='Insufficient free GPU memory even for one camera; free GPU memory and rerun')]
    middle = len(items)//2
    _worker['batch_size'] = min(capacity,middle)
    print(f'{_worker["device"]}: VRAM pressure; split {len(items)} cameras into {middle}+{len(items)-middle}',flush=True)
    # If OOM happened after a checkpoint, reload its state before continuing.
    for item in items:
        item['saved'] = load_state(state_path(_worker['work'],item['task']))
        if item['saved']:
            item['task']['iteration'] = item['saved']['state']['iteration']
    output = []
    for part in (items[:middle],items[middle:]):
        groups = defaultdict(list)
        for item in part:
            groups[item['task']['iteration']].append(item)
        for group in groups.values():
            output.extend(fit_with_backoff(group))
    return output


def process_batch(tasks, next_tasks):
    """CPU prefetch of exactly one reserved batch overlaps the current GPU work."""
    w = _worker
    import torch
    torch.cuda.reset_peak_memory_stats(w['device'])
    start = time.perf_counter()
    tag = [task_key(t) for t in tasks]
    if w['prefetched'] is not None:
        old_tag,future = w['prefetched']
        require(old_tag == tag,'Internal prefetch reservation mismatch')
        prepared = future.result()
    else:
        prepared = prepare_tasks(tasks)
    w['prefetched'] = ([task_key(t) for t in next_tasks],w['loader'].submit(prepare_tasks,next_tasks)) if next_tasks else None
    output,groups = [],defaultdict(list)
    for item in prepared:
        if 'error' in item:
            output.append(dict(task=item['task'],error=item['error']))
        elif item['saved'] and 'result' in item['saved']:
            output.append(dict(task=item['task'],**item['saved']['result']))
        else:
            retry_phase = bool((item['saved'] or {}).get('attempts'))
            shape = (item['data']['depths'][0].shape,item['data']['points'][0].shape,item['task']['iteration'],retry_phase)
            groups[shape].append(item)
    for group in groups.values():
        try:
            output.extend(fit_with_backoff(group))
        except Exception as exc:
            output.extend(dict(task=item['task'],error=str(exc)) for item in group)
    return dict(results=output,device=w['device'],pid=os.getpid(),elapsed_seconds=time.perf_counter()-start,
                cameras=len(tasks),retry_cameras=sum(bool(r.get('retry')) for r in output),
                effective_camera_limit=w['batch_size'],peak_allocated_gib=torch.cuda.max_memory_allocated(w['device'])/2**30)


def _run_reference(root, info, pending, urdf, devices, iterations):
    from .pipeline import calibrate_one
    queues = deque(pending)
    pools = [ProcessPoolExecutor(max_workers=1,mp_context=get_context('spawn')) for _ in devices]
    active,errors = {},[]
    def submit(slot):
        if queues:
            job,output = queues.popleft()
            future = pools[slot].submit(calibrate_one,(str(root),info,job,str(output),str(urdf),devices[slot],iterations))
            active[future] = (slot,job['episode_index'])
    try:
        for slot in range(len(pools)):
            submit(slot)
        completed = 0
        while active:
            for future in wait(active,return_when=FIRST_COMPLETED).done:
                slot,i = active.pop(future)
                try:
                    future.result()
                except Exception as exc:
                    errors.append(dict(episode_index=i,error=str(exc)))
                completed += 1
                print(f'Calibrate {completed}/{len(pending)}',flush=True)
                submit(slot)
    finally:
        for pool in pools:
            pool.shutdown(wait=True,cancel_futures=True)
    return errors


def run_calibrations(root, info, pending, urdf, work, args, backend):
    import torch
    devices = args.devices.split(',')
    require(len(set(devices)) == len(devices),'Duplicate GPUs in --devices')
    require(all(d == 'cpu' or (d.isdigit() and int(d)<torch.cuda.device_count()) for d in devices),
            'Unavailable GPU; check --devices and CUDA_VISIBLE_DEVICES')
    devices = ['cpu' if d == 'cpu' else 'cuda:'+d for d in devices]
    if backend == 'reference':
        errors = _run_reference(root,info,pending,urdf,devices,args.iterations)
    else:
        errors = _run_batched(root,info,pending,urdf,work,args,devices)
    write_json(Path(work)/'calibration_failures.json',errors)
    require(not errors,'Calibration jobs failed; inspect calibration_failures.json and rerun this step')


def _run_batched(root, info, pending, urdf, work, args, devices):
    queues = defaultdict(deque)
    for job,_ in pending:
        for cam in (1,2):
            task = dict(job=job,cam=cam,iteration=0)
            state = load_state(state_path(work,task))
            if state:
                task['iteration'] = state['state']['iteration']
            queues[task['iteration']].append(task)
    batch_size = getattr(args,'gpu_batch_size',0)
    require(batch_size >= 0,'--gpu-batch-size must be zero (auto) or positive')
    graphs = not getattr(args,'no_cuda_graphs',False)
    pools = [ProcessPoolExecutor(max_workers=1,mp_context=get_context('spawn'),initializer=init_worker,
              initargs=(str(root),info,str(urdf),str(work),d,args.iterations,batch_size,graphs)) for d in devices]
    active,reserved,errors,finished = {},{},[],{}
    outputs = {j['episode_index']:Path(p) for j,p in pending}
    start = time.perf_counter()
    def take(capacity):
        # Prefer full batches, including retry cameras from different episodes.
        sizes = [(len(q),age) for age,q in queues.items() if q]
        if not sizes:
            return []
        _,age = max(sizes)
        q = queues[age]
        return [q.popleft() for _ in range(min(capacity,len(q)))]
    try:
        initializers = [p.submit(worker_capacity) for p in pools]
        capacities = [f.result() for f in initializers]
        write_json(Path(work)/'calibration_workers.json',capacities)
        for c in capacities:
            print(f'{c["device"]}: batch up to {c["batch_size"]} cameras, free VRAM {c["free_vram_gib"]:.1f} GiB, CUDA graphs={graphs}',flush=True)
        def submit(slot,current=None):
            current = current or take(capacities[slot]['batch_size'])
            if current:
                following = take(capacities[slot]['batch_size'])
                reserved[slot] = following
                active[pools[slot].submit(process_batch,current,following)] = (slot,current)
        # Give every GPU its first batch before reserving prefetch batches.
        first = [take(c['batch_size']) for c in capacities]
        for slot,current in enumerate(first):
            if current:
                submit(slot,current)
        completed = 0
        log_path = Path(work)/'calibration_performance.jsonl'
        while active:
            for future in wait(active,return_when=FIRST_COMPLETED).done:
                slot,current = active.pop(future)
                try:
                    report = future.result()
                    capacities[slot]['batch_size'] = min(capacities[slot]['batch_size'],report.get('effective_camera_limit',capacities[slot]['batch_size']))
                    with log_path.open('a',encoding='utf-8') as f:
                        f.write(json.dumps({k:v for k,v in report.items() if k != 'results'})+'\n')
                    for result in report['results']:
                        task = result['task']
                        i,cam = task['job']['episode_index'],task['cam']
                        if 'error' in result:
                            errors.append(dict(episode_index=i,camera=cam,error=result['error']))
                        elif result['retry']:
                            task['iteration'] = 2000
                            queues[2000].append(task)
                        else:
                            finished[i,cam] = result
                            if (i,1) in finished and (i,2) in finished:
                                rr = [finished.pop((i,c)) for c in (1,2)]
                                write_json(outputs[i],dict(episode_index=i,source_episode_id=task['job']['source']['source_episode_id'],
                                    source='pointworld_method_droid_initialization',camera_to_base=[r['pose'] for r in rr],
                                    metrics=[r['metric'] for r in rr],sample_frame_indices=rr[0]['frames'],pointworld_commit=POINTWORLD_COMMIT))
                                completed += 1
                    print(f'Calibrate {completed}/{len(pending)} episodes; {report["device"]} {report["cameras"]} cameras '
                          f'in {report["elapsed_seconds"]:.1f}s, {report["retry_cameras"]} queued continuations; errors={len(errors)}',flush=True)
                except Exception as exc:
                    errors.extend(dict(episode_index=t['job']['episode_index'],camera=t['cam'],error=str(exc)) for t in current)
                following = reserved.pop(slot,[])
                submit(slot,following)
            # A GPU that drained its queue can take continuations produced later.
            busy = {slot for slot,_ in active.values()}
            for slot in range(len(pools)):
                if slot not in busy:
                    submit(slot)
        require(not any(queues.values()),'Internal scheduler left pending cameras')
        for i,path in outputs.items():
            if not path.exists() and not any(e['episode_index'] == i for e in errors):
                errors.append(dict(episode_index=i,error='Incomplete camera pair'))
        write_json(Path(work)/'calibration_timing.json',dict(episodes=completed,elapsed_seconds=time.perf_counter()-start,
            backend='batched',cuda_graphs=graphs,workers=capacities,failures=len(errors)))
    finally:
        for pool in pools:
            pool.shutdown(wait=True,cancel_futures=True)
    return errors
