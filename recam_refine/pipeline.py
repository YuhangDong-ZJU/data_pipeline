"""Resumable ReCam refinement orchestration."""
from __future__ import annotations

from .progress import phase, tracked
from .parallel import io_map
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from functools import lru_cache, partial
import os
import io
import hashlib
from pathlib import Path
import shutil

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from .archives import checked_png_array, frame_files, unpack_archive
from .common import (Journal, array_hash, preserved_hashes, check_transform, media_path, parquet_path, read_json,
                     read_jsonl, require, safe_path, set_values, sha256, values,
                     sync_dir, write_json, write_jsonl, acquire_directory_lock, validate_lock_mount)
from .inputs import canonical_manifest, download_inputs, load_depth_records
from .media import trim_video, video_info
from .pointworld import Robot, prepare_assets, refine_camera_with_retry, release_pose, POINTWORLD_COMMIT, CalibrationRejected
from .stats import aggregate, table_stats


def log(message):
    print(message, flush=True)


def discover(root):
    result = sorted(p.parent.parent for p in Path(root).glob("*/*/meta/info.json"))
    require(result, f"No LeRobot subsets found under {root}; expected simulation/* and real_world/*")
    for domain in (Path(root) / "simulation", Path(root) / "real_world"):
        if domain.is_dir():
            for child in domain.iterdir():
                if child.is_dir() and any((child / marker).exists() for marker in ("data", "images", "videos", "meta")):
                    require(child in result, f"Incomplete/unrecognized LeRobot subset (missing meta/info.json): {child}")
    for subset in result:
        info = read_json(subset / "meta/info.json")
        require(info.get("codebase_version") == "v2.1", f"Only LeRobot v2.1 supported: {subset}")
    return result


def corrected_droid_info(root, info):
    info = deepcopy(info)
    features = info["features"]
    for k in list(features):
        if k in ("observation.images.depth_00", "observation.images.normal_00") or k.startswith("observation.images.normals_"):
            del features[k]
    for cam in (1, 2):
        key = f"observation.images.normal_{cam:02d}"
        paths = sorted((root / "videos").glob(f"chunk-*/{key}/episode_*.mp4"))
        require(paths, f"Missing NormalCrafter {key}. Place the completed normal outputs under {root / 'videos'}.")
        probe = video_info(paths[0])
        features[key] = dict(dtype="video", shape=[probe["height"], probe["width"], 3], names=["height", "width", "channel"],
                             info={"video.height":probe["height"], "video.width":probe["width"],
                                   "video.fps":info["fps"], "video.channels":3, "video.codec":probe["codec"],
                                   "video.pix_fmt":probe["pix_fmt"], "video.is_depth_map":False, "has_audio":False,
                                   "source":"NormalCrafter", "normal_convention":"native NormalCrafter view space; RGB=(normal+1)/2"})
        features[f"observation.images.depth_{cam:02d}"]["info"] = {
            "encoding":"png", "pixel_dtype":"uint16", "source":"FoundationStereo", "unit":"millimeter",
            "metric_decode":"depth_m = pixel_value / 1000", "invalid_value":0}
    return info


def plan_episode(args):
    root, info, episode, row, records = args
    root = Path(root)
    i = episode["episode_index"]
    p = parquet_path(root, info, i)
    parquet_bytes = p.read_bytes()
    table = pq.read_table(io.BytesIO(parquet_bytes))
    length = int(episode["length"])
    original = int(row["length"])
    require(0 <= original - len(table) <= 2 and 0 <= original - length <= 2,
            f"Unexplained Parquet/meta lengths: episode {i}")
    require(np.array_equal(values(table["frame_index"]).ravel(), np.arange(len(table))), f"Non-prefix frame_index: {p}")
    require(np.all(values(table["episode_index"]) == i), f"Wrong episode_index: {p}")
    n = min(length, len(table))
    reasons = []
    depths = {}
    for cam in (1, 2):
        key = f"observation.images.depth_{cam:02d}"
        paths = frame_files(media_path(root, info, i, key, 0).parent)
        require(0 <= original - len(paths) <= 2, f"Depth length exceeds the two-frame tail rule: {paths[0].parent}")
        n = min(n, len(paths))
        rec = records.get((i, cam))
        if rec:
            require(0 <= original - rec["frame_count"] <= 2, f"Wrong depth sidecar length: episode {i}/{cam}")
            n = min(n, rec["decoded"])
            if rec["missing"]:
                reasons.append(dict(camera=cam, reason="recorded_SVO_tail_padding", frames=rec["missing"], source=rec["path"]))
        # The generation code writes zeros, not repeated pictures. Missing old
        # sidecars can therefore be handled as explicitly invalid-depth tails.
        # This is NOT called proof of SVO padding; the audit records the reason.
        zero_tail = 0
        for p in reversed(paths[-3:]):
            is_zero = not checked_png_array(p, info["features"][key]["shape"]).any()
            if not is_zero:
                break
            zero_tail += 1
        require(zero_tail <= 2, f"More than two all-zero tail frames: {paths[0].parent}")
        if zero_tail:
            n = min(n, len(paths) - zero_tail)
            reasons.append(dict(camera=cam, reason="all_zero_invalid_depth_tail", frames=zero_tail))
        depths[str(cam)] = dict(count=len(paths), record=rec)
    require(n > 0 and original - n <= 2, f"Refusing to remove >2 total timesteps: episode {i}")
    streams = {}
    for key, feature in info["features"].items():
        if feature["dtype"] == "video":
            path = media_path(root, info, i, key)
            require(path.is_file(), f"Missing modality: {path}")
            probe = video_info(path)
            # If container counts are absent, decode and count before planning.
            if probe["frames"] == 0:
                import av
                with av.open(str(path)) as container:
                    probe["frames"] = sum(1 for _ in container.decode(video=0))
            require(n <= probe["frames"] <= original, f"Video length cannot align without inventing frames: {path}: {probe['frames']} -> {n}")
            require([probe["height"], probe["width"], 3] == feature["shape"], f"Video shape mismatch: {path}")
            require(abs(probe["fps"] - info["fps"]) < 1e-5, f"Video FPS mismatch: {path}")
            streams[key] = probe
    k = values(table["observation.camera.intrinsics"])
    e = check_transform(values(table["observation.camera.extrinsics"]), p)
    require(e.shape == (len(table), 3, 4, 4) and k.shape == (len(table), 3, 3, 3), f"Wrong camera array shape: {p}")
    require(np.allclose(e[:, 1:], e[0, 1:], atol=1e-5), f"External cameras are not constant: {p}")
    require(np.isfinite(k).all() and np.all(k[:, :, [0,1], [0,1]] > 0), f"Invalid intrinsics: {p}")
    return dict(episode_index=i, length=n, previous_length=length, original_length=original,
                original_parquet_sha256=hashlib.sha256(parquet_bytes).hexdigest(), source=row, reasons=reasons,
                depths=depths, videos=streams, initial_intrinsics=k[0].tolist(), initial_extrinsics=e[0].tolist())


from .transfer import transfer_depth


@lru_cache(maxsize=1)
def calibration_robot(urdf):
    return Robot(urdf)


def calibrate_one(args):
    root, info, job, output, urdf, device, iterations = args
    output = Path(output)
    if output.exists():
        return read_json(output)
    i = job["episode_index"]
    expected = job.get('calibration_inputs')
    if expected:
        import hashlib
        import io
        raw = parquet_path(root,info,i).read_bytes()
        require(hashlib.sha256(raw).hexdigest()==expected['parquet_sha256'],f'Shard Parquet changed: episode {i}')
        table = pq.read_table(io.BytesIO(raw)).slice(0,job['length'])
    else:
        table = pq.read_table(parquet_path(root, info, i)).slice(0, job["length"])
    joints = values(table["observation.states.joint_state"])
    gripper = values(table["observation.states.gripper_state"]).ravel()
    require(np.isfinite(joints).all() and np.isfinite(gripper).all() and np.all((gripper >= 0) & (gripper <= 1)), f"Invalid robot configuration: {i}")
    indices = np.unique(np.linspace(0, job["length"] - 1, min(16, job["length"]), dtype=int))
    require(not expected or expected['frames']==indices.tolist(),'Shard frame selection changed')
    robot = calibration_robot(urdf)
    points = [robot.points(joints[t], gripper[t]) for t in indices]
    poses, metrics = [], []
    for cam in (1, 2):
        k = np.asarray(job["initial_intrinsics"][cam]).copy()
        rec = job["depths"][str(cam)]["record"]
        if rec:
            k = np.asarray(rec["intrinsic"])
        k[:2] *= .5
        depths = []
        for index,t in enumerate(indices):
            path = media_path(root, info, i, f"observation.images.depth_{cam:02d}", int(t))
            if expected:
                raw = path.read_bytes()
                require(hashlib.sha256(raw).hexdigest()==expected['depth_sha256'][str(cam)][index],
                        f'Shard depth changed: episode {i} camera {cam} frame {t}')
                path = io.BytesIO(raw)
            with Image.open(path) as im:
                # Preserve pixel centers using explicit nearest decimation.
                depths.append(np.asarray(im, dtype=np.float32)[::2, ::2] / 1000.)
        try:
            pose, metric = refine_camera_with_retry(np.asarray(job["initial_extrinsics"][cam]), k, depths, points,
                                                    device=device, iterations=iterations)
        except CalibrationRejected as exc:
            pose = np.asarray(job["initial_extrinsics"][cam])
            metric = dict(accepted=False, status="official_initial_retained", reason=str(exc), failed_metrics=exc.metrics)
        poses.append(pose.tolist())
        metrics.append(metric)
    value = dict(episode_index=i, source_episode_id=job["source"]["source_episode_id"],
                 source="pointworld_method_droid_initialization", camera_to_base=poses, metrics=metrics,
                 sample_frame_indices=indices.tolist(), pointworld_commit=POINTWORLD_COMMIT)
    write_json(output, value)
    return value


def apply_episode(args):
    root, droid, info, job, camera_result, offset, work, *options = args
    root, droid, work = Path(root), Path(droid), Path(work)
    journal = Journal(root, work)
    i, n = job["episode_index"], job["length"]
    marker = work / (options[0] if options else "applied") / f"episode_{i:06d}.json"
    if marker.exists():
        return read_json(marker)
    for key, original in job["videos"].items():
        path = media_path(droid, info, i, key)
        probe = video_info(path)
        if probe["frames"] == n:
            continue
        require(probe["frames"] == original["frames"], f"Video changed during refine: {path}")
        staged = path.with_name("." + path.stem + ".refine-part.mp4")
        trim_video(path, staged, n, info["fps"])
        journal.replace(path, staged)
    for cam in (1, 2):
        directory = media_path(droid, info, i, f"observation.images.depth_{cam:02d}", 0).parent
        for path in frame_files(directory)[n:]:
            journal.retire(path, "removed_tails")
    path = parquet_path(droid, info, i)
    saved = journal.backup(path)
    # Always derive from the original, including after an interrupted run.
    raw = saved.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == job["original_parquet_sha256"], f"Original Parquet identity changed: {path}")
    table = pq.read_table(io.BytesIO(raw)).slice(0, n)
    unchanged = preserved_hashes(table)
    wrist_hash = array_hash(values(table["observation.camera.extrinsics"])[:, 0])
    table = set_values(table, "index", np.arange(offset, offset + n))
    extrinsics = values(table["observation.camera.extrinsics"]).copy()
    if camera_result['source']!='droid_initial_alignment_only':
        extrinsics[:, 1:] = np.asarray(camera_result["camera_to_base"])
    table = set_values(table, "observation.camera.extrinsics", extrinsics)
    intrinsics = values(table["observation.camera.intrinsics"]).copy()
    for cam in (1, 2):
        rec = job["depths"][str(cam)]["record"]
        if rec:
            intrinsics[:, cam] = rec["intrinsic"]
    table = set_values(table, "observation.camera.intrinsics", intrinsics)
    # Recorded terminal flags describe success, not whether this file ended.
    # Do not fabricate successful termination after discarding invalid frames.
    staged = path.with_name("." + path.name + ".refine-part")
    pq.write_table(table, staged, compression="zstd")
    require(pq.read_table(staged).equals(table), f"Parquet round-trip mismatch: {path}")
    journal.replace(path, staged)
    result = dict(episode_index=i, stats=table_stats(table), preserved_columns_sha256=unchanged,
                  wrist_extrinsics_sha256=wrist_hash)
    write_json(marker, result)
    return result


def update_metadata(root, droid, info, jobs, stats, work, camera_directory=None, calibration_pending=False):
    phase('更新 metadata（任务）',0,1)
    journal = Journal(root, work)
    episodes = read_json(work / "original_droid_episodes.json")
    by_id = {j["episode_index"]:j for j in jobs}
    for e in episodes:
        e["length"] = by_id[e["episode_index"]]["length"]
    info["total_frames"] = sum(j["length"] for j in jobs)
    info["total_images"] = info["total_frames"] * sum(v["dtype"] == "image" for v in info["features"].values())
    info["total_videos"] = len(jobs) * sum(v["dtype"] == "video" for v in info["features"].values())
    info["total_episodes"] = len(jobs)
    info["total_chunks"] = len({j["episode_index"] // info["chunks_size"] for j in jobs})
    cameras = deepcopy(read_json(droid / "meta/cameras.json"))
    for camera in cameras["cameras"]:
        i = camera["camera_index"]
        camera["depth_key"] = f"observation.images.depth_{i:02d}" if i else None
        camera["normal_key"] = f"observation.images.normal_{i:02d}" if i else None
    cameras["depth"] = dict(feature_prefix="observation.images.depth_", source="FoundationStereo",
                            camera_indices=[1, 2], method="FoundationStereo stereo inference",
                            calibration_source="native rectified SVO calibration_parameters",
                            measurement="distance_to_image_plane", source_unit="millimeter",
                            valid_range_meters=[.02, 10.],
                            storage={"format":"png", "pixel_dtype":"uint16", "invalid_value":0, "scale":1000.,
                                     "quantization":"round", "out_of_range":"set_to_zero",
                                     "metric_decode":"depth_m = pixel_value / 1000.0"})
    cameras.pop("normal", None)
    cameras["normals"] = dict(source="NormalCrafter", feature_prefix="observation.images.normal_", camera_indices=[1, 2],
                              convention="native NormalCrafter view space; RGB=(normal+1)/2",
                              approximate_decode="normal_native = normalize(2 * decoded_RGB / 255 - 1)",
                              storage={"format":"video", "codec":"h264", "pixel_dtype":"uint8", "channels":3,
                                       "encoding":"RGB = (normal_native + 1) / 2",
                                       "approximate_decode":"normal_native = normalize(2 * decoded_RGB / 255 - 1)",
                                       "note":"Lossy original annotation, lossless tail trimming"},
                              coordinate_note="Native NormalCrafter view-space axes; no unverified OpenCV axis conversion is asserted.")
    cameras["calibration"]["source"] = ('DROID initial calibration; external refinement pending'
        if calibration_pending else 'DROID wrist calibration; PointWorld release or PointWorld depth alignment for external cameras')
    cameras["calibration"]["extrinsics"]["convention"] = "camera_to_robot_base"
    cameras["calibration"]["provenance_path"] = "meta/refinement.jsonl"
    journal.json(droid / "meta/cameras.json", cameras)
    journal.json(droid / "meta/info.json", info)
    journal.json(droid / "meta/episodes.jsonl", episodes, lines=True)
    journal.json(droid / "meta/episodes_stats.jsonl", [dict(episode_index=s["episode_index"], stats=s["stats"]) for s in stats], lines=True)
    journal.json(droid / "meta/stats.json", aggregate(stats))
    provenance = []
    stat_by_id = {s["episode_index"]:s for s in stats}
    for job in jobs:
        camera = read_json((camera_directory or work / "cameras") / f"episode_{job['episode_index']:06d}.json")
        provenance.append(dict(episode_index=job["episode_index"], source_episode_id=job["source"]["source_episode_id"],
                               camera_serials=job["source"]["camera_serials"], length=job["length"],
                               original_length=job["original_length"], retained_source_frame_range=[0, job["length"]],
                               tail_reasons=job["reasons"], calibration_source=camera["source"],
                               calibration_metrics=camera.get("metrics"), pointworld_commit=POINTWORLD_COMMIT,
                               external_camera_to_base=camera["camera_to_base"],
                               preserved_columns_sha256=stat_by_id[job["episode_index"]]["preserved_columns_sha256"],
                               wrist_extrinsics_sha256=stat_by_id[job["episode_index"]]["wrist_extrinsics_sha256"]))
    journal.json(droid / "meta/refinement.jsonl", provenance, lines=True)
    # The immutable input manifest remains outside the dataset. Publish a new,
    # aligned manifest rather than rewriting historical source lengths in place.
    write_jsonl(work / "refined_episode_manifest.jsonl", [dict(j["source"], length=j["length"],
                before_refine_length=j["original_length"]) for j in jobs])
    phase('更新 metadata（任务）',1,1)



def _cleanup_source(rec, work):
    src, dst = Path(rec["source"]), Path(rec["target"])
    if not src.exists():
        return
    for p in sorted(src.glob("frame_*.png")):
        target = dst / p.name
        if target.exists():
            from .archives import _same_bytes
            require(_same_bytes(p, target), f"Source/final depth differs: {p}")
        else:
            backup = work / "source_removed_tails" / f"episode_{rec['episode_index']:06d}" / str(rec["camera"]) / p.name
            backup.parent.mkdir(parents=True, exist_ok=True)
            if not backup.exists():
                shutil.copy2(p, backup)
            from .archives import _same_bytes
            require(_same_bytes(p, backup), f"Tail backup differs: {p}")
        p.unlink()
    src.rmdir()


def finalize(root, droid, subsets, work):
    """Called only after ALL subsets pass full decoding and metadata checks."""
    journal = Journal(root, work)
    moved = []
    # Caller has completed frame, metadata and geometry validation.
    # Keep simulation and other real-world TARs. DROID TARs are obsolete after
    # trimming; remove only archives whose extraction receipt was verified.
    for receipt in tracked(read_json(work / 'unpacked.json'),'清理：TAR 记录'):
        p = Path(receipt["archive"])
        if p.is_relative_to(droid) and p.exists():
            if receipt.get('archive_state') is not None:
                from .archives import _file_state, _same_state
                require(_same_state(_file_state(p), receipt['archive_state']), f'Archive changed during processing: {p}')
            else:
                # Old receipts without a file-state snapshot need one identity
                # check before deleting their archive; new receipts use no hash.
                require(sha256(p) == receipt["sha256"], f"Archive changed during processing: {p}")
            p.unlink()
    # Discard redundant source copies only after verifying the retained prefix
    # against the final dataset and saving trimmed source frames for recovery.
    if (work / "depth_transfer.json").exists():
        io_map(partial(_cleanup_source, work=work),
               read_json(work / 'depth_transfer.json'), 8, '清理源副本（相机序列）')
    # Retire obsolete wrist depth/normal and old plural normal annotations.
    for folder in ("images", "videos"):
        for chunk in (droid / folder).glob("chunk-*"):
            for camera in chunk.iterdir():
                if camera.name in ("observation.images.depth_00", "observation.images.normal_00") or camera.name.startswith("observation.images.normals_"):
                    if camera.is_dir():
                        for p in tracked(sorted(camera.rglob('*')),'移出旧模态：目录项'):
                            if p.is_file():
                                journal.retire(p, "unused_modalities")
    # Keep the training tree at subset level to data/images/videos/meta. Save
    # dataset cards, download manifests, previews and other auxiliary entries
    # outside it, including files whose names do not contain the word "log".
    containers = {root:set()}
    for subset in subsets:
        current = root
        for part in subset.relative_to(root).parts:
            containers.setdefault(current, set()).add(part)
            current = current / part
        containers[subset] = {"data", "images", "videos", "meta"}
    for container, allowed in containers.items():
        for entry in list(container.iterdir()):
            if entry.name in allowed:
                continue
            require(not entry.is_symlink(), f"Symlink in auxiliary files: {entry}")
            files = [entry] if entry.is_file() else sorted(p for p in entry.rglob("*") if p.is_file())
            for p in tracked(files,'移出辅助内容：文件'):
                journal.retire(p, "auxiliary")
                moved.append(str(p.relative_to(root)))
    known_dirs = {"logs", "log", "annotations", "__pycache__", ".cache", ".huggingface"}
    # Walk directory names, pruning training media so this does not traverse
    # tens of millions of PNG paths just to discover logs.
    for current, dirs, files in os.walk(root, followlinks=False):
        current = Path(current)
        for name in list(dirs):
            path = current / name
            require(not path.is_symlink(), f"Symlink in dataset: {path}")
            if name in known_dirs:
                for p in tracked(sorted(path.rglob('*')),'移出辅助目录：目录项'):
                    if p.is_file():
                        journal.retire(p, "auxiliary")
                        moved.append(str(p.relative_to(root)))
                dirs.remove(name)
            elif name in ("images", "videos", "data"):
                dirs.remove(name)
        for name in files:
            if name.endswith((".log", ".pid", ".out", ".err")) or ".log." in name or name.endswith("failures.jsonl"):
                p = current / name
                journal.retire(p, "auxiliary")
                moved.append(str(p.relative_to(root)))
    write_json(work / "moved_auxiliary.json", moved)


def run(args):
    """Lock both the work directory and the dataset, even across work dirs."""
    import fcntl
    root, work = args.root.resolve(), args.work_dir.resolve()
    require(root.is_dir(), f"Missing dataset root: {root}")
    require(not work.is_relative_to(root) and not root.is_relative_to(work), "work-dir must be outside and separate from recam_lerobot")
    work.mkdir(parents=True, exist_ok=True)
    validate_lock_mount(work)
    with (work / "run.lock").open("a+") as lock:
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquire_directory_lock(fd, fcntl.LOCK_EX)
            except BlockingIOError:
                raise RuntimeError("Another refinement is using this dataset or work directory")
            _run_locked(args)
        finally:
            os.close(fd)


def _run_locked(args):
    from .validate import check_subset
    root, work = args.root.resolve(), args.work_dir.resolve()
    require(not (work/'manual_workflow.json').exists(), 'This work directory uses manual steps. Use run_step.sh instead of the automatic run entry.')
    require(root.is_dir(), f"Missing dataset root: {root}")
    require(not work.is_relative_to(root) and not root.is_relative_to(work), "work-dir must be outside and separate from recam_lerobot")
    work.mkdir(parents=True, exist_ok=True)
    droid = root / "real_world/droid"
    require(droid.is_dir(), f"Expected {droid}")
    camera_meta = read_json(droid / "meta/cameras.json")
    convention = camera_meta.get("calibration", {}).get("extrinsics", {}).get("convention", "camera_to_robot_base")
    require(convention == "camera_to_robot_base", f"Unsupported input extrinsics convention: {convention}")
    subsets = discover(root)
    config = dict(root=str(root), depth_output=str(args.depth_output.resolve()) if args.depth_output else None,
                  depth_chunks=args.depth_chunks, depth_metadata=sorted(str(p.resolve()) for p in args.depth_metadata),
                  episode_manifest=str(args.episode_manifest.resolve()) if args.episode_manifest else None,
                  pointworld_cameras=str(args.pointworld_cameras.resolve()) if args.pointworld_cameras else None,
                  iterations=args.iterations, version=1)
    config_path = work / "configuration.json"
    from .calibration import run_calibrations, select_backend, BATCHED_BACKEND_VERSION
    saved = read_json(config_path) if config_path.exists() else None
    backend = select_backend(args,saved)
    if saved is None or 'backend' in saved:
        config['backend'] = backend
    if backend == 'batched':
        config['backend_version'] = BATCHED_BACKEND_VERSION
    if config_path.exists():
        require(read_json(config_path) == config, f"Configuration differs from saved run: {config_path}. Resume with the original arguments.")
    else:
        write_json(config_path, config)
    if (work / "SUCCESS.json").exists():
        log(f"This run already completed: {work / 'SUCCESS.json'}. Use the check command for a fresh integrity check.")
        return
    def done(stage):
        return (work / (stage + ".complete.json")).exists()
    def finish(stage):
        write_json(work / (stage + ".complete.json"), {"complete":True})
    if not (work / "original_droid_episodes.json").exists():
        episodes = read_jsonl(droid / "meta/episodes.jsonl")
        require([e["episode_index"] for e in episodes] == list(range(len(episodes))), "DROID episode catalogue must be complete and ordered")
        write_json(work / "original_droid_episodes.json", episodes)
        write_json(work / "original_droid_info.json", read_json(droid / "meta/info.json"))
    episodes = read_json(work / "original_droid_episodes.json")
    info = corrected_droid_info(droid, read_json(work / "original_droid_info.json"))
    manifest_path, camera_dir = args.episode_manifest, args.pointworld_cameras
    if manifest_path is None or camera_dir is None:
        auto_manifest, auto_cameras = download_inputs(work, sorted({int(e.get('source_episode_index', e['episode_index'])) // info["chunks_size"] for e in episodes}))
        manifest_path = manifest_path or auto_manifest
        camera_dir = camera_dir or auto_cameras
    manifest = canonical_manifest(manifest_path, episodes)
    require(Path(camera_dir).is_dir(), f"Missing PointWorld cameras directory: {camera_dir}")
    if not done("01_unpack"):
        receipts = []
        for subset in subsets:
            for archive in sorted((subset / "images").glob("chunk-*/observation.images.depth_*/*.tar")):
                if subset == droid and archive.parent.name == "observation.images.depth_00":
                    continue
                log(f"Unpack {archive}")
                receipts.append(unpack_archive(archive, subset, work / "archive_receipts"))
        write_json(work / "unpacked.json", receipts)
        finish("01_unpack")
    if not done("02_transfer"):
        if args.depth_output:
            transfer_depth(root, droid, args.depth_output, manifest, parse_chunks(args.depth_chunks), work, workers=args.workers)
        finish("02_transfer")
    if not done("03_plan"):
        records = load_depth_records([droid, *args.depth_metadata, *([args.depth_output] if args.depth_output else [])], manifest)
        grouped = {}
        for key, value in records.items():
            grouped.setdefault(key[0], {})[key] = value
        inputs = [(str(droid), info, e, manifest[e["episode_index"]], grouped.get(e["episode_index"], {})) for e in episodes]
        jobs = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for n, job in enumerate(pool.map(plan_episode, inputs, chunksize=1), 1):
                jobs.append(job)
                if n % 100 == 0 or n == len(episodes):
                    log(f"Plan {n}/{len(episodes)}")
        write_json(work / "plan.json", jobs)
        write_json(work / "training_info.json", info)
        finish("03_plan")
    jobs = read_json(work / "plan.json")
    info = read_json(work / "training_info.json")
    if not done("04_cameras"):
        pending, overlap = [], Counter()
        for job in jobs:
            result = work / "cameras" / f"episode_{job['episode_index']:06d}.json"
            poses, status = release_pose(camera_dir, job["source"])
            overlap[status] += 1
            if result.exists():
                continue
            if poses is not None:
                write_json(result, dict(episode_index=job["episode_index"], source_episode_id=job["source"]["source_episode_id"],
                                        source=status, camera_to_base=poses.tolist()))
            else:
                pending.append((job, result))
        write_json(work / "pointworld_overlap.json", dict(overlap))
        log(f"PointWorld overlap: {dict(overlap)}; pending optimizations: {len(pending)}")
        if pending:
            urdf = prepare_assets(work / "pointworld")
            run_calibrations(droid,info,pending,urdf,work,args,backend)
        finish("04_cameras")
    if not done("05_apply"):
        offset, tasks = 0, []
        for job in jobs:
            camera = read_json(work / "cameras" / f"episode_{job['episode_index']:06d}.json")
            tasks.append((str(root), str(droid), info, job, camera, offset, str(work)))
            offset += job["length"]
        stats = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for n, result in enumerate(pool.map(apply_episode, tasks, chunksize=1), 1):
                stats.append(result)
                if n % 25 == 0 or n == len(jobs):
                    log(f"Apply aligned prefix {n}/{len(jobs)}")
        update_metadata(root, droid, info, jobs, stats, work)
        finish("05_apply")
    if not done("06_check"):
        summaries = [check_subset(s, work / "checks", args.workers, droid=s == droid) for s in subsets]
        write_json(work / "checks.json", summaries)
        finish("06_check")
    if getattr(args, "defer_cleanup", False):
        write_json(work / "READY_FOR_GEOMETRY_AUDIT.json", dict(root=str(root), full_decode=True))
        log("Media and metadata checks complete; awaiting geometry audit before archive/source cleanup.")
        return
    if not done("07_cleanup"):
        finalize(root, droid, subsets, work)
        finish("07_cleanup")
    retained = []
    for job in jobs:
        camera = read_json(work / "cameras" / f"episode_{job['episode_index']:06d}.json")
        for index, metric in enumerate(camera.get("metrics", []), 1):
            if metric.get("accepted") is False:
                retained.append(dict(episode_index=job["episode_index"], camera=index, **metric))
    write_json(work / "retained_official_calibrations.json", retained)
    write_json(work / "SUCCESS.json", dict(root=str(root), subsets=read_json(work / "checks.json"),
                pointworld=read_json(work / "pointworld_overlap.json"),
                dropped_timesteps=sum(j["previous_length"]-j["length"] for j in jobs),
                backup_directory=str(work / "original"), full_decode=True,
                all_external_calibrations_accepted=not retained, retained_official_cameras=len(retained)))
    (work / "FAILED.json").unlink(missing_ok=True)
    (work / "READY_FOR_GEOMETRY_AUDIT.json").unlink(missing_ok=True)
    log(f"COMPLETE: {work / 'SUCCESS.json'}")
    if retained:
        log(f"{len(retained)} camera refinements did not pass geometric acceptance; official initial poses were retained. See retained_official_calibrations.json.")


def parse_chunks(value):
    import re
    result = set()
    for part in value.split(","):
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        require(m, f"Invalid chunks: {value}")
        a, b = int(m[1]), int(m[2] or m[1])
        require(0 <= a <= b, f"Invalid chunks: {value}")
        result.update(range(a, b + 1))
    return result
