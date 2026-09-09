"""Full streaming integrity checks, including every decoded video/PNG frame."""
from __future__ import annotations

from .progress import tracked
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import re

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from .archives import check_png, frame_files
from .common import array_hash, preserved_hashes, check_transform, media_path, parquet_path, read_json, read_jsonl, require, values, write_json
from .media import decode_check
from .stats import aggregate, table_stats, Moments


def compare_stats(a, b, label):
    require(a.keys() == b.keys(), f"Statistics feature mismatch: {label}")
    for k in a:
        for field in ("min", "max", "mean", "std", "count"):
            aa, bb = np.asarray(a[k][field]), np.asarray(b[k][field])
            require(aa.shape == bb.shape and np.allclose(aa, bb, rtol=2e-5, atol=1e-6), f"Stale stats: {label}/{k}/{field}")


def validate_episode(args):
    subset, info, e, offset, expected_stats, strict_stats, task_ids, provenance = args
    subset = Path(subset)
    i, n = int(e["episode_index"]), int(e["length"])
    path = parquet_path(subset, info, i)
    table = pq.read_table(path)
    require(len(table) == n, f"Parquet/meta mismatch: {path}")
    require(np.all(values(table["episode_index"]) == i), f"Wrong episode_index: {path}")
    require(np.array_equal(values(table["frame_index"]).ravel(), np.arange(n)), f"Invalid frame_index: {path}")
    require(np.array_equal(values(table["index"]).ravel(), np.arange(offset, offset+n)), f"Invalid global index: {path}")
    require(np.allclose(values(table["timestamp"]).ravel(), np.arange(n) / info["fps"], atol=1e-4, rtol=0), f"Misaligned timestamp: {path}")
    require(set(values(table["task_index"]).ravel().tolist()) <= task_ids, f"Invalid task_index: {path}")
    stats = table_stats(table)
    for k in table.column_names:
        require(k in info["features"], f"Parquet column absent from info.features: {path}:{k}")
        shape = list(values(table[k]).shape[1:])
        require(shape == info["features"][k]["shape"], f"Numerical shape mismatch: {path}:{k}")
    if "observation.camera.extrinsics" in table.column_names:
        extrinsics = check_transform(values(table["observation.camera.extrinsics"]), path)
        if strict_stats:
            require(np.allclose(extrinsics[:, 1:], extrinsics[0, 1:], atol=1e-5), f"External camera moves: {path}")
            require(provenance["length"] == n, f"Stale refinement provenance: {path}")
            require(np.allclose(extrinsics[:, 1:], provenance["external_camera_to_base"], atol=2e-6), f"Applied calibration differs from provenance: {path}")
            require(array_hash(values(table["observation.camera.extrinsics"])[:, 0]) == provenance["wrist_extrinsics_sha256"], f"Wrist extrinsics changed: {path}")
            require(preserved_hashes(table) == provenance["preserved_columns_sha256"], f"Robot/action/timestamp data changed beyond prefix slicing: {path}")
    if "observation.camera.intrinsics" in table.column_names:
        k = values(table["observation.camera.intrinsics"])
        require(np.isfinite(k).all() and np.all(k[..., [0,1], [0,1]] > 0), f"Bad intrinsics: {path}")
        require(np.allclose(k[..., 2, :], [0,0,1], atol=1e-5), f"Bad pinhole matrix: {path}")
    png_count = video_count = 0
    for key, feature in info["features"].items():
        kind = feature["dtype"]
        if kind == "video":
            result = decode_check(media_path(subset, info, i, key), n, feature["shape"], info["fps"],
                                  pixel_stats=key in expected_stats)
            if result is not None:
                stats[key] = result
            video_count += 1
        elif kind == "image":
            files = frame_files(media_path(subset, info, i, key, 0).parent)
            require(len(files) == n, f"Image/meta length mismatch: {files[0].parent}")
            moments = Moments() if key in expected_stats else None
            for frame, p in enumerate(files):
                check_png(p, feature["shape"], depth="depth_" in key)
                with Image.open(p) as im:
                    a = np.asarray(im)
                    require(a.size and np.isfinite(a).all(), f"Invalid PNG: {p}")
                    if strict_stats and "depth_" in key:
                        require(a.any(), f"All-zero retained depth frame: {p}")
                        require(np.all((a == 0) | ((a >= 20) & (a <= 10000))), f"Depth outside FoundationStereo range: {p}")
                    if moments:
                        moments.add(a.reshape(-1, feature["shape"][-1]), frames=1)
                png_count += 1
            if moments:
                stats[key] = moments.result(media=True)
        else:
            require(key in table.column_names, f"Missing declared numerical feature: {key} in {path}")
    if strict_stats:
        compare_stats(stats, expected_stats, path)
    else:
        # Simulation/other real datasets keep their original normalization
        # policy; validate exact lowdim stats for the features already recorded.
        lowdim = {k:v for k,v in expected_stats.items() if k in table.column_names}
        compare_stats({k:stats[k] for k in lowdim}, lowdim, path)
    return dict(episode_index=i, frames=n, pngs=png_count, videos=video_count, stats=stats)


def check_subset(subset, work, workers=4, droid=False):
    subset, work = Path(subset), Path(work)
    info = read_json(subset / "meta/info.json")
    episodes = read_jsonl(subset / "meta/episodes.jsonl")
    ids = [e["episode_index"] for e in episodes]
    require(ids == list(range(len(episodes))), f"Episode IDs not contiguous/ordered: {subset}")
    require(len(episodes) == info["total_episodes"], f"Wrong total_episodes: {subset}")
    frames = sum(e["length"] for e in episodes)
    require(frames == info["total_frames"], f"Wrong total_frames: {subset}")
    videos = sum(v["dtype"] == "video" for v in info["features"].values())
    images = sum(v["dtype"] == "image" for v in info["features"].values())
    require(info["total_videos"] == videos * len(episodes), f"Wrong total_videos: {subset}")
    require(info["total_images"] == images * frames, f"Wrong total_images: {subset}")
    require(info["total_chunks"] == len({i // info["chunks_size"] for i in ids}), f"Wrong total_chunks: {subset}")
    for split, spec in info["splits"].items():
        require(re.fullmatch(r"\d+:\d+", spec), f"Unsupported split specification: {split}={spec}")
        start, end = map(int, spec.split(":"))
        require(0 <= start < end <= len(episodes), f"Invalid split bounds: {split}")
    tasks = read_jsonl(subset / "meta/tasks.jsonl")
    task_ids = {t["task_index"] for t in tasks}
    require(len(tasks) == len(task_ids) == info["total_tasks"], f"Invalid task catalogue: {subset}")
    stats_rows = read_jsonl(subset / "meta/episodes_stats.jsonl")
    require([r["episode_index"] for r in stats_rows] == ids, f"Missing/duplicate episode statistics: {subset}")
    stats = {r["episode_index"]:r["stats"] for r in stats_rows}
    provenance = {}
    if droid:
        provenance_rows = read_jsonl(subset / "meta/refinement.jsonl")
        require([r["episode_index"] for r in provenance_rows] == ids, f"Missing/duplicate calibration provenance: {subset}")
        provenance = {r["episode_index"]:r for r in provenance_rows}
    for e in episodes:
        for key, st in stats[e["episode_index"]].items():
            require(st["count"] == [e["length"]], f"Stale statistics count: {subset}:{e['episode_index']}:{key}")
    expected_paths = {parquet_path(subset, info, i) for i in ids}
    require(set((subset / "data").glob("chunk-*/episode_*.parquet")) == expected_paths, f"Orphan/missing Parquet files: {subset}")
    offset, tasks_to_run = 0, []
    for e in episodes:
        tasks_to_run.append((str(subset), info, e, offset, stats[e["episode_index"]], droid, task_ids, provenance.get(e["episode_index"])))
        offset += e["length"]
    results, failures = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(validate_episode, task):task[2]["episode_index"] for task in tasks_to_run}
        for n, future in enumerate(tracked(as_completed(futures),f'完整解码校验：{subset.name} episode',len(futures)), 1):
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append(dict(episode_index=futures[future], error=str(exc)))
            if n % 25 == 0 or n == len(episodes):
                print(f"Full decode check {subset.name}: {n}/{len(episodes)} episodes; failures={len(failures)}", flush=True)
    work.mkdir(parents=True, exist_ok=True)
    write_json(work / (subset.name + "_failures.json"), sorted(failures, key=lambda r:r["episode_index"]))
    require(not failures, f"{len(failures)} episode checks failed in {subset}; see {work / (subset.name + '_failures.json')}")
    results.sort(key=lambda r:r["episode_index"])
    calculated = aggregate([dict(stats={k:v for k,v in r["stats"].items() if k in stats[r['episode_index']]}) for r in results])
    compare_stats(calculated, read_json(subset / "meta/stats.json"), subset)
    summary = dict(subset=str(subset), episodes=len(episodes), frames=frames,
                   pngs=sum(r["pngs"] for r in results), videos=sum(r["videos"] for r in results), full_decode=True)
    write_json(work / (subset.name + "_check.json"), summary)
    return summary
