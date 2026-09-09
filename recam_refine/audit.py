"""Read-only camera comparison on ReCam RGB/FS depth, outside fitting frames."""
from __future__ import annotations

from .progress import phase
import html
import hashlib
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

from .common import (require, read_json, read_jsonl, write_json, values, sha256,
                     check_transform, media_path, parquet_path, array_hash, acquire_directory_lock, validate_lock_mount)
from .geometry_metrics import (robot_mask, scene_cloud, cloud_matches, aggregate_matches,
                               aggregate_depth, contour_overlay, WORKSPACE_MIN, WORKSPACE_MAX)
from .inputs import load_depth_records
from .pointworld import (Robot, prepare_assets, release_pose, refine_camera_with_retry, CalibrationRejected,
                        depth_loss, POINTWORLD_COMMIT)


PROTOCOL = dict(
    audit_implementation_version=2,
    reference="https://arxiv.org/html/2601.03782v1#A2.SS3",
    pointworld_commit=POINTWORLD_COMMIT, image_scale=.5,
    depth_range_robot_m=[.3, 2.], depth_range_scene_m=[0., 4.],
    dedup_pixels=.5, mesh_samples=25000, mesh_seed=42,
    workspace_min_m=WORKSPACE_MIN.tolist(), workspace_max_m=WORKSPACE_MAX.tolist(),
    f1_thresholds_m=[.005, .020], scene_subsampling="none after half-resolution decimation",
    paper_depth_aggregation="sum(loss * robot_points) / sum(robot_points)",
    release_code_depth_aggregation="mean of valid camera-frame mean losses",
    f1_aggregation="sum directional NN hits/counts across frames, then harmonic mean of P/R",
    robot_exclusion="filled URDF triangle silhouettes; also report union-of-candidates mask sensitivity",
    limitations=["Consistency metrics, not pose ground truth.",
                 "Published camera poses may have fitted on these frames; only our fit is held out.",
                 "Paper benchmark masks/evaluation code are not published in the pinned data branch; "
                 "this implements its metric definitions with declared rasterization and workspace.",
                 "The paper uses 2000 visible points/100 iterations/1e-3; released CLI uses "
                 "1000 points/2000 iterations/.05. Our fit follows the released objective plus holdout checks."])


def assess_comparison(summary):
    """Report regressions without inventing a PointWorld absolute F1 cutoff."""
    initial = summary["droid_initial"]
    result = {}
    for name, candidate in summary.items():
        if name == "droid_initial":
            continue
        deltas, unavailable = {}, []
        for group, key in (("depth", "point_weighted_m"), ("two_view", "f1_5mm"), ("two_view", "f1_20mm"),
                           ("fixed_mask_two_view", "f1_5mm"), ("fixed_mask_two_view", "f1_20mm")):
            a, b = initial[group][key], candidate[group][key]
            label = f"{group}.{key}"
            if a is None or b is None:
                unavailable.append(label)
            else:
                deltas[label] = b - a
        regressions = [key for key, delta in deltas.items()
                       if (delta > 1e-6 if key.startswith("depth.") else delta < -1e-6)]
        loss = candidate["depth"]["frame_mean_m"]
        result[name] = dict(deltas_candidate_minus_initial=deltas, regressions=regressions,
                            unavailable=unavailable, depth_under_release_0_10=loss is not None and loss < .10,
                            status="needs_review" if regressions or unavailable or loss is None or loss >= .10 else "sample_metrics_non_regressing",
                            note="This is a paired sample diagnostic, not PointWorld certification or pose ground truth.")
    return result


def evaluation_frames(length, excluded, count):
    excluded = set(excluded)
    available = np.array([i for i in range(length) if i not in excluded], dtype=int)
    require(len(available) >= 4, "Fewer than four evaluation frames outside fitting/selection frames")
    return available[np.unique(np.linspace(0, len(available) - 1, min(count, len(available)), dtype=int))].tolist()


def read_rgb(path, indices):
    requested, result = set(indices), {}
    with av.open(str(path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i in requested:
                result[i] = frame.to_ndarray(format="rgb24")[::2, ::2]
            if i >= max(requested):
                break
    require(result.keys() == requested, f"Missing requested RGB display frames: {path}")
    return result


def pose_change(initial, candidate):
    from scipy.spatial.transform import Rotation
    return dict(translation_m=float(np.linalg.norm(initial[:3, 3] - candidate[:3, 3])),
                rotation_deg=float(np.rad2deg(Rotation.from_matrix(
                    candidate[:3, :3] @ initial[:3, :3].T).magnitude())))


def panel(images, labels, subtitle, path):
    w, h = images[0].size
    canvas = Image.new("RGB", (w * len(images), h + 66), "#111827")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=17)
    small = ImageFont.load_default(size=13)
    for i, (im, label) in enumerate(zip(images, labels)):
        canvas.paste(im, (w * i, 42))
        draw.text((w * i + 12, 12), label, font=font, fill="white")
    draw.text((12, h + 48), subtitle, font=small, fill="#d1d5db")
    canvas.save(path)


def cloud_panel(clouds, path):
    """Metric orthographic X/Y view. Shared bounds, colors, pixels per meter."""
    images, labels = [], []
    for label, pair in clouds.items():
        canvas = np.full((640, 640, 3), 245, dtype=np.uint8)
        layers = []
        for points in pair:
            xy = points[:, :2]
            u = ((xy[:, 0] + .05) / .8 * 639).astype(int)
            v = ((.4 - xy[:, 1]) / .8 * 639).astype(int)
            keep = (u >= 0) & (u < 640) & (v >= 0) & (v < 640)
            mask = np.zeros((640, 640), dtype=bool)
            mask[v[keep], u[keep]] = True
            layers.append(mask)
        canvas[layers[0]] = [237, 75, 85]
        canvas[layers[1]] = [35, 145, 230]
        canvas[layers[0] & layers[1]] = [115, 75, 160]
        im = Image.fromarray(canvas)
        draw = ImageDraw.Draw(im)
        draw.line((24, 610, 104, 610), fill="black", width=3)
        draw.text((24, 618), "10 cm", fill="black", font=ImageFont.load_default(size=13))
        images.append(im)
        labels.append(label)
    panel(images, labels, "Robot removed | base-frame XY | camera 1 red / camera 2 blue | exact F1 uses 3D, not this projection", path)


def fit_candidate(initial, kk, joints, gripper, depths, robot, indices, device, iterations):
    points = [robot.points(joints[t], gripper[t]) for t in indices]
    poses, metrics = [], []
    for cam in range(2):
        try:
            pose, metric = refine_camera_with_retry(initial[cam], kk[cam], [depths[t][cam] for t in indices],
                                                    points, device=device, iterations=iterations)
        except CalibrationRejected as exc:
            pose, metric = initial[cam], dict(accepted=False, reason=str(exc))
        poses.append(pose.tolist())
        metrics.append(metric)
        print(f"AUDIT FIT external_{cam + 1}: {metric}", flush=True)
    return dict(camera_to_base=poses, metrics=metrics, sample_frame_indices=indices,
                source="pointworld_method_droid_initialization")


def audit_episode(root, info, row, job, records, camera_dir, candidate_dir, robot, args, out):
    import torch
    i = row["episode_index"]
    out.mkdir(parents=True, exist_ok=True)
    release_path = Path(camera_dir) / (row["source_episode_id"] + "_cameras.json")
    candidate_path = candidate_dir / f"episode_{i:06d}.json" if candidate_dir else None
    parameters = dict(frames=args.frames, image_frames=getattr(args, "image_frames", 3), fit=args.fit,
                      iterations=args.iterations, dataset=str(root), row=row,
                      plan_hash=hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest(),
                      release=str(release_path), release_exists=release_path.exists(),
                      candidate=str(candidate_path), candidate_exists=candidate_path is not None and candidate_path.exists())
    receipt = out / "metrics.json"
    if receipt.exists():
        cached = read_json(receipt)
        if cached.get("protocol") == PROTOCOL and cached.get("audit_parameters") == parameters:
            tracked = {**cached["input_sha256"], **cached.get("artifact_sha256", {})}
            if all(Path(p).is_file() and sha256(Path(p)) == h for p, h in tracked.items()):
                print(f"AUDIT episode {i}: verified cached report", flush=True)
                return cached
    source_path = parquet_path(root, info, i)
    hashes = {}
    def track(p):
        hashes[str(p)] = sha256(p)
        return p
    table = pq.read_table(track(source_path))
    joints = values(table["observation.states.joint_state"])
    gripper = values(table["observation.states.gripper_state"]).ravel()
    initial = np.asarray(job["initial_extrinsics"])[1:] if job else values(table["observation.camera.extrinsics"])[0, 1:]
    kk = np.asarray(job["initial_intrinsics"])[1:].copy() if job else values(table["observation.camera.intrinsics"])[0, 1:].copy()
    intrinsic_sources = []
    n = min(len(table), job["length"] if job else len(table))
    for cam in (1, 2):
        rec = job["depths"][str(cam)]["record"] if job else records.get((i, cam))
        if rec:
            kk[cam - 1] = rec["intrinsic"]
            n = min(n, rec["decoded"])
        intrinsic_sources.append("FS sidecar" if rec else "ReCam Parquet")
    # Audit original samples too: exclude only at most two terminal zero PNGs.
    original_n = n
    for cam in (1, 2):
        valid_n = original_n
        while valid_n > 0 and original_n - valid_n <= 2:
            p = media_path(root, info, i, f"observation.images.depth_{cam:02d}", valid_n - 1)
            with Image.open(track(p)) as im:
                if np.any(np.asarray(im)):
                    break
            valid_n -= 1
        require(original_n - valid_n <= 2, f"More than two zero tail frames: {i}:{cam}")
        n = min(n, valid_n)
    kk[:, :2] *= .5
    depth_cache = {}
    def load_depth(t):
        if t not in depth_cache:
            dd = []
            for cam in (1, 2):
                p = media_path(root, info, i, f"observation.images.depth_{cam:02d}", t)
                with Image.open(track(p)) as im:
                    dd.append(np.asarray(im, dtype=np.float32)[::2, ::2] / 1000.)
            require(dd[0].shape == dd[1].shape, f"External camera resolutions differ: {i}")
            depth_cache[t] = dd
        return depth_cache[t]
    variants = {"droid_initial": check_transform(initial, "audit initial")}
    if release_path.exists():
        track(release_path)
    released, status = release_pose(camera_dir, row)
    if args.fit:
        fit_indices = np.unique(np.linspace(0, n - 1, min(16, n), dtype=int)).tolist()
        for t in fit_indices:
            load_depth(t)
        identity = dict(initial=array_hash(initial), intrinsic=array_hash(kk),
                        joints=array_hash(joints[fit_indices]), gripper=array_hash(gripper[fit_indices]),
                        depths=[array_hash(np.stack(depth_cache[t])) for t in fit_indices],
                        indices=fit_indices, iterations=args.iterations, protocol_version=2)
        cache = out / "candidate.json"
        if cache.exists():
            candidate = read_json(cache)
            require(candidate.get("audit_fit_identity") == identity,
                    f"Cached fit inputs/settings changed; use a new report directory: {cache}")
        else:
            candidate = fit_candidate(initial, kk, joints, gripper, depth_cache, robot, fit_indices, args.device, args.iterations)
            candidate.update(source_episode_id=row["source_episode_id"], audit_fit_identity=identity)
            write_json(cache, candidate)
    elif candidate_path and candidate_path.exists():
        candidate = read_json(track(candidate_path))
        require(candidate["source_episode_id"] == row["source_episode_id"], "Candidate source UUID mismatch")
    else:
        require(released is not None, f"No candidate or PointWorld release for episode {i}; pass --fit for a read-only trial")
        candidate = None
    excluded = set()
    if candidate:
        variants["recam_candidate"] = check_transform(candidate["camera_to_base"], "audit candidate")
        excluded.update(candidate.get("sample_frame_indices", []))
    if released is not None:
        variants["pointworld_release"] = released
    require(all(p.shape == (2, 4, 4) for p in variants.values()), "Expected two external camera poses")
    test_indices = evaluation_frames(n, excluded, args.frames)
    # Image choices fixed before scores exist, no cherry-picking of improvements.
    selected = [test_indices[j] for j in np.unique(np.linspace(0, len(test_indices) - 1,
                min(getattr(args, "image_frames", 3), len(test_indices)), dtype=int))]
    rgb = [read_rgb(track(media_path(root, info, i, f"observation.images.rgb_{cam:02d}")), selected) for cam in (1, 2)] if selected else None
    results = {name: dict(depth=[], matches=[], fixed_mask_matches=[]) for name in variants}
    frame_rows, pictures = [], []
    for t in test_indices:
        depths = load_depth(t)
        points = torch.tensor(robot.points(joints[t], gripper[t]), device=args.device)
        geometry = robot.geometry(joints[t], gripper[t])
        masks = {name: [robot_mask(*geometry, poses[c], kk[c], depths[c].shape) for c in range(2)]
                 for name, poses in variants.items()}
        union = [np.logical_or.reduce([m[c] for m in masks.values()]) for c in range(2)]
        frame_result, clouds_plot = dict(frame_index=t, variants={}), {}
        for name, poses in variants.items():
            per_camera, clouds, fixed_clouds = [], [], []
            for c in range(2):
                with torch.no_grad():
                    loss, count = depth_loss(points, torch.tensor(np.linalg.inv(poses[c]), dtype=torch.float32, device=args.device),
                                             torch.tensor(depths[c], device=args.device), torch.tensor(kk[c], dtype=torch.float32, device=args.device))
                metric = dict(frame_index=t, camera=c + 1, loss_m=float(loss) if count else None, robot_points=count)
                per_camera.append(metric)
                results[name]["depth"].append(metric)
                clouds.append(scene_cloud(depths[c], kk[c], poses[c], masks[name][c])[0])
                fixed_clouds.append(scene_cloud(depths[c], kk[c], poses[c], union[c])[0])
            matching, fixed_matching = cloud_matches(*clouds), cloud_matches(*fixed_clouds)
            results[name]["matches"].append(matching)
            results[name]["fixed_mask_matches"].append(fixed_matching)
            frame_result["variants"][name] = dict(depth=per_camera, matches=matching, fixed_mask_matches=fixed_matching)
            clouds_plot[name] = clouds
        if t in selected:
            for c in range(2):
                images, labels = [Image.fromarray(rgb[c][t])], ["Observed RGB"]
                for name in variants:
                    images.append(contour_overlay(rgb[c][t], masks[name][c]))
                    loss = frame_result["variants"][name]["depth"][c]["loss_m"]
                    labels.append(f"{name} | {100 * loss:.2f} cm" if loss is not None else f"{name} | unavailable")
                filename = f"frame_{t:06d}_cam{c + 1}.png"
                panel(images, labels, f"Episode {i} | frame {t} | external_{c + 1} | same RGB/K/depth/URDF | green = projected robot", out / filename)
                pictures.append(filename)
            filename = f"frame_{t:06d}_clouds.png"
            cloud_panel(clouds_plot, out / filename)
            pictures.append(filename)
        frame_rows.append(frame_result)
    summary = {}
    for name, r in results.items():
        summary[name] = dict(depth=aggregate_depth(r["depth"]), two_view=aggregate_matches(r["matches"]),
                             fixed_mask_two_view=aggregate_matches(r["fixed_mask_matches"]),
                             pose_change_from_initial=[pose_change(initial[c], variants[name][c]) for c in range(2)],
                             cameras=[aggregate_depth([m for m in r["depth"] if m["camera"] == c]) for c in (1, 2)])
    unchanged = all(sha256(Path(p)) == h for p, h in hashes.items())
    require(unchanged, "An audit input changed while being read")
    report = dict(episode_index=i, source_episode_id=row["source_episode_id"], release_status=status,
                  candidate=candidate, intrinsics_sources=intrinsic_sources, intrinsics_half_resolution=kk.tolist(),
                  poses={name:p.tolist() for name, p in variants.items()}, summary=summary,
                  assessment=assess_comparison(summary),
                  evaluation_frames=test_indices, excluded_fit_or_selection_frames=sorted(excluded),
                  image_frames=selected, images=pictures, per_frame=frame_rows, input_sha256=hashes,
                  inputs_unchanged=unchanged, protocol=PROTOCOL, audit_parameters=parameters,
                  artifact_sha256={str(out / p):sha256(out / p) for p in pictures})
    if args.fit:
        report["artifact_sha256"][str(out / "candidate.json")] = sha256(out / "candidate.json")
    write_json(out / "metrics.json", report)
    print(f"AUDIT episode {i}: " + "; ".join(f"{name}: {r['status']}" for name, r in report["assessment"].items()), flush=True)
    return report


def write_html(output, reports, errors):
    blocks = ["<!doctype html><html lang='en'><meta charset='utf-8'><title>ReCam camera audit</title>",
              "<style>body{font:16px system-ui;max-width:1700px;margin:32px auto;padding:0 24px;color:#182333}"
              "table{border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #ccd}img{width:100%;margin:12px 0}"
              "summary{cursor:pointer;padding:16px;background:#eef3f9}pre{white-space:pre-wrap}</style>",
              "<h1>ReCam camera calibration audit</h1><p>Same RGB, depth, intrinsics, URDF and evaluation frames for each variant. "
              "Lower depth L1 and higher F1 are better. Consistency metrics are not pose ground truth. "
              "Our fit/selection frames are excluded; released PointWorld poses may have seen evaluation frames.</p>",
              "<p>Robot-depth L1 uses point weighting (paper); the JSON also reports frame averaging (released code). "
              "Two-view F1 removes robot pixels, crops to the published workspace, and counts symmetric nearest neighbors "
              "within 5/20 mm. No ICP alignment is applied. Scene point counts and a fixed-mask sensitivity check are saved.</p>"]
    for name in ("comparison.png", "paired_cdf.png"):
        if (output / name).exists():
            blocks.append(f"<img src='{name}' alt='Recorded metric comparison'>")
    for report in reports:
        i = report["episode_index"]
        blocks.append(f"<h2>Episode {i}: {html.escape(report['source_episode_id'])}</h2><table><tr>"
                      "<th>Variant</th><th>Robot depth L1</th><th>F1 @ 5 mm</th><th>F1 @ 20 mm</th><th>Scene points (both views)</th></tr>")
        for name, s in report["summary"].items():
            depth, match = s["depth"]["point_weighted_m"], s["two_view"]
            f5, f20 = match["f1_5mm"], match["f1_20mm"]
            cells = [name, f"{depth * 100:.3f} cm" if depth is not None else "Unavailable",
                     f"{100 * f5:.2f}%" if f5 is not None else "Unavailable",
                     f"{100 * f20:.2f}%" if f20 is not None else "Unavailable",
                     str(match["points_a"] + match["points_b"])]
            blocks.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in cells) + "</tr>")
        blocks.append(f"</table><p>Frames: {report['evaluation_frames']}. <a href='episode_{i:06d}/metrics.json'>Full measurements and hashes</a></p>")
        for name, assessment in report.get("assessment", {}).items():
            blocks.append(f"<p>{html.escape(name)}: <b>{assessment['status']}</b>; "
                          f"regressing metrics: {html.escape(', '.join(assessment['regressions']) or 'none')}.</p>")
        blocks.append("<details open><summary>RGB robot overlays and two-camera point clouds</summary>")
        blocks.extend(f"<a href='episode_{i:06d}/{p}'><img loading='lazy' src='episode_{i:06d}/{p}'></a>" for p in report["images"])
        blocks.append("</details>")
    if errors:
        blocks.append("<h2>Unavailable / failed episodes</h2><pre>" + html.escape(json.dumps(errors, indent=2)) + "</pre>")
    blocks.append("<details><summary>Protocol and limitations</summary><pre>" + html.escape(json.dumps(PROTOCOL, indent=2)) + "</pre></details></html>")
    (output / "index.html").write_text("\n".join(blocks), encoding="utf-8")


def _init_audit_worker(urdf):
    import torch
    torch.set_num_threads(2)
    global _audit_robot
    _audit_robot = Robot(urdf, keep_meshes=True)


def _audit_worker(task):
    root, info, row, job, records, camera_dir, candidate_dir, args, out = task
    return audit_episode(root, info, row, job, records, camera_dir, candidate_dir, _audit_robot, args, out)


def run_audit(args):
    """Hold shared input locks and an exclusive report lock during the audit."""
    import fcntl
    from contextlib import ExitStack
    root, work, out = args.root.resolve(), args.work_dir.resolve(), args.report_dir.resolve()
    require(root.is_dir(), f"Missing dataset: {root}")
    require(not work.is_relative_to(root) and not out.is_relative_to(root), "Audit work/report must be outside the dataset")
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    validate_lock_mount(work)
    validate_lock_mount(out)
    with ExitStack() as stack:
        run_lock = stack.enter_context((work / "run.lock").open("a+"))
        report_lock = stack.enter_context((out / "audit.lock").open("a+"))
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        stack.callback(os.close, fd)
        try:
            fcntl.flock(run_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            acquire_directory_lock(fd, fcntl.LOCK_SH)
            fcntl.flock(report_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Dataset refinement or another writer is using this audit report") from exc
        return _run_audit_locked(args)


def _run_audit_locked(args):
    root, work, out = args.root.resolve(), args.work_dir.resolve(), args.report_dir.resolve()
    require(not work.is_relative_to(root) and not out.is_relative_to(root), "Audit work/report must be outside the dataset")
    if (root / "real_world/droid").is_dir():
        root = root / "real_world/droid"
    require(not work.is_relative_to(root) and not out.is_relative_to(root), "Audit work/report must be outside the dataset")
    require(args.frames >= 4 and args.iterations > 0, "Need >=4 test frames and positive iterations")
    require(getattr(args, "image_frames", 3) >= 0 and getattr(args, "workers", 1) > 0, "image-frames must be nonnegative and workers positive")
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".matplotlib"))
    info = read_json(root / "meta/info.json")
    plan = {r["episode_index"]:r for r in read_json(work / "plan.json")} if (work / "plan.json").exists() else {}
    require(plan or not (root / "meta/refinement.jsonl").exists(),
            "This dataset is already refined; supply its original work directory to recover the true initial poses")
    manifest_path = args.episode_manifest or work / "inputs/episode_manifest.jsonl"
    if plan and not args.episode_manifest:
        rows = {i:job['source'] for i,job in plan.items()}
    else:
        available = {int(r['episode_index']):r for r in read_jsonl(manifest_path)}
        catalogue = [e for e in read_jsonl(root/'meta/episodes.jsonl')
                     if int(e.get('source_episode_index',e['episode_index'])) in available]
        rows = {e['episode_index']:dict(available[int(e.get('source_episode_index',e['episode_index']))],
                    episode_index=e['episode_index'],source_episode_index=int(e.get('source_episode_index',e['episode_index'])))
                for e in catalogue}
    ids = sorted(rows) if args.episodes == ["all"] else sorted(set(map(int, args.episodes)))
    require(set(ids) <= rows.keys(), "An episode is missing from the source manifest")
    for i in ids:
        if "camera_serials" not in rows[i]:
            rows[i]["camera_serials"] = {role:str(v["serial"]) for role, v in rows[i]["cameras"].items()}
    records = load_depth_records(args.depth_metadata, rows)
    camera_dir = args.pointworld_cameras or work / "inputs/pointworld_cameras"
    candidate_dir = args.candidate_dir or work / "cameras"
    urdf = prepare_assets(work / "pointworld")
    reports, errors = [], []
    out.mkdir(parents=True, exist_ok=True)
    (out / "COMPLETE.json").unlink(missing_ok=True)
    (out / "QUALITY_REVIEW_REQUIRED.json").unlink(missing_ok=True)
    for name in ("comparison.png", "comparison.pdf", "paired_cdf.png", "paired_cdf.pdf"):
        (out / name).unlink(missing_ok=True)
    grouped = {}
    for key, record_value in records.items():
        grouped.setdefault(key[0], {})[key] = record_value
    tasks = [(root, info, rows[i], plan.get(i), grouped.get(i, {}),
              camera_dir, candidate_dir, args, out / f"episode_{i:06d}") for i in ids]
    phase('几何校验：episode',0,len(tasks))
    def record(i, report=None, error=None):
        if error is not None:
            errors.append(dict(episode_index=i, error=str(error)))
            print(f"AUDIT ERROR episode {i}: {error}", flush=True)
        else:
            reports.append({key:report[key] for key in ("episode_index", "source_episode_id", "summary", "assessment",
                                                       "evaluation_frames", "images")})
        phase('几何校验：episode',len(reports)+len(errors),len(tasks),f'errors={len(errors)}')
        if (len(reports) + len(errors)) % 100 == 0:
            write_html(out, sorted(reports, key=lambda r:r["episode_index"]), errors)
    workers = getattr(args, "workers", 1)
    if workers == 1:
        _init_audit_worker(urdf)
        for task in tasks:
            try:
                record(task[2]["episode_index"], _audit_worker(task))
            except Exception as exc:
                record(task[2]["episode_index"], error=exc)
    else:
        from multiprocessing import get_context
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"),
                                 initializer=_init_audit_worker, initargs=(urdf,)) as pool:
            futures = {pool.submit(_audit_worker, task):task[2]["episode_index"] for task in tasks}
            for future in as_completed(futures):
                try:
                    record(futures[future], future.result())
                except Exception as exc:
                    record(futures[future], error=exc)
    reports.sort(key=lambda r:r["episode_index"])
    review = [dict(episode_index=r["episode_index"], variant=name, **assessment)
              for r in reports for name, assessment in r["assessment"].items() if assessment["status"] == "needs_review"]
    write_json(out / "summary.json", dict(episodes=[dict(episode_index=r["episode_index"], summary=r["summary"], assessment=r["assessment"]) for r in reports],
                                          errors=errors, quality_review=review, protocol=PROTOCOL))
    from .audit_plots import comparison_plots
    comparison_plots(out, reports)
    write_html(out, reports, errors)
    require(not errors, f"Audit incomplete for {len(errors)} episodes; see {out / 'summary.json'}")
    write_json(out / "COMPLETE.json", dict(episodes=len(reports), read_only=True, note="Report generation complete; NOT a calibration quality certificate"))
    if review:
        write_json(out / "QUALITY_REVIEW_REQUIRED.json", review)
    print(f"AUDIT REPORT {out / 'index.html'}", flush=True)
    return 2 if review and getattr(args, "fail_on_review", False) else 0
