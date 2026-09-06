# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# Depth projection objective adapted from NVlabs/PointWorld (data branch),
# real/extrinsics_pipeline.py and real/compute_extrinsics_utils.py.
# Changes: LeRobot inputs, DROID initialization, explicit holdout evaluation,
# best-iterate selection, bounded pose updates, and dependency-light URDF FK.
"""PointWorld robot-depth calibration, initialized by official DROID poses."""
from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from .common import check_transform, require, read_json, write_json


POINTWORLD_COMMIT = "3872ec6ee73146aa671192ef79b5dfbedc0246e3"
URDF_RELATIVE = "assets/franka_description/franka_panda_robotiq_2f85_og.urdf"


class CalibrationRejected(RuntimeError):
    """Geometrically unobservable or worse than the official initialization."""


def release_pose(path, row, max_loss=.10):
    """Published matrices transform panda_link0 points INTO OpenCV cameras."""
    p = Path(path) / (row["source_episode_id"] + "_cameras.json")
    if not p.exists():
        return None, "missing"
    d = read_json(p)
    if d.get("uuid") != row["source_episode_id"]:
        return None, "uuid_mismatch"
    if not d.get("optimization_success") or d.get("error_info"):
        return None, "unsuccessful"
    loss = d.get("optimization_summary", {}).get("final_loss", float("inf"))
    if not isinstance(loss, (int, float)) or not math.isfinite(loss) or not 0 <= loss < max_loss:
        return None, "quality_rejected"
    poses = []
    for role in ("external_1", "external_2"):
        serial = str(row["camera_serials"][role])
        m = d.get(serial, {}).get("optimized_extrinsics")
        if m is None:
            return None, "serial_missing"
        try:
            poses.append(np.linalg.inv(check_transform(m, f"{p}:{serial}")))
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            return None, "invalid_matrix"
    return np.stack(poses), "pointworld_release"


def prepare_assets(root):
    from .bootstrap import download
    from .common import sha256
    root = Path(root)
    base = f"https://raw.githubusercontent.com/NVlabs/PointWorld/{POINTWORLD_COMMIT}/"
    def fetch(relative):
        target = root / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            part = target.with_name(target.name + ".part")
            download(base + relative, part)
            part.replace(target)
        return target
    urdf = fetch(URDF_RELATIVE)
    files = [URDF_RELATIVE, "LICENSE"]
    for node in ET.parse(urdf).iter("mesh"):
        rel = (Path(URDF_RELATIVE).parent / node.attrib["filename"]).as_posix()
        require(".." not in Path(rel).parts, "Unsafe mesh path")
        files.append(rel)
    for rel in sorted(set(files)):
        fetch(rel)
    stamp = root / "assets.sha256.json"
    hashes = {rel: sha256(root / rel) for rel in sorted(set(files))}
    if stamp.exists():
        require(read_json(stamp) == hashes, "PointWorld assets changed")
    else:
        write_json(stamp, hashes)
    return urdf


def origin(node):
    t = np.eye(4)
    if node is not None:
        t[:3, 3] = np.fromstring(node.get("xyz", "0 0 0"), sep=" ")
        t[:3, :3] = Rotation.from_euler("xyz", np.fromstring(node.get("rpy", "0 0 0"), sep=" ")).as_matrix()
    return t


class Robot:
    """URDF visual meshes and mimic joints, using PointWorld's exact asset.

    No renderer/OpenGL/CUDA extension/urdfpy/networkx-2.2 dependency. Geometry,
    area-weighted sampling and gripper scaling follow the upstream pipeline.
    """
    def __init__(self, urdf, samples=25000, seed=42):
        import trimesh
        self.urdf = Path(urdf)
        tree = ET.parse(urdf).getroot()
        self.joints = list(tree.findall("joint"))
        roots = {n.attrib["name"] for n in tree.findall("link")} - {n.find("child").attrib["link"] for n in self.joints}
        require(roots == {"panda_link0"}, f"Unexpected robot root: {roots}")
        visuals = []
        for link in tree.findall("link"):
            for v in link.findall("visual"):
                m = v.find("geometry/mesh")
                require(m is not None, "Unsupported non-mesh visual in PointWorld URDF")
                mesh = trimesh.load(self.urdf.parent / m.attrib["filename"], force="mesh", process=False)
                mesh.apply_scale(np.fromstring(m.get("scale", "1 1 1"), sep=" "))
                area = mesh.area * (1e-6 if "hand_camera_part" in m.attrib["filename"] else 1)
                if area > 0:
                    visuals.append((link.attrib["name"], origin(v.find("origin")), mesh, area))
        total = sum(v[3] for v in visuals)
        self.visuals = []
        for i, (name, transform, mesh, area) in enumerate(visuals):
            # Same area allocation and minimum 200 points per mesh as upstream.
            points, _ = trimesh.sample.sample_surface(mesh, max(200, int(samples * area / total)), seed=seed + i)
            self.visuals.append((name, transform, points))

    def transforms(self, joints, gripper):
        cfg = {f"panda_joint{i + 1}": float(v) for i, v in enumerate(joints)}
        # DROID normalized closedness -> Robotiq 2F85 finger_joint radians.
        cfg["finger_joint"] = float(gripper) * .725
        result = {"panda_link0": np.eye(4)}
        pending = list(self.joints)
        while pending:
            progressed = []
            for joint in pending:
                parent = joint.find("parent").attrib["link"]
                if parent not in result:
                    continue
                q = cfg.get(joint.attrib["name"], 0.)
                mimic = joint.find("mimic")
                if mimic is not None:
                    q = cfg[mimic.attrib["joint"]] * float(mimic.get("multiplier", 1)) + float(mimic.get("offset", 0))
                motion = np.eye(4)
                kind = joint.attrib["type"]
                if kind in ("revolute", "continuous", "prismatic"):
                    axis = np.fromstring(joint.find("axis").get("xyz", "1 0 0"), sep=" ")
                    axis /= np.linalg.norm(axis)
                    if kind == "prismatic":
                        motion[:3, 3] = axis * q
                    else:
                        motion[:3, :3] = Rotation.from_rotvec(axis * q).as_matrix()
                else:
                    require(kind == "fixed", f"Unsupported joint type: {kind}")
                result[joint.find("child").attrib["link"]] = result[parent] @ origin(joint.find("origin")) @ motion
                progressed.append(joint)
            require(progressed, "Invalid/cyclic URDF joint graph")
            pending = [j for j in pending if j not in progressed]
        return result

    def points(self, joints, gripper):
        transforms = self.transforms(joints, gripper)
        out = []
        for name, visual, points in self.visuals:
            t = transforms[name] @ visual
            out.append(points @ t[:3, :3].T + t[:3, 3])
        return np.concatenate(out).astype(np.float32)


def pose_matrix(p):
    import torch
    x, y, z, roll, pitch, yaw = p.unbind()
    cr, sr, cp, sp, cy, sy = torch.cos(roll), torch.sin(roll), torch.cos(pitch), torch.sin(pitch), torch.cos(yaw), torch.sin(yaw)
    zero, one = x * 0, x * 0 + 1
    return torch.stack([cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr, x,
                        sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr, y,
                        -sp, cp*sr, cp*cr, z, zero, zero, zero, one]).reshape(4, 4)


def depth_loss(points, world_to_camera, depth, intrinsic, dedup=.5):
    """Upstream L1 projected mesh-depth objective with bilinear depth sampling."""
    import torch
    import torch.nn.functional as F
    xyz = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    xyz = xyz[xyz[:, 2] > 0]
    uvw = xyz @ intrinsic.T
    uv = uvw[:, :2] / (uvw[:, 2:3] + 1e-8)
    h, w = depth.shape
    valid = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    uv, z = uv[valid], xyz[valid, 2]
    if dedup > 0 and len(uv):
        q = torch.round(uv / dedup).long()
        _, inv = torch.unique(q[:, 0] * 131071 + q[:, 1], return_inverse=True)
        first = torch.full((int(inv.max()) + 1,), len(uv), device=uv.device, dtype=torch.long)
        first.scatter_reduce_(0, inv, torch.arange(len(uv), device=uv.device), reduce="amin")
        uv, z = uv[first], z[first]
    if not len(uv):
        return world_to_camera.sum() * 0, 0
    grid = torch.stack((2 * uv[:, 0] / (w - 1) - 1, 2 * uv[:, 1] / (h - 1) - 1), -1)
    observed = F.grid_sample(depth[None, None], grid[None, None], align_corners=True).reshape(-1)
    valid = (observed >= .3) & (observed <= 2.)
    count = int(valid.sum())
    return (torch.abs(observed[valid] - z[valid]).mean() if count else world_to_camera.sum() * 0), count


def refine_camera(initial_c2b, k, depths, points, device="cuda", iterations=2000, min_points=1000):
    """Fit on even sampled frames; accept only with held-out non-regression.

    Failing calibration is reported, never mislabeled as a successful refine.
    """
    import torch
    torch.set_num_threads(2)
    initial = torch.tensor(np.linalg.inv(initial_c2b), dtype=torch.float32, device=device)
    kk = torch.tensor(k, dtype=torch.float32, device=device)
    dd = [torch.tensor(d, dtype=torch.float32, device=device) for d in depths]
    pp = [torch.tensor(p, dtype=torch.float32, device=device) for p in points]
    with torch.no_grad():
        visible = [i for i in range(len(dd)) if depth_loss(pp[i], initial, dd[i], kk)[1] >= min_points]
    if len(visible) < 4:
        raise CalibrationRejected(f"Only {len(visible)} usable frames; need 4 with >= {min_points} robot points")
    train, holdout = visible[::2], visible[1::2]
    def evaluate(matrix, indices):
        terms = [depth_loss(pp[i], matrix, dd[i], kk) for i in indices]
        if any(n < min_points for _, n in terms):
            return None
        return torch.stack([loss for loss, _ in terms]).mean()
    with torch.no_grad():
        initial_train = float(evaluate(initial, train))
        initial_test = float(evaluate(initial, holdout))
    param = torch.zeros(6, requires_grad=True, device=device)
    scale = torch.tensor([.01] * 3 + [np.deg2rad(.05)] * 3, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([param], lr=.05, eps=1e-6)
    best, best_loss = initial.detach().clone(), initial_train
    for _ in range(iterations):
        optimizer.zero_grad()
        matrix = pose_matrix(param * scale) @ initial
        loss = evaluate(matrix, train)
        if loss is None or not torch.isfinite(loss):
            break
        if float(loss.detach()) < best_loss:
            best_loss, best = float(loss.detach()), matrix.detach().clone()
        loss.backward()
        optimizer.step()
        # Respect a high-quality initialization: 20 cm/20 degrees per axis.
        with torch.no_grad():
            param[:3].clamp_(-20, 20)
            param[3:].clamp_(-400, 400)
    with torch.no_grad():
        final_test_t = evaluate(best, holdout)
        final_test = float(final_test_t) if final_test_t is not None else float("inf")
    result = np.linalg.inv(best.cpu().numpy())
    shift = float(np.linalg.norm(result[:3, 3] - initial_c2b[:3, 3]))
    angle = float(np.rad2deg(Rotation.from_matrix(result[:3, :3] @ initial_c2b[:3, :3].T).magnitude()))
    accepted = bool(np.isfinite(final_test) and final_test < .1 and
                    final_test <= initial_test + 1e-4 and best_loss <= initial_train and shift <= .30 and angle <= 25)
    metrics = dict(initial_train_loss=initial_train, final_train_loss=best_loss,
                   initial_holdout_loss=initial_test, final_holdout_loss=final_test if np.isfinite(final_test) else None,
                   translation_change_m=shift, rotation_change_deg=angle, train_frames=train,
                   holdout_frames=holdout, accepted=accepted, iterations=iterations)
    if not accepted:
        raise CalibrationRejected(f"Calibration quality gate failed: {metrics}")
    return check_transform(result, "refined camera"), metrics
