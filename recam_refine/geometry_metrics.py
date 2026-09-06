"""PointWorld paper A.2.3 metrics, with all implementation choices explicit.

These measure depth/geometry consistency, NOT pose ground-truth accuracy.
No OpenGL, Open3D, renderer service, learned model, or extra dependency.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

# NVlabs/PointWorld, real/workspace.py at POINTWORLD_COMMIT.
WORKSPACE_MIN = np.array([0., -.4, -.3])
WORKSPACE_MAX = np.array([.7, .4, 1.2])
THRESHOLDS_M = (.005, .020)


def robot_mask(vertices, faces, camera_to_base, intrinsic, shape):
    """Union of projected URDF triangles, at integer-centered depth pixels.

    Exact triangle silhouettes (not a sparse point splat). Hidden robot
    surfaces need no z-buffer for the silhouette union. No scene occlusion
    inference: foreground scene occluders are also excluded inside the mask.
    Clip near-plane crossings before projection (the base can extend behind
    an external camera even when the arm is clearly visible).
    """
    w2c = np.linalg.inv(camera_to_base)
    xyz = vertices @ w2c[:3, :3].T + w2c[:3, 3]
    tri_z = xyz[faces, 2]
    near = .001
    front = xyz[faces[tri_z.min(1) >= near]]
    clipped = []
    for triangle in xyz[faces[(tri_z.min(1) < near) & (tri_z.max(1) >= near)]]:
        polygon = []
        for previous, current in zip(np.roll(triangle, 1, axis=0), triangle):
            a, b = previous[2] >= near, current[2] >= near
            if a != b:
                polygon.append(previous + (current - previous) * ((near - previous[2]) / (current[2] - previous[2])))
            if b:
                polygon.append(current)
        for j in range(1, len(polygon) - 1):
            clipped.append([polygon[0], polygon[j], polygon[j + 1]])
    if clipped:
        front = np.concatenate([front, np.array(clipped)])
    uvw = front @ intrinsic.T
    triangles = uvw[:, :, :2] / uvw[:, :, 2:3]
    h, w = shape
    lo, hi = triangles.min(1), triangles.max(1)
    triangles = triangles[(hi[:, 0] >= 0) & (lo[:, 0] < w) & (hi[:, 1] >= 0) & (lo[:, 1] < h)]
    result = Image.new("1", (w, h))
    draw = ImageDraw.Draw(result)
    for triangle in triangles:
        draw.polygon([tuple(p) for p in triangle], fill=1)
    return np.asarray(result, dtype=bool)


def scene_cloud(depth, intrinsic, camera_to_base, mask, bounds=(WORKSPACE_MIN, WORKSPACE_MAX)):
    """All half-resolution valid scene pixels, no random/voxel subsampling."""
    valid = np.isfinite(depth) & (depth > 0) & (depth <= 4.) & ~mask
    y, x = np.nonzero(valid)
    rays = np.column_stack([x, y, np.ones(len(x))]) @ np.linalg.inv(intrinsic).T
    xyz = rays * depth[y, x, None]
    world = xyz @ camera_to_base[:3, :3].T + camera_to_base[:3, 3]
    valid_workspace = ((world >= bounds[0]) & (world <= bounds[1])).all(1)
    return world[valid_workspace], np.column_stack([x, y])[valid_workspace]


def cloud_matches(a, b):
    """Symmetric NN hit counts; aggregate counts BEFORE P/R/F1 across time."""
    counts = dict(points_a=len(a), points_b=len(b), hits_a=[0, 0], hits_b=[0, 0])
    if len(a) and len(b):
        da = cKDTree(b).query(a, workers=2)[0]
        db = cKDTree(a).query(b, workers=2)[0]
        counts["hits_a"] = [int((da <= t).sum()) for t in THRESHOLDS_M]
        counts["hits_b"] = [int((db <= t).sum()) for t in THRESHOLDS_M]
    return counts


def aggregate_matches(rows):
    na, nb = sum(r["points_a"] for r in rows), sum(r["points_b"] for r in rows)
    if not na or not nb:
        return dict(points_a=na, points_b=nb, available=False, f1_5mm=None, f1_20mm=None)
    result = dict(points_a=na, points_b=nb, available=True)
    for i, label in enumerate(("5mm", "20mm")):
        precision = sum(r["hits_a"][i] for r in rows) / na
        recall = sum(r["hits_b"][i] for r in rows) / nb
        result.update({f"precision_{label}": precision, f"recall_{label}": recall,
                       f"f1_{label}": 2 * precision * recall / (precision + recall) if precision + recall else 0.})
    return result


def aggregate_depth(rows):
    valid = [r for r in rows if r["robot_points"] > 0 and r["loss_m"] is not None]
    count = sum(r["robot_points"] for r in valid)
    return dict(valid_camera_frames=len(valid), robot_points=count,
                frame_mean_m=float(np.mean([r["loss_m"] for r in valid])) if count else None,
                point_weighted_m=sum(r["loss_m"] * r["robot_points"] for r in valid) / count if count else None)


def contour_overlay(rgb, mask, color=(40, 240, 150)):
    """Shared colors/opacity across candidates; contours come from the URDF."""
    rgb = np.asarray(rgb).copy()
    rgb[mask] = (.85 * rgb[mask] + .15 * np.array(color)).astype(np.uint8)
    edge = mask & ~binary_erosion(mask, iterations=1)
    rgb[edge] = color
    return Image.fromarray(rgb)
