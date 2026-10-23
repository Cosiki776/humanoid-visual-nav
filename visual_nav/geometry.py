"""RGB-D geometry. Habitat camera axes: +X right, +Y up, -Z forward."""
import numpy as np


def unproject(depth, mask, intrinsics, camera_to_world, max_depth=8.0):
    depth = np.asarray(depth)
    mask = np.asarray(mask, dtype=bool)
    if depth.shape != mask.shape or depth.ndim != 2:
        raise ValueError("mask/depth must be aligned H x W arrays")
    valid = mask & np.isfinite(depth) & (depth > 0.15) & (depth < max_depth)
    v, u = np.nonzero(valid)
    if len(u) < 25:
        return None
    z = depth[v, u]
    # Reject isolated background contamination without assuming an object size.
    lo, hi = np.quantile(z, [0.05, 0.95])
    keep = (z >= lo) & (z <= hi)
    u, v, z = u[keep], v[keep], z[keep]
    k = np.asarray(intrinsics)
    xyz = np.column_stack(((u-k[0, 2])*z/k[0, 0], -(v-k[1, 2])*z/k[1, 1], -z))
    pose = np.asarray(camera_to_world)
    points = xyz @ pose[:3, :3].T + pose[:3, 3]
    return points[::max(1, len(points)//1500)]


def horizontal_distance(a, b):
    return float(np.linalg.norm((np.asarray(a)-np.asarray(b))[[0, 2]]))


def path_length(points):
    p = np.asarray(points)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0
