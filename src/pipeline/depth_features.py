"""Depth-derived features: edges, surface normals, gravity-aligned height.

Normals computed from in-painted depth via central-difference gradients and
pinhole back-projection. NYUv2 default intrinsics (Kinect v1):
  fx=518.857901, fy=519.469611, cx=325.582449, cy=253.736166
(color-aligned intrinsics from NYU toolbox camera_params.m)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NYU_FX = 518.857901
NYU_FY = 519.469611
NYU_CX = 325.582449
NYU_CY = 253.736166


@dataclass
class DepthFeatures:
    depth: np.ndarray          # (H,W) float32 meters
    valid: np.ndarray          # (H,W) bool
    edges: np.ndarray          # (H,W) float32, |grad depth|
    points: np.ndarray         # (H,W,3) float32, camera coords (X,Y,Z)
    normals: np.ndarray        # (H,W,3) float32, unit vectors
    up_proj: np.ndarray        # (H,W) float32, dot(normal, gravity_up)


def backproject(depth: np.ndarray,
                fx: float = NYU_FX, fy: float = NYU_FY,
                cx: float = NYU_CX, cy: float = NYU_CY) -> np.ndarray:
    H, W = depth.shape
    u = np.arange(W, dtype=np.float32)[None, :].repeat(H, 0)
    v = np.arange(H, dtype=np.float32)[:, None].repeat(W, 1)
    X = (u - cx) / fx * depth
    Y = (v - cy) / fy * depth
    Z = depth
    return np.stack([X, Y, Z], axis=-1)


def compute_normals(points: np.ndarray) -> np.ndarray:
    dzdx = np.zeros_like(points)
    dzdy = np.zeros_like(points)
    dzdx[:, 1:-1, :] = points[:, 2:, :] - points[:, :-2, :]
    dzdy[1:-1, :, :] = points[2:, :, :] - points[:-2, :, :]
    n = np.cross(dzdx, dzdy)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.maximum(norm, 1e-6)
    return n.astype(np.float32)


def sobel_edges(depth: np.ndarray) -> np.ndarray:
    import cv2
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def compute_features(depth: np.ndarray, valid: np.ndarray | None = None) -> DepthFeatures:
    d = depth.astype(np.float32)
    if valid is None:
        valid = np.isfinite(d) & (d > 1e-3)
    d_safe = np.where(valid, d, 0.0)
    pts = backproject(d_safe).astype(np.float32)
    n = compute_normals(pts)
    edges = sobel_edges(d_safe)
    # Gravity: NYUv2 camera is roughly level; camera +Y points down in image space,
    # so world-up ≈ -Y_cam. Planar-floor detection: normal ≈ +Y (cross-product sign
    # depends on our gradient order). Use |n . up| as alignment strength.
    up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    up_proj = (n * up).sum(-1)
    return DepthFeatures(
        depth=d_safe, valid=valid.astype(bool),
        edges=edges, points=pts, normals=n, up_proj=up_proj,
    )


def fit_dominant_planes(
    feat: DepthFeatures,
    dist_thr: float = 0.04,
    min_inliers: int = 4000,
    num_iters: int = 200,
    max_planes: int = 3,
    rng_seed: int = 0,
) -> list[dict]:
    """RANSAC dominant planes on backprojected points (valid-gated).

    Returns list of {mask: (H,W) bool, normal: (3,) unit, d: float, role: str}.
    Role classification uses gravity dot-product: floor if n·up > +0.7,
    ceiling if n·up < -0.7 with mean y in upper image half, else wall.
    """
    H, W = feat.depth.shape
    pts = feat.points.reshape(-1, 3)
    valid = feat.valid.reshape(-1)
    idx = np.flatnonzero(valid & (pts[:, 2] > 1e-3))
    if idx.size < min_inliers:
        return []
    rng = np.random.default_rng(rng_seed)
    remaining = idx.copy()
    planes: list[dict] = []
    up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    ys_img = np.arange(H, dtype=np.float32)[:, None].repeat(W, 1).reshape(-1)

    for _ in range(max_planes):
        if remaining.size < min_inliers:
            break
        best_inl: np.ndarray | None = None
        best_n = None
        for _it in range(num_iters):
            pick = rng.choice(remaining.size, 3, replace=False)
            p = pts[remaining[pick]]
            v1 = p[1] - p[0]
            v2 = p[2] - p[0]
            n = np.cross(v1, v2)
            nn = np.linalg.norm(n)
            if nn < 1e-6:
                continue
            n = n / nn
            d = -float(n @ p[0])
            dists = np.abs(pts[remaining] @ n + d)
            inl = dists < dist_thr
            if best_inl is None or inl.sum() > best_inl.sum():
                best_inl = inl
                best_n = n
        if best_inl is None or best_inl.sum() < min_inliers:
            break
        inlier_idx = remaining[best_inl]
        P = pts[inlier_idx]
        centroid = P.mean(0)
        _, _, Vt = np.linalg.svd(P - centroid, full_matrices=False)
        n_refined = Vt[-1].astype(np.float32)
        if n_refined @ best_n < 0:
            n_refined = -n_refined
        d_refined = float(-n_refined @ centroid)

        mask_flat = np.zeros(pts.shape[0], dtype=bool)
        mask_flat[inlier_idx] = True
        mask = mask_flat.reshape(H, W)

        up_dot = float(n_refined @ up)
        mean_y = float(ys_img[inlier_idx].mean())
        if up_dot > 0.7 and mean_y > 0.4 * H:
            role = "floor"
        elif up_dot < -0.7 and mean_y < 0.35 * H:
            role = "ceiling"
        elif abs(up_dot) < 0.3:
            role = "wall"
        else:
            role = "other"
        planes.append(dict(mask=mask, normal=n_refined, d=d_refined, role=role))
        remaining = remaining[~best_inl]

    return planes


def mask_depth_stats(feat: DepthFeatures, mask: np.ndarray) -> dict[str, float]:
    """Per-mask depth summary."""
    m = mask.astype(bool)
    if not m.any():
        return dict(area=0, mean_depth=0.0, depth_std=0.0, edge_mean=0.0,
                    up_mean=0.0, valid_frac=0.0, bottom_frac=0.0, top_frac=0.0)
    area = int(m.sum())
    d = feat.depth[m]
    up = feat.up_proj[m]
    v = feat.valid[m]
    e = feat.edges[m]
    H = feat.depth.shape[0]
    ys = np.where(m)[0]
    bottom = float((ys > 0.66 * H).mean())
    top = float((ys < 0.34 * H).mean())
    return dict(
        area=area,
        mean_depth=float(d.mean()),
        depth_std=float(d.std()),
        edge_mean=float(e.mean()),
        up_mean=float(up.mean()),
        valid_frac=float(v.mean()),
        bottom_frac=bottom,
        top_frac=top,
    )
