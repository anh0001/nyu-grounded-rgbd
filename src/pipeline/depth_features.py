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
