"""Rasterize Candidate list into dense semantic map.

Three phases:
  A. place instance-like classes by descending score
  B. fill structural classes (floor/wall/ceiling) using depth/gravity priors
  C. residual fill via class-prior argmax over remaining stuff candidates
"""
from __future__ import annotations

import numpy as np

from src.pipeline.depth_features import DepthFeatures
from src.pipeline.mask_refine import Candidate

# NYU40 class id groupings
STRUCTURAL = {1, 2, 22}                      # wall, floor, ceiling
OPENINGS = {9, 13, 16, 19, 28}               # window/blinds/curtain/mirror/shower curtain
INSTANCE_LIKE = set(range(1, 41)) - STRUCTURAL - {38}  # everything except wall/floor/ceiling/otherstructure
FALLBACK_STUFF = 38   # otherstructure
FALLBACK_PROP = 40    # otherprop
FALLBACK_FURN = 39    # otherfurniture


def rasterize(
    cands: list[Candidate],
    feat: DepthFeatures,
    num_classes: int = 40,
    fill_structural_by_geometry: bool = True,
) -> np.ndarray:
    H, W = feat.depth.shape
    sem = np.zeros((H, W), dtype=np.uint8)      # 0 = unassigned
    scoremap = np.full((H, W), -np.inf, dtype=np.float32)

    # Phase A: instance-like candidates, high score first.
    instance_cands = [c for c in cands if c.class_id in INSTANCE_LIKE or c.class_id in OPENINGS]
    instance_cands.sort(key=lambda c: c.score, reverse=True)
    for c in instance_cands:
        m = c.mask & (c.score > scoremap)
        sem[m] = c.class_id
        scoremap[m] = c.score

    # Phase B: structural fill (wall/floor/ceiling).
    struct_cands = [c for c in cands if c.class_id in STRUCTURAL]
    if fill_structural_by_geometry:
        # geometry-derived masks as extra candidates
        geo_masks = _geometry_structural_masks(feat)
        for cid, m, s in geo_masks:
            existing = sem != 0
            mm = m & (~existing)
            sem[mm] = cid
            scoremap[mm] = np.maximum(scoremap[mm], s)
    for c in struct_cands:
        existing = sem != 0
        m = c.mask & (~existing)
        sem[m] = c.class_id
        scoremap[m] = np.maximum(scoremap[m], c.score)

    # Phase C: residual fill — for every unassigned pixel, use best candidate
    # that covered that pixel at all; else fallback by geometry.
    unassigned = sem == 0
    if unassigned.any():
        for c in sorted(cands, key=lambda x: x.score, reverse=True):
            m = c.mask & unassigned
            if not m.any():
                continue
            sem[m] = c.class_id
            unassigned &= ~m
            if not unassigned.any():
                break

    # Final fallback for anything still 0: use geometric priors.
    if (sem == 0).any():
        sem = _fallback_fill(sem, feat)

    if num_classes == 13:
        sem = _map_40_to_13(sem)
    return sem


def _geometry_structural_masks(feat: DepthFeatures) -> list[tuple[int, np.ndarray, float]]:
    """Return list of (class_id, mask, confidence) from geometric priors."""
    out: list[tuple[int, np.ndarray, float]] = []
    H, _ = feat.depth.shape
    up = feat.up_proj
    v = feat.valid
    ys = np.arange(H)[:, None].repeat(feat.depth.shape[1], 1)

    # Floor: up_proj large positive, lower half.
    floor = v & (up > 0.8) & (ys > 0.4 * H)
    if floor.sum() > 2000:
        out.append((2, floor, 0.6))
    # Ceiling: up_proj large negative, upper region.
    ceiling = v & (up < -0.8) & (ys < 0.35 * H)
    if ceiling.sum() > 1500:
        out.append((22, ceiling, 0.55))
    # Wall: near-horizontal normal, not yet floor/ceiling.
    wall = v & (np.abs(up) < 0.3) & (~floor) & (~ceiling)
    if wall.sum() > 5000:
        out.append((1, wall, 0.45))
    return out


def _fallback_fill(sem: np.ndarray, feat: DepthFeatures) -> np.ndarray:
    """Fill remaining 0 pixels by geometric heuristic."""
    H, W = sem.shape
    ys = np.arange(H)[:, None].repeat(W, 1)
    unl = sem == 0
    if not unl.any():
        return sem
    up = feat.up_proj
    floor_like = unl & (up > 0.5) & (ys > 0.5 * H)
    ceil_like = unl & (up < -0.5) & (ys < 0.4 * H)
    wall_like = unl & (~floor_like) & (~ceil_like) & (np.abs(up) < 0.5)
    sem[floor_like] = 2
    sem[ceil_like] = 22
    sem[wall_like] = 1
    sem[sem == 0] = FALLBACK_PROP
    return sem


def _map_40_to_13(sem40: np.ndarray) -> np.ndarray:
    from src.datasets.nyuv2_meta import nyu40_to_nyu13_lut
    lut = np.array(nyu40_to_nyu13_lut(), dtype=np.uint8)
    return lut[sem40]
