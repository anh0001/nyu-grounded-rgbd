"""Rasterize Candidate list into dense semantic map.

Three phases:
  A. place instance-like classes by descending score
  B. fill structural classes (floor/wall/ceiling) using depth/gravity priors
  C. residual fill via class-prior argmax over remaining stuff candidates
"""
from __future__ import annotations

import numpy as np

from src.pipeline.depth_features import DepthFeatures, fit_dominant_planes
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
    cc_min_area: int = 0,
    use_ransac_planes: bool = False,
    dense_logits: np.ndarray | None = None,
    dense_class_ids: list[int] | None = None,
    dense_min_logit: float = 0.0,
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
        if use_ransac_planes:
            geo_masks = _ransac_structural_masks(feat)
            if not geo_masks:
                geo_masks = _geometry_structural_masks(feat)
        else:
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

    # Dense-CLIP replacement: pixels assigned to FALLBACK_PROP (i.e. no
    # candidate, no geometric prior) get a second look from dense CLIP/SigLIP
    # logits, restricted to tail classes most often missed by GDino+SAM.
    if dense_logits is not None and dense_class_ids is not None:
        sem = _dense_residual_fill(
            sem, feat, dense_logits, dense_class_ids, dense_min_logit
        )

    if cc_min_area > 0:
        sem = _drop_small_islands(sem, cc_min_area)

    if num_classes == 13:
        sem = _map_40_to_13(sem)
    return sem


def _drop_small_islands(sem: np.ndarray, min_area: int) -> np.ndarray:
    """Relabel connected components smaller than ``min_area`` to neighbor majority.

    Why: SAM masks leak tiny fragments of wrong class along object boundaries;
    dropping components under ``min_area`` and reassigning them to the surrounding
    class reduces mIoU penalties on small erroneous regions without touching
    genuine small objects larger than the threshold.
    """
    from scipy import ndimage as ndi

    out = sem.copy()
    H, W = sem.shape
    for cid in np.unique(sem):
        if cid == 0:
            continue
        cls_mask = sem == cid
        labeled, n = ndi.label(cls_mask)
        if n == 0:
            continue
        sizes = ndi.sum(cls_mask, labeled, index=np.arange(1, n + 1))
        small_ids = np.where(sizes < min_area)[0] + 1
        if small_ids.size == 0:
            continue
        small_mask = np.isin(labeled, small_ids)
        out[small_mask] = 0
    # Reassign zeroed pixels to dominant neighbor class via dilation vote.
    zeroed = out == 0
    if zeroed.any():
        # Nearest-neighbor fill among non-zero labels.
        from scipy.ndimage import distance_transform_edt

        _, (iy, ix) = distance_transform_edt(zeroed, return_indices=True)
        out[zeroed] = out[iy[zeroed], ix[zeroed]]
    return out


def _ransac_structural_masks(feat: DepthFeatures) -> list[tuple[int, np.ndarray, float]]:
    """RANSAC-fit dominant planes; map to floor/ceiling/wall via gravity role."""
    planes = fit_dominant_planes(feat)
    out: list[tuple[int, np.ndarray, float]] = []
    for p in planes:
        role = p["role"]
        m = p["mask"]
        if role == "floor" and m.sum() > 2000:
            out.append((2, m, 0.7))
        elif role == "ceiling" and m.sum() > 1500:
            out.append((22, m, 0.65))
        elif role == "wall" and m.sum() > 3000:
            out.append((1, m, 0.5))
    return out


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


_DENSE_FILL_ALLOW = frozenset({
    # Tail classes GDino+SAM misses most. Pre-curated from per-class IoU
    # audit (week2_wshift) where these classes sat at 0-0.2 IoU.
    21,  # clothes
    23,  # books
    26,  # paper
    27,  # towel
    29,  # box
    31,  # person
    37,  # bag
    39,  # otherfurniture
    40,  # otherprop
})


def _dense_residual_fill(
    sem: np.ndarray,
    feat: DepthFeatures,
    dense_logits: np.ndarray,
    dense_class_ids: list[int],
    min_logit: float,
) -> np.ndarray:
    """Overwrite FALLBACK_PROP pixels with tail-class dense CLIP predictions.

    Why: after geometric fallback, pixels with no candidate + no floor/wall/
    ceiling prior are dumped into FALLBACK_PROP=40. Dense CLIP can recover
    tail-class signal there (person, bag, books, clothes...) without
    overwriting confident instance masks or structural planes.
    """
    target = sem == FALLBACK_PROP
    if not target.any():
        return sem
    allowed_cols = [
        i for i, cid in enumerate(dense_class_ids) if int(cid) in _DENSE_FILL_ALLOW
    ]
    if not allowed_cols:
        return sem
    sub_logits = dense_logits[allowed_cols]
    best_sub_idx = np.argmax(sub_logits, axis=0)
    best_logit = np.take_along_axis(
        sub_logits, best_sub_idx[None, ...], axis=0
    )[0]
    class_lut = np.array(
        [int(dense_class_ids[i]) for i in allowed_cols], dtype=np.uint8
    )
    best_cid = class_lut[best_sub_idx]
    confident = target & (best_logit >= min_logit)
    sem[confident] = best_cid[confident]
    return sem


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
