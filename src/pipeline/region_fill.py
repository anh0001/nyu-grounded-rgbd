"""Residual-region backfill (optional Day-6 stage).

Uses RGB+depth SLIC superpixels over unassigned pixels and assigns each
superpixel to the best-overlapping Candidate or fallback class.
"""
from __future__ import annotations

import numpy as np

from src.pipeline.depth_features import DepthFeatures
from src.pipeline.mask_refine import Candidate
from src.pipeline.semantic_fusion import FALLBACK_PROP


def fill_residual_slic(
    sem: np.ndarray,
    rgb: np.ndarray,
    feat: DepthFeatures,
    cands: list[Candidate],
    n_segments: int = 400,
    min_overlap: float = 0.1,
) -> np.ndarray:
    try:
        from skimage.segmentation import slic
    except ImportError:
        return sem
    unl = sem == 0
    if not unl.any():
        return sem
    # SLIC on RGB + depth as 4th channel.
    d = feat.depth / max(feat.depth.max(), 1e-3)
    x = np.concatenate([rgb.astype(np.float32) / 255.0, d[..., None]], axis=-1)
    segs = slic(x, n_segments=n_segments, compactness=10.0, channel_axis=-1, start_label=1)
    out = sem.copy()
    for sid in np.unique(segs[unl]):
        if sid == 0:
            continue
        smask = (segs == sid) & unl
        if smask.sum() == 0:
            continue
        best_cid, best_overlap = None, 0.0
        for c in cands:
            overlap = np.logical_and(smask, c.mask).sum() / smask.sum()
            if overlap > best_overlap and overlap > min_overlap:
                best_overlap = overlap
                best_cid = c.class_id
        out[smask] = best_cid if best_cid is not None else FALLBACK_PROP
    return out
