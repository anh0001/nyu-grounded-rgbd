"""Apply valid-gated depth priors to dense semantic logits."""
from __future__ import annotations

import numpy as np

from src.pipeline.depth_features import DepthFeatures


def apply_depth_logit_fusion(
    logits: np.ndarray,
    feat: DepthFeatures,
    wall_boost: float = 0.35,
    floor_boost: float = 0.9,
    ceiling_boost: float = 0.55,
    contradict_penalty: float = 0.2,
    floor_thr: float = 0.8,
    ceiling_thr: float = -0.8,
    wall_thr: float = 0.3,
) -> np.ndarray:
    out = logits.astype(np.float32).copy()
    height, width = feat.depth.shape
    ys = np.arange(height, dtype=np.float32)[:, None].repeat(width, axis=1)
    valid = feat.valid.astype(bool)

    floor_like = valid & (feat.up_proj > floor_thr) & (ys > 0.4 * height)
    ceiling_like = valid & (feat.up_proj < ceiling_thr) & (ys < 0.35 * height)
    wall_like = valid & (np.abs(feat.up_proj) < wall_thr) & (~floor_like) & (~ceiling_like)

    wall_idx = 0
    floor_idx = 1
    ceiling_idx = 21

    out[floor_idx, floor_like] += float(floor_boost)
    out[wall_idx, floor_like] -= float(contradict_penalty)
    out[ceiling_idx, floor_like] -= float(contradict_penalty)

    out[ceiling_idx, ceiling_like] += float(ceiling_boost)
    out[wall_idx, ceiling_like] -= float(contradict_penalty)
    out[floor_idx, ceiling_like] -= float(contradict_penalty)

    out[wall_idx, wall_like] += float(wall_boost)
    return out


def logits_to_labels(logits: np.ndarray) -> np.ndarray:
    labels = np.argmax(logits, axis=0).astype(np.uint8) + 1
    return np.clip(labels, 1, 40)
