"""Seed SAM candidates from dense-CLIP logit peaks for tail NYU40 classes.

Why: GroundingDINO often returns zero detections for classes like `person`,
`bag`, `otherstructure`, `otherfurniture` even with expanded aliases. A frozen
dense-CLIP pathway (SCLIP/MaskCLIP-style) produces per-class dense logits that
fire on these classes; thresholding + connected components yields point prompts
that SAM turns into precise masks. The resulting masks are inserted as normal
Candidate objects before NMS/rasterization, so the rest of the pipeline is
unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.models.sam_wrapper import SAMWrapper
from src.pipeline.mask_refine import Candidate


TAIL_CLASSES: tuple[int, ...] = (
    21,  # clothes
    23,  # books
    26,  # paper
    27,  # towel
    29,  # box
    31,  # person
    35,  # lamp
    37,  # bag
    38,  # otherstructure
    39,  # otherfurniture
)


@dataclass
class DenseSeedConfig:
    min_logit: float = 10.0
    min_component_area: int = 400
    max_seeds_per_class: int = 2
    sam_score_floor: float = 0.6


def _top_components(
    logit_map: np.ndarray,
    min_logit: float,
    min_area: int,
    max_components: int,
) -> list[tuple[float, int, int]]:
    """Return list of (peak_logit, peak_y, peak_x) for connected high-logit regions."""
    from scipy import ndimage as ndi

    binary = logit_map >= min_logit
    if not binary.any():
        return []
    labeled, n = ndi.label(binary)
    if n == 0:
        return []
    sizes = ndi.sum(binary, labeled, index=np.arange(1, n + 1))
    keep = [i + 1 for i, s in enumerate(sizes) if s >= min_area]
    if not keep:
        return []
    scored: list[tuple[float, int, int]] = []
    for comp_id in keep:
        region = labeled == comp_id
        comp_logits = np.where(region, logit_map, -np.inf)
        peak_idx = int(np.argmax(comp_logits))
        peak_y, peak_x = divmod(peak_idx, logit_map.shape[1])
        scored.append((float(comp_logits[peak_y, peak_x]), peak_y, peak_x))
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[:max_components]


def seed_candidates_from_dense(
    rgb: np.ndarray,
    dense_logits: np.ndarray,
    dense_class_ids: list[int],
    sam: SAMWrapper,
    cfg: DenseSeedConfig,
    existing_class_ids: set[int] | None = None,
    exclusion_mask: np.ndarray | None = None,
) -> list[Candidate]:
    """Build Candidate list for tail classes from dense-CLIP peaks + SAM points.

    - ``existing_class_ids``: tail classes already covered by GDino; skip them.
    - ``exclusion_mask``: (H, W) bool; skip seeds landing on these pixels
      (e.g., pixels already assigned a high-confidence instance mask).

    Peak selection uses the *argmax* requirement: the pixel must have the
    tail class as its top-1 prediction across all ``dense_class_ids``. This
    eliminates spurious seeds on pixels where a head class (wall/floor) has
    higher logit but the tail class also crosses ``cfg.min_logit``.
    """
    existing = existing_class_ids or set()
    id_to_idx = {int(cid): i for i, cid in enumerate(dense_class_ids)}
    sam_cands: list[Candidate] = []

    argmax_cid_idx = np.argmax(dense_logits, axis=0)

    for tail_cid in TAIL_CLASSES:
        if tail_cid in existing:
            continue
        col = id_to_idx.get(int(tail_cid))
        if col is None:
            continue
        logit_map = dense_logits[col].copy()
        logit_map[argmax_cid_idx != col] = -np.inf
        peaks = _top_components(
            logit_map,
            min_logit=cfg.min_logit,
            min_area=cfg.min_component_area,
            max_components=cfg.max_seeds_per_class,
        )
        for peak_logit, py, px in peaks:
            if exclusion_mask is not None and exclusion_mask[py, px]:
                continue
            result = sam.predict_points(np.array([[px, py]], dtype=np.float32))
            if result.score < cfg.sam_score_floor:
                continue
            if not result.mask.any():
                continue
            ys, xs = np.where(result.mask)
            bbox = np.array(
                [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32
            )
            cand = Candidate(
                class_id=int(tail_cid),
                box_xyxy=bbox,
                box_score=0.25,
                mask=result.mask,
                mask_score=float(result.score) * 0.6,
                chunk="dense_seed",
                text_label=f"dense_{tail_cid}",
                score=0.3 * float(result.score),
            )
            sam_cands.append(cand)
    return sam_cands
