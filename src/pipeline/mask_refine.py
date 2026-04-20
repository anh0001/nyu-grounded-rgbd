"""Class-aware box + mask NMS, plus composite scoring."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.models.gdino import Detection
from src.models.sam_wrapper import SAMMaskResult


@dataclass
class Candidate:
    class_id: int
    box_xyxy: np.ndarray        # (4,)
    box_score: float
    mask: np.ndarray            # (H,W) bool
    mask_score: float
    chunk: str
    text_label: str
    depth_stats: dict[str, float] = field(default_factory=dict)
    score: float = 0.0          # composite, set by score_candidates


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / max(union, 1e-6))


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / max(union, 1))


def box_nms(
    cands: list[Candidate],
    iou_same: float = 0.5,
    iou_cross: float = 0.9,
) -> list[Candidate]:
    order = sorted(range(len(cands)), key=lambda i: cands[i].box_score, reverse=True)
    kept: list[int] = []
    suppressed = set()
    for i in order:
        if i in suppressed:
            continue
        kept.append(i)
        for j in order:
            if j == i or j in suppressed:
                continue
            iou = box_iou(cands[i].box_xyxy, cands[j].box_xyxy)
            thr = iou_same if cands[i].class_id == cands[j].class_id else iou_cross
            if iou >= thr:
                suppressed.add(j)
    return [cands[i] for i in kept]


def mask_nms(
    cands: list[Candidate],
    iou_same: float = 0.5,
    iou_cross: float = 0.85,
) -> list[Candidate]:
    order = sorted(range(len(cands)), key=lambda i: cands[i].score, reverse=True)
    kept: list[int] = []
    suppressed = set()
    for i in order:
        if i in suppressed:
            continue
        kept.append(i)
        mi = cands[i].mask
        for j in order:
            if j == i or j in suppressed:
                continue
            iou = mask_iou(mi, cands[j].mask)
            thr = iou_same if cands[i].class_id == cands[j].class_id else iou_cross
            if iou >= thr:
                suppressed.add(j)
    return [cands[i] for i in kept]


def class_geometry_prior(class_id: int, stats: dict[str, float]) -> float:
    """Heuristic bonus in [-1,1] from depth/position priors."""
    # NYU40 ids: 1=wall, 2=floor, 22=ceiling, 9=window, 13=blinds, 16=curtain,
    # 19=mirror, 20=floor mat, 38=otherstructure.
    if stats.get("area", 0) == 0:
        return 0.0
    up = stats.get("up_mean", 0.0)
    bot = stats.get("bottom_frac", 0.0)
    top = stats.get("top_frac", 0.0)
    area = stats.get("area", 0)
    if class_id == 2 or class_id == 20:            # floor / floor mat
        return 0.6 * up + 0.4 * bot
    if class_id == 22:                             # ceiling
        return 0.6 * (-up) + 0.4 * top
    if class_id == 1:                              # wall
        # wall normals tend to be horizontal (|up|~0) and large area
        return 0.4 * (1.0 - abs(up)) + 0.2 * (area > 20000)
    if class_id in (9, 13, 16, 19, 28):            # window/blinds/curtain/mirror/shower curtain
        return 0.3 * (1.0 - abs(up))
    return 0.0


def score_candidates(
    cands: list[Candidate],
    w_box: float = 1.0, w_mask: float = 0.5, w_depth: float = 0.3,
    w_geo: float = 0.4,
) -> None:
    for c in cands:
        depth_consistency = 1.0 - min(1.0, c.depth_stats.get("depth_std", 0.0) / 0.5)
        geo = class_geometry_prior(c.class_id, c.depth_stats)
        c.score = (
            w_box * c.box_score
            + w_mask * c.mask_score
            + w_depth * depth_consistency
            + w_geo * geo
        )


def make_candidates(
    detections: list[Detection],
    masks: list[SAMMaskResult],
) -> list[Candidate]:
    assert len(detections) == len(masks)
    out: list[Candidate] = []
    for d, m in zip(detections, masks):
        if m.mask.sum() == 0:
            continue
        out.append(Candidate(
            class_id=d.class_id,
            box_xyxy=d.box_xyxy,
            box_score=d.score,
            mask=m.mask,
            mask_score=m.score,
            chunk=d.chunk,
            text_label=d.label_text,
        ))
    return out
