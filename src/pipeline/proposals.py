"""Glue: (image, depth) -> scored Candidate list."""
from __future__ import annotations

import numpy as np

from src.models.gdino import GroundingDINO
from src.models.sam_wrapper import SAMWrapper
from src.pipeline.depth_features import DepthFeatures, mask_depth_stats
from src.pipeline.mask_refine import (
    Candidate,
    box_nms,
    make_candidates,
    mask_nms,
    score_candidates,
)
from src.prompts.alias_bank import PromptChunk


def build_proposals(
    rgb: np.ndarray,
    depth_feat: DepthFeatures,
    gdino: GroundingDINO,
    sam: SAMWrapper,
    chunks: list[PromptChunk],
    image_id: str | None = None,
    box_threshold: float = 0.25,
    text_threshold: float = 0.25,
    box_iou_same: float = 0.5,
    box_iou_cross: float = 0.9,
    mask_iou_same: float = 0.5,
    mask_iou_cross: float = 0.85,
    per_chunk_thresholds: dict[str, tuple[float, float]] | None = None,
    clip_reranker=None,
    w_clip: float = 0.0,
    clip_reassign_margin: float = 0.0,
    clip_reassign_min_top: float = 0.0,
    clip_background_fill: str = "mean",
    clip_small_mask_context: float = 0.40,
    clip_small_mask_area_frac: float = 0.02,
    clip_bg_alpha: float = 0.5,
) -> list[Candidate]:
    # 1. detect
    bundle = gdino.detect(rgb, chunks, box_threshold=box_threshold,
                          text_threshold=text_threshold,
                          per_chunk_thresholds=per_chunk_thresholds)
    if not bundle.detections:
        return []

    # 2. SAM masks (batched if supported)
    sam.set_image(rgb, image_id=image_id)
    boxes = np.stack([d.box_xyxy for d in bundle.detections], axis=0)
    try:
        masks = sam.predict_boxes(boxes)
    except Exception:
        masks = [sam.predict_box(b) for b in boxes]

    # 3. candidates + per-mask depth stats
    cands = make_candidates(bundle.detections, masks)
    for c in cands:
        c.depth_stats = mask_depth_stats(depth_feat, c.mask)

    # 4. dedup boxes first (cheap)
    cands = box_nms(cands, iou_same=box_iou_same, iou_cross=box_iou_cross)

    # 4b. SigLIP reranking — score each candidate mask against class text bank.
    if clip_reranker is not None and cands:
        rerank = clip_reranker.score_masks(
            rgb=rgb,
            masks=[c.mask for c in cands],
            candidate_class_ids=[c.class_id for c in cands],
            background_fill=clip_background_fill,
            small_mask_context=clip_small_mask_context,
            small_mask_area_frac=clip_small_mask_area_frac,
            bg_alpha=clip_bg_alpha,
        )
        for i, c in enumerate(cands):
            c.clip_own_score = float(rerank.own_score[i])
            c.clip_top_score = float(rerank.top_score[i])
            c.clip_top_class = int(rerank.top_class[i])
            # Reassign class if SigLIP strongly disagrees with GDINO phrase.
            if (
                clip_reassign_margin > 0.0
                and c.clip_top_class != c.class_id
                and c.clip_top_score >= clip_reassign_min_top
                and (c.clip_top_score - c.clip_own_score) >= clip_reassign_margin
            ):
                c.class_id = c.clip_top_class
                c.clip_own_score = c.clip_top_score

    # 5. composite scoring, then mask NMS
    score_candidates(cands, w_clip=w_clip)
    cands = mask_nms(cands, iou_same=mask_iou_same, iou_cross=mask_iou_cross)
    return cands
