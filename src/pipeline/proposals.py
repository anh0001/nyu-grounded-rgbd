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
) -> list[Candidate]:
    # 1. detect
    bundle = gdino.detect(rgb, chunks, box_threshold=box_threshold,
                          text_threshold=text_threshold)
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

    # 4. dedup boxes first (cheap), then score, then mask NMS
    cands = box_nms(cands, iou_same=box_iou_same, iou_cross=box_iou_cross)
    score_candidates(cands)
    cands = mask_nms(cands, iou_same=mask_iou_same, iou_cross=mask_iou_cross)
    return cands
