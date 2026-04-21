"""Generate SAM regions for dense logit aggregation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class SAMRegion:
    mask: np.ndarray
    score: float


class SAMRegionGenerator:
    def __init__(
        self,
        model_id: str = "facebook/sam-vit-huge",
        device: str = "cuda",
        points_per_batch: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.92,
        min_mask_area: int = 256,
        max_regions: int = 64,
    ) -> None:
        from transformers import pipeline

        self.points_per_batch = int(points_per_batch)
        self.pred_iou_thresh = float(pred_iou_thresh)
        self.stability_score_thresh = float(stability_score_thresh)
        self.min_mask_area = int(min_mask_area)
        self.max_regions = int(max_regions)
        device_id = 0 if str(device).startswith("cuda") else -1
        self.generator = pipeline(
            task="mask-generation",
            model=model_id,
            device=device_id,
        )

    def _iter_raw_regions(self, outputs) -> list[SAMRegion]:
        regions: list[SAMRegion] = []
        if isinstance(outputs, dict):
            masks = outputs.get("masks", [])
            scores = outputs.get("scores", [0.0] * len(masks))
            for mask, score in zip(masks, scores):
                regions.append(SAMRegion(mask=np.asarray(mask).astype(bool), score=float(score)))
            return regions

        for item in outputs:
            if isinstance(item, dict):
                mask = item.get("mask")
                if mask is None:
                    mask = item.get("segmentation")
                if mask is None:
                    continue
                score = item.get("score")
                if score is None:
                    score = item.get("predicted_iou", 0.0)
                regions.append(SAMRegion(mask=np.asarray(mask).astype(bool), score=float(score)))
            else:
                regions.append(SAMRegion(mask=np.asarray(item).astype(bool), score=0.0))
        return regions

    def generate(self, rgb: np.ndarray) -> list[np.ndarray]:
        outputs = self.generator(
            Image.fromarray(rgb),
            points_per_batch=self.points_per_batch,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
        )
        regions = [
            region for region in self._iter_raw_regions(outputs)
            if int(region.mask.sum()) >= self.min_mask_area
        ]
        regions.sort(key=lambda x: (x.score, int(x.mask.sum())), reverse=True)
        return [region.mask for region in regions[:self.max_regions]]
