"""SAM / MobileSAM / HQ-SAM thin wrapper for box-prompted masks.

Backends (pick first available at construct time):
  - "hqsam"     : segment_anything_hq (HQ-SAM)
  - "mobilesam" : mobile_sam
  - "sam"       : segment_anything

All expose the same predictor API: set_image(rgb); predict(box=...).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class SAMMaskResult:
    mask: np.ndarray        # (H,W) bool
    score: float            # SAM quality score


class SAMWrapper:
    def __init__(
        self,
        backend: str = "auto",
        checkpoint: str | Path | None = None,
        model_type: str = "vit_h",
        device: str | torch.device = "cuda",
    ) -> None:
        self.device = torch.device(device)
        self.backend = backend if backend != "auto" else self._pick_backend()
        self.predictor = self._load(self.backend, checkpoint, model_type)
        self._current_image_id: str | None = None

    @staticmethod
    def _pick_backend() -> str:
        for cand in ("segment_anything_hq", "mobile_sam", "segment_anything"):
            try:
                __import__(cand)
                return {
                    "segment_anything_hq": "hqsam",
                    "mobile_sam": "mobilesam",
                    "segment_anything": "sam",
                }[cand]
            except ImportError:
                continue
        raise ImportError("install one of: segment-anything-hq, mobile-sam, segment-anything")

    def _load(self, backend: str, ckpt: str | Path | None, model_type: str):
        ckpt = str(ckpt) if ckpt else None
        if backend == "hqsam":
            from segment_anything_hq import SamPredictor, sam_model_registry
            assert ckpt, "hqsam checkpoint required"
            sam = sam_model_registry[model_type](checkpoint=ckpt).to(self.device).eval()
            return SamPredictor(sam)
        if backend == "mobilesam":
            from mobile_sam import SamPredictor, sam_model_registry
            assert ckpt, "mobilesam checkpoint required"
            sam = sam_model_registry["vit_t"](checkpoint=ckpt).to(self.device).eval()
            return SamPredictor(sam)
        if backend == "sam":
            from segment_anything import SamPredictor, sam_model_registry
            assert ckpt, "sam checkpoint required"
            sam = sam_model_registry[model_type](checkpoint=ckpt).to(self.device).eval()
            return SamPredictor(sam)
        raise ValueError(backend)

    @torch.inference_mode()
    def set_image(self, rgb: np.ndarray, image_id: str | None = None) -> None:
        if image_id is not None and image_id == self._current_image_id:
            return
        self.predictor.set_image(rgb)
        self._current_image_id = image_id

    @torch.inference_mode()
    def predict_box(self, box_xyxy: np.ndarray) -> SAMMaskResult:
        box = np.asarray(box_xyxy, dtype=np.float32)[None, :]  # (1,4)
        masks, scores, _ = self.predictor.predict(
            box=box, multimask_output=False,
        )
        return SAMMaskResult(mask=masks[0].astype(bool), score=float(scores[0]))

    @torch.inference_mode()
    def predict_boxes(self, boxes_xyxy: np.ndarray) -> list[SAMMaskResult]:
        if len(boxes_xyxy) == 0:
            return []
        import torch as _t
        boxes = _t.as_tensor(np.asarray(boxes_xyxy, dtype=np.float32), device=self.device)
        transformed = self.predictor.transform.apply_boxes_torch(boxes, self.predictor.original_size)
        masks, scores, _ = self.predictor.predict_torch(
            point_coords=None, point_labels=None, boxes=transformed, multimask_output=False,
        )
        masks = masks.squeeze(1).cpu().numpy()
        scores = scores.squeeze(-1).cpu().numpy()
        return [SAMMaskResult(mask=m.astype(bool), score=float(s)) for m, s in zip(masks, scores)]
