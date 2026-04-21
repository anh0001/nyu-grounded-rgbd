"""SigLIP-based mask reranker.

Crops each candidate SAM mask, encodes with a frozen SigLIP vision tower, and
scores against a bank of per-class text prompts. The score is returned in
[0, 1] (sigmoid of SigLIP logit) and can be used either to re-weight the
composite score, or to reassign a candidate's class if the top-1 text class
disagrees with GroundingDINO and the margin exceeds a threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from PIL import Image


DEFAULT_TEMPLATES: tuple[str, ...] = (
    "a photo of a {}",
    "a photo of a {} in a room",
    "an indoor photo of a {}",
    "a close-up photo of a {}",
)


@dataclass
class RerankResult:
    scores: np.ndarray            # (num_masks, num_classes) in [0, 1]
    top_class: np.ndarray         # (num_masks,) — class_id of argmax
    top_score: np.ndarray         # (num_masks,) — max score
    own_score: np.ndarray         # (num_masks,) — score for candidate's own class


class SigLIPReranker:
    def __init__(
        self,
        model_id: str = "google/siglip-base-patch16-224",
        device: str | torch.device = "cuda",
        class_names: Sequence[str] | None = None,
        class_ids: Sequence[int] | None = None,
        templates: Sequence[str] = DEFAULT_TEMPLATES,
    ) -> None:
        from transformers import AutoModel, AutoProcessor

        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.templates = tuple(templates)
        self.class_names = tuple(class_names) if class_names is not None else tuple()
        self.class_ids = tuple(class_ids) if class_ids is not None else tuple()
        self._text_features: torch.Tensor | None = None
        if self.class_names:
            self._encode_text_bank()

    @torch.inference_mode()
    def _encode_text_bank(self) -> None:
        all_prompts: list[str] = []
        per_class_slice: list[tuple[int, int]] = []
        cursor = 0
        for name in self.class_names:
            prompts = [t.format(name) for t in self.templates]
            all_prompts.extend(prompts)
            per_class_slice.append((cursor, cursor + len(prompts)))
            cursor += len(prompts)

        inputs = self.processor(
            text=all_prompts, return_tensors="pt", padding="max_length", truncation=True,
        ).to(self.device)
        out = self.model.get_text_features(**inputs)
        feats = out.pooler_output if hasattr(out, "pooler_output") else out
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        # average templates per class
        per_class = torch.stack([feats[s:e].mean(0) for s, e in per_class_slice], dim=0)
        per_class = per_class / per_class.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        self._text_features = per_class  # (num_classes, D)

    @torch.inference_mode()
    def score_masks(
        self,
        rgb: np.ndarray,
        masks: list[np.ndarray],
        candidate_class_ids: list[int],
        pad_ratio: float = 0.15,
        background_fill: str = "mean",
    ) -> RerankResult:
        if self._text_features is None:
            raise RuntimeError("text features not initialized")
        num_classes = len(self.class_ids)
        if not masks:
            empty = np.zeros((0, num_classes), dtype=np.float32)
            return RerankResult(
                scores=empty,
                top_class=np.zeros((0,), dtype=np.int32),
                top_score=np.zeros((0,), dtype=np.float32),
                own_score=np.zeros((0,), dtype=np.float32),
            )

        H, W = rgb.shape[:2]
        crops: list[Image.Image] = []
        for m in masks:
            ys, xs = np.where(m)
            if ys.size == 0:
                y1 = y2 = 0
                x1 = x2 = 1
            else:
                y1, y2 = int(ys.min()), int(ys.max()) + 1
                x1, x2 = int(xs.min()), int(xs.max()) + 1
            dy = int((y2 - y1) * pad_ratio)
            dx = int((x2 - x1) * pad_ratio)
            y1 = max(0, y1 - dy)
            y2 = min(H, y2 + dy)
            x1 = max(0, x1 - dx)
            x2 = min(W, x2 + dx)
            crop = rgb[y1:y2, x1:x2].astype(np.float32)
            crop_mask = m[y1:y2, x1:x2]
            if background_fill == "mean" and crop_mask.any():
                mean_rgb = crop[crop_mask].mean(axis=0)
                bg = ~crop_mask
                crop[bg] = mean_rgb
            elif background_fill == "black":
                crop[~crop_mask] = 0.0
            crops.append(Image.fromarray(crop.clip(0, 255).astype(np.uint8)))

        inputs = self.processor(images=crops, return_tensors="pt").to(self.device)
        img_out = self.model.get_image_features(**inputs)
        img_feats = img_out.pooler_output if hasattr(img_out, "pooler_output") else img_out
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        # Cosine similarity → softmax over the class set for sharp per-mask
        # distribution. Temperature 100 approximates the 1/0.01 scale used in
        # CLIP zero-shot classification.
        logits = (img_feats @ self._text_features.T) * 100.0  # (N, C)
        scores = torch.softmax(logits, dim=-1).cpu().numpy()

        top_idx = scores.argmax(axis=1)
        top_class = np.array([self.class_ids[i] for i in top_idx], dtype=np.int32)
        top_score = scores[np.arange(scores.shape[0]), top_idx].astype(np.float32)

        id_to_idx = {cid: i for i, cid in enumerate(self.class_ids)}
        own_score = np.zeros((scores.shape[0],), dtype=np.float32)
        for i, cid in enumerate(candidate_class_ids):
            idx = id_to_idx.get(int(cid))
            if idx is not None:
                own_score[i] = float(scores[i, idx])
        return RerankResult(
            scores=scores.astype(np.float32),
            top_class=top_class,
            top_score=top_score,
            own_score=own_score,
        )
