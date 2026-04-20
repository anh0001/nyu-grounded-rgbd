"""GroundingDINO wrapper (HF transformers AutoModelForZeroShotObjectDetection).

Inputs one PIL/np RGB image + a list of PromptChunks. Returns per-chunk box
detections keyed by NYUv2 class id.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image

from src.prompts.alias_bank import PromptChunk
from src.prompts.builders import alias_for_label, build_prompt


@dataclass
class Detection:
    class_id: int
    box_xyxy: np.ndarray          # (4,) pixel coords on input image
    score: float                  # detector score
    label_text: str               # raw label returned by HF model
    chunk: str                    # source chunk name


@dataclass
class DetectionBundle:
    image_size: tuple[int, int]   # (H, W)
    detections: list[Detection] = field(default_factory=list)


class GroundingDINO:
    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        device: str | torch.device = "cuda",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ) -> None:
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
        )
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device).eval()
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    @torch.inference_mode()
    def detect(
        self,
        image_rgb: np.ndarray,
        chunks: list[PromptChunk],
        box_threshold: float | None = None,
        text_threshold: float | None = None,
    ) -> DetectionBundle:
        if image_rgb.dtype != np.uint8:
            image_rgb = image_rgb.astype(np.uint8)
        pil = Image.fromarray(image_rgb)
        H, W = image_rgb.shape[:2]
        bundle = DetectionBundle(image_size=(H, W))

        bt = box_threshold if box_threshold is not None else self.box_threshold
        tt = text_threshold if text_threshold is not None else self.text_threshold

        for chunk in chunks:
            prompt = build_prompt(chunk)
            inputs = self.processor(
                images=pil, text=prompt.text, return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**inputs)

            # post-process: HF API: transformers>=4.51 uses `threshold=`, older uses `box_threshold=`.
            try:
                results = self.processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    threshold=bt,
                    text_threshold=tt,
                    target_sizes=[(H, W)],
                )[0]
            except TypeError:
                results = self.processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=bt,
                    text_threshold=tt,
                    target_sizes=[(H, W)],
                )[0]

            boxes = results["boxes"].detach().cpu().numpy()
            scores = results["scores"].detach().cpu().numpy()
            labels = results.get("text_labels")
            if labels is None:
                labels = results.get("labels", [])

            for b, s, lab in zip(boxes, scores, labels):
                cid = alias_for_label(str(lab), prompt.alias_to_class)
                if cid is None:
                    continue
                bundle.detections.append(
                    Detection(
                        class_id=int(cid),
                        box_xyxy=np.asarray(b, dtype=np.float32),
                        score=float(s),
                        label_text=str(lab),
                        chunk=chunk.name,
                    )
                )
        return bundle
