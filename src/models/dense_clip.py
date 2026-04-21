"""Dense CLIP features and NYUv2 text-bank construction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_TEMPLATES: tuple[str, ...] = (
    "{}",
    "a photo of a {}",
    "an indoor photo of a {}",
)


@dataclass
class DenseTextBank:
    class_ids: list[int]
    class_names: list[str]
    embeddings: np.ndarray  # (C, D) float32


class DenseCLIPEncoder:
    def __init__(
        self,
        model_id: str = "openai/clip-vit-large-patch14",
        input_size: int = 336,
        device: str | torch.device = "cuda",
    ) -> None:
        from transformers import AutoProcessor, CLIPModel

        self.model_id = model_id
        self.input_size = int(input_size)
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        image_processor = self.processor.image_processor
        self.mean = torch.tensor(image_processor.image_mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(image_processor.image_std, dtype=torch.float32).view(3, 1, 1)

    def _prepare_pixels(self, rgb: np.ndarray) -> torch.Tensor:
        resized = cv2.resize(
            rgb,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_CUBIC,
        )
        pixel_values = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        pixel_values = (pixel_values - self.mean) / self.std
        return pixel_values.unsqueeze(0).to(self.device)

    @staticmethod
    def _crop_with_mask(
        rgb: np.ndarray,
        mask: np.ndarray,
        pad_ratio: float = 0.15,
        background_fill: str = "mean",
    ) -> np.ndarray:
        ys, xs = np.where(mask)
        if ys.size == 0:
            return rgb
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        dy = int((y1 - y0) * pad_ratio)
        dx = int((x1 - x0) * pad_ratio)
        y0 = max(0, y0 - dy)
        y1 = min(rgb.shape[0], y1 + dy)
        x0 = max(0, x0 - dx)
        x1 = min(rgb.shape[1], x1 + dx)
        crop = rgb[y0:y1, x0:x1].astype(np.float32)
        crop_mask = mask[y0:y1, x0:x1]
        if background_fill == "mean" and crop_mask.any():
            fill = crop[crop_mask].mean(axis=0)
            crop[~crop_mask] = fill
        elif background_fill == "black":
            crop[~crop_mask] = 0.0
        return crop.clip(0, 255).astype(np.uint8)

    @torch.inference_mode()
    def build_text_bank(
        self,
        class_entries: Sequence[dict[str, object]],
        templates: Sequence[str] = DEFAULT_TEMPLATES,
    ) -> DenseTextBank:
        prompts: list[str] = []
        class_slices: list[tuple[int, int]] = []
        class_ids: list[int] = []
        class_names: list[str] = []
        cursor = 0
        for entry in class_entries:
            class_id = int(entry["class_id"])
            class_name = str(entry["name"])
            aliases = [class_name] + [str(a) for a in entry.get("aliases", [])]
            description = str(entry.get("description", "")).strip()

            unique_terms: list[str] = []
            seen: set[str] = set()
            for term in aliases:
                key = term.strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    unique_terms.append(term.strip())

            class_prompts: list[str] = []
            for term in unique_terms:
                for template in templates:
                    class_prompts.append(template.format(term))
            if description:
                class_prompts.append(description)
                class_prompts.append(f"{class_name}. {description}")

            prompts.extend(class_prompts)
            class_slices.append((cursor, cursor + len(class_prompts)))
            class_ids.append(class_id)
            class_names.append(class_name)
            cursor += len(class_prompts)

        inputs = self.processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        text_outputs = self.model.text_model(**inputs)
        pooled = text_outputs.pooler_output
        text_features = self.model.text_projection(pooled)
        text_features = F.normalize(text_features, dim=-1)

        per_class = torch.stack(
            [text_features[s:e].mean(dim=0) for s, e in class_slices],
            dim=0,
        )
        per_class = F.normalize(per_class, dim=-1)
        return DenseTextBank(
            class_ids=class_ids,
            class_names=class_names,
            embeddings=per_class.float().cpu().numpy(),
        )

    @torch.inference_mode()
    def encode_logits(self, rgb: np.ndarray, text_bank: DenseTextBank) -> np.ndarray:
        pixel_values = self._prepare_pixels(rgb)
        vision_outputs = self.model.vision_model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=True,
        )
        tokens = vision_outputs.last_hidden_state[:, 1:, :]
        token_features = self.model.visual_projection(tokens)
        token_features = F.normalize(token_features, dim=-1)

        side = int(round(tokens.shape[1] ** 0.5))
        token_features = token_features.view(1, side, side, -1).permute(0, 3, 1, 2)
        text_features = torch.from_numpy(text_bank.embeddings).to(
            self.device,
            dtype=token_features.dtype,
        )
        logits = torch.einsum("bdhw,cd->bchw", token_features, text_features)
        logits = logits * self.model.logit_scale.exp()
        logits = F.interpolate(
            logits,
            size=rgb.shape[:2],
            mode="bilinear",
            align_corners=False,
        )
        return logits[0].float().cpu().numpy()

    @torch.inference_mode()
    def encode_region_logits(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
        text_bank: DenseTextBank,
        pad_ratio: float = 0.15,
    ) -> np.ndarray:
        crop = self._crop_with_mask(rgb, mask, pad_ratio=pad_ratio)
        pixel_values = self._prepare_pixels(crop)
        vision_outputs = self.model.vision_model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=True,
        )
        cls_token = vision_outputs.last_hidden_state[:, 0, :]
        image_embed = self.model.visual_projection(cls_token)
        image_embed = F.normalize(image_embed, dim=-1)
        text_features = torch.from_numpy(text_bank.embeddings).to(
            self.device,
            dtype=image_embed.dtype,
        )
        logits = image_embed @ text_features.T
        logits = logits * self.model.logit_scale.exp()
        return logits[0].float().cpu().numpy()
