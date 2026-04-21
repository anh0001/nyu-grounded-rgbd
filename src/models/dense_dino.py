"""Dense DINOv2 features for mask-consistency refinement."""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class DenseDINOEncoder:
    def __init__(
        self,
        model_id: str = "facebook/dinov2-large",
        input_size: int = 336,
        output_dims: int = 64,
        device: str | torch.device = "cuda",
    ) -> None:
        from transformers import AutoImageProcessor, Dinov2Model

        self.model_id = model_id
        self.input_size = int(input_size)
        self.output_dims = int(output_dims)
        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = Dinov2Model.from_pretrained(model_id).to(self.device).eval()
        self.mean = torch.tensor(self.processor.image_mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(self.processor.image_std, dtype=torch.float32).view(3, 1, 1)

    def _prepare_pixels(self, rgb: np.ndarray) -> torch.Tensor:
        resized = cv2.resize(
            rgb,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_CUBIC,
        )
        pixel_values = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        pixel_values = (pixel_values - self.mean) / self.std
        return pixel_values.unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def encode_features(self, rgb: np.ndarray) -> np.ndarray:
        pixel_values = self._prepare_pixels(rgb)
        outputs = self.model(pixel_values=pixel_values)
        tokens = outputs.last_hidden_state[:, 1:, :]
        side = int(round(tokens.shape[1] ** 0.5))
        token_features = tokens.view(1, side, side, -1).permute(0, 3, 1, 2)

        if token_features.shape[1] > self.output_dims:
            idx = torch.linspace(
                0,
                token_features.shape[1] - 1,
                steps=self.output_dims,
                device=token_features.device,
            ).round().long()
            token_features = token_features[:, idx, :, :]

        token_features = F.normalize(token_features, dim=1)
        token_features = F.interpolate(
            token_features,
            size=rgb.shape[:2],
            mode="bilinear",
            align_corners=False,
        )
        return token_features[0].float().cpu().numpy()
