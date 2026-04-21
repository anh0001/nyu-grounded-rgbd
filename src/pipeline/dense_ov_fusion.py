"""Dense CLIP+DINO fusion with SAM region aggregation."""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np


@dataclass(frozen=True)
class TileWindow:
    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.y1 - self.y0, self.x1 - self.x0)


def compute_axis_windows(length: int, num_tiles: int, overlap: float) -> list[tuple[int, int]]:
    if num_tiles <= 1:
        return [(0, int(length))]
    denom = num_tiles - (num_tiles - 1) * float(overlap)
    tile = int(ceil(length / max(denom, 1.0)))
    tile = min(tile, length)
    last_start = max(length - tile, 0)
    starts = [int(round(i * last_start / (num_tiles - 1))) for i in range(num_tiles)]
    windows: list[tuple[int, int]] = []
    for start in starts:
        end = min(start + tile, length)
        windows.append((max(end - tile, 0), end))
    return windows


def make_tile_windows(
    image_shape: tuple[int, int],
    tiles_y: int = 2,
    tiles_x: int = 2,
    overlap: float = 0.25,
) -> list[TileWindow]:
    height, width = image_shape
    ys = compute_axis_windows(height, tiles_y, overlap)
    xs = compute_axis_windows(width, tiles_x, overlap)
    return [TileWindow(y0, y1, x0, x1) for y0, y1 in ys for x0, x1 in xs]


def stitch_tile_maps(
    tile_maps: list[np.ndarray],
    windows: list[TileWindow],
    out_shape: tuple[int, int],
) -> np.ndarray:
    if not tile_maps:
        raise ValueError("tile_maps must not be empty")
    channels = tile_maps[0].shape[0]
    height, width = out_shape
    accum = np.zeros((channels, height, width), dtype=np.float32)
    weight = np.zeros((height, width), dtype=np.float32)
    for tile_map, window in zip(tile_maps, windows):
        accum[:, window.y0:window.y1, window.x0:window.x1] += tile_map.astype(np.float32)
        weight[window.y0:window.y1, window.x0:window.x1] += 1.0
    return accum / np.maximum(weight[None, ...], 1e-6)


def aggregate_region_logits(
    base_logits: np.ndarray,
    dino_features: np.ndarray,
    masks: list[np.ndarray],
    min_region_area: int = 256,
) -> np.ndarray:
    if not masks:
        return base_logits.copy()

    channels, height, width = base_logits.shape
    assert dino_features.shape[1:] == (height, width)
    accum = np.zeros_like(base_logits, dtype=np.float32)
    weight = np.zeros((height, width), dtype=np.float32)

    for mask in masks:
        region = np.asarray(mask).astype(bool)
        area = int(region.sum())
        if area < min_region_area:
            continue

        pooled_logits = base_logits[:, region].mean(axis=1).astype(np.float32)
        region_feat = dino_features[:, region].mean(axis=1).astype(np.float32)
        region_feat /= max(np.linalg.norm(region_feat), 1e-6)

        region_pixels = dino_features[:, region].T
        sims = region_pixels @ region_feat
        sims = np.clip((sims + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)
        pixel_weight = 0.5 + 0.5 * sims

        accum[:, region] += pooled_logits[:, None] * pixel_weight[None, :]
        weight[region] += pixel_weight

    fallback = base_logits.astype(np.float32)
    refined = np.where(
        weight[None, ...] > 0,
        accum / np.maximum(weight[None, ...], 1e-6),
        fallback,
    )
    return refined


def classify_region_masks(
    rgb: np.ndarray,
    clip_encoder,
    text_bank,
    dino_features: np.ndarray,
    masks: list[np.ndarray],
    min_region_area: int = 256,
) -> np.ndarray:
    num_classes = len(text_bank.class_ids)
    height, width = rgb.shape[:2]
    accum = np.zeros((num_classes, height, width), dtype=np.float32)
    weight = np.zeros((height, width), dtype=np.float32)

    for mask in masks:
        region = np.asarray(mask).astype(bool)
        area = int(region.sum())
        if area < min_region_area:
            continue
        region_logits = clip_encoder.encode_region_logits(rgb, region, text_bank)
        region_feat = dino_features[:, region].mean(axis=1).astype(np.float32)
        region_feat /= max(np.linalg.norm(region_feat), 1e-6)
        pixel_feats = dino_features[:, region].T
        sims = pixel_feats @ region_feat
        sims = np.clip((sims + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)
        pixel_weight = 0.5 + 0.5 * sims
        accum[:, region] += region_logits[:, None] * pixel_weight[None, :]
        weight[region] += pixel_weight

    return np.where(
        weight[None, ...] > 0,
        accum / np.maximum(weight[None, ...], 1e-6),
        0.0,
    )


def run_dense_ov_fusion(
    rgb: np.ndarray,
    clip_encoder,
    dino_encoder,
    sam_generator,
    text_bank,
    tiles_y: int = 2,
    tiles_x: int = 2,
    overlap: float = 0.25,
    clip_logit_weight: float = 0.65,
    region_logit_weight: float = 0.35,
    min_region_area: int = 256,
) -> np.ndarray:
    windows = make_tile_windows(rgb.shape[:2], tiles_y=tiles_y, tiles_x=tiles_x, overlap=overlap)
    clip_tiles: list[np.ndarray] = []
    dino_tiles: list[np.ndarray] = []

    for window in windows:
        crop = rgb[window.y0:window.y1, window.x0:window.x1]
        clip_tiles.append(clip_encoder.encode_logits(crop, text_bank))
        dino_tiles.append(dino_encoder.encode_features(crop))

    base_logits = stitch_tile_maps(clip_tiles, windows, rgb.shape[:2])
    dino_features = stitch_tile_maps(dino_tiles, windows, rgb.shape[:2])
    region_masks = sam_generator.generate(rgb)
    region_logits = classify_region_masks(
        rgb=rgb,
        clip_encoder=clip_encoder,
        text_bank=text_bank,
        dino_features=dino_features,
        masks=region_masks,
        min_region_area=min_region_area,
    )
    return (
        float(clip_logit_weight) * base_logits +
        float(region_logit_weight) * region_logits
    ).astype(np.float32)
