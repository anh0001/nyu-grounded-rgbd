from __future__ import annotations

import numpy as np

from src.pipeline.dense_ov_fusion import (
    TileWindow,
    aggregate_region_logits,
    make_tile_windows,
    stitch_tile_maps,
)
from src.pipeline.depth_features import DepthFeatures
from src.pipeline.depth_logit_fusion import apply_depth_logit_fusion, logits_to_labels


def test_make_tile_windows_covers_image() -> None:
    windows = make_tile_windows((480, 640), tiles_y=2, tiles_x=2, overlap=0.25)
    cover = np.zeros((480, 640), dtype=np.uint8)
    for window in windows:
        cover[window.y0:window.y1, window.x0:window.x1] += 1
    assert len(windows) == 4
    assert cover.min() >= 1


def test_stitch_tile_maps_shape_and_overlap_average() -> None:
    windows = [
        TileWindow(0, 4, 0, 3),
        TileWindow(0, 4, 1, 4),
    ]
    tile_maps = [
        np.ones((2, 4, 3), dtype=np.float32),
        np.full((2, 4, 3), 3.0, dtype=np.float32),
    ]
    stitched = stitch_tile_maps(tile_maps, windows, (4, 4))
    assert stitched.shape == (2, 4, 4)
    assert np.allclose(stitched[:, :, 0], 1.0)
    assert np.allclose(stitched[:, :, 3], 3.0)
    assert np.allclose(stitched[:, :, 1:3], 2.0)


def test_aggregate_region_logits_preserves_shape() -> None:
    base_logits = np.zeros((40, 8, 8), dtype=np.float32)
    base_logits[5] = 2.0
    dino = np.ones((8, 8, 8), dtype=np.float32)
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    refined = aggregate_region_logits(base_logits, dino, [mask], min_region_area=4)
    assert refined.shape == base_logits.shape
    assert np.isfinite(refined).all()


def test_apply_depth_logit_fusion_respects_invalid_mask() -> None:
    logits = np.zeros((40, 4, 4), dtype=np.float32)
    logits[0] = 1.0
    feat = DepthFeatures(
        depth=np.ones((4, 4), dtype=np.float32),
        valid=np.zeros((4, 4), dtype=bool),
        edges=np.zeros((4, 4), dtype=np.float32),
        points=np.zeros((4, 4, 3), dtype=np.float32),
        normals=np.zeros((4, 4, 3), dtype=np.float32),
        up_proj=np.ones((4, 4), dtype=np.float32),
    )
    fused = apply_depth_logit_fusion(logits, feat)
    assert np.allclose(fused, logits)


def test_logits_to_labels_stays_in_range() -> None:
    logits = np.random.RandomState(0).randn(40, 6, 7).astype(np.float32)
    labels = logits_to_labels(logits)
    assert labels.shape == (6, 7)
    assert labels.min() >= 1
    assert labels.max() <= 40
