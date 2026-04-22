"""Export the repo's NYUv2 layout into backend-specific supervised layouts."""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


DFORMER_EXPORT_DIRNAME = "dformerv2"
ESANET_EXPORT_DIRNAME = "esanet"


@dataclass(frozen=True)
class ExportSummary:
    backend: str
    root: Path
    train_count: int
    test_count: int


def export_nyuv2_for_backend(
    source_root: Path,
    splits_file: Path,
    out_root: Path,
    backend: str,
    force: bool = False,
) -> ExportSummary:
    backend = str(backend).lower()
    if backend == "dformerv2":
        return export_nyuv2_for_dformerv2(source_root, splits_file, out_root, force=force)
    if backend == "esanet":
        return export_nyuv2_for_esanet(source_root, splits_file, out_root, force=force)
    raise ValueError(f"unsupported supervised backend: {backend}")


def export_nyuv2_for_dformerv2(
    source_root: Path,
    splits_file: Path,
    out_root: Path,
    force: bool = False,
) -> ExportSummary:
    split_ids = _load_splits(splits_file)
    rgb_dir = out_root / "RGB"
    depth_dir = out_root / "Depth"
    label_dir = out_root / "Label"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for idx in split_ids["train"] + split_ids["test"]:
        stem = f"{idx:04d}"
        _link_or_copy(
            source_root / "rgb" / f"{stem}.png",
            rgb_dir / f"{stem}.png",
            force=force,
        )
        _link_or_copy(
            source_root / "labels40" / f"{stem}.png",
            label_dir / f"{stem}.png",
            force=force,
        )
        depth_dst = depth_dir / f"{stem}.png"
        if force or not depth_dst.exists():
            depth = np.load(source_root / "depth" / f"{stem}.npy").astype(np.float32)
            depth_img = _depth_to_dformerv2_uint8(depth)
            Image.fromarray(depth_img, mode="L").save(depth_dst)

    _write_index_file(out_root / "train.txt", split_ids["train"], prefix="RGB", suffix=".png")
    _write_index_file(out_root / "test.txt", split_ids["test"], prefix="RGB", suffix=".png")
    return ExportSummary(
        backend="dformerv2",
        root=out_root,
        train_count=len(split_ids["train"]),
        test_count=len(split_ids["test"]),
    )


def export_nyuv2_for_esanet(
    source_root: Path,
    splits_file: Path,
    out_root: Path,
    force: bool = False,
) -> ExportSummary:
    split_ids = _load_splits(splits_file)
    for split_name in ("train", "test"):
        (out_root / split_name / "rgb").mkdir(parents=True, exist_ok=True)
        (out_root / split_name / "depth").mkdir(parents=True, exist_ok=True)
        (out_root / split_name / "labels_40").mkdir(parents=True, exist_ok=True)

    for split_name, ids in split_ids.items():
        for idx in ids:
            stem = f"{idx:04d}"
            split_root = out_root / split_name
            _link_or_copy(
                source_root / "rgb" / f"{stem}.png",
                split_root / "rgb" / f"{stem}.png",
                force=force,
            )
            _link_or_copy(
                source_root / "labels40" / f"{stem}.png",
                split_root / "labels_40" / f"{stem}.png",
                force=force,
            )
            depth_dst = split_root / "depth" / f"{stem}.png"
            if force or not depth_dst.exists():
                depth = np.load(source_root / "depth" / f"{stem}.npy").astype(np.float32)
                depth_mm = _depth_to_esanet_uint16(depth)
                Image.fromarray(depth_mm, mode="I;16").save(depth_dst)

    _write_flat_index(out_root / "train.txt", split_ids["train"])
    _write_flat_index(out_root / "test.txt", split_ids["test"])
    return ExportSummary(
        backend="esanet",
        root=out_root,
        train_count=len(split_ids["train"]),
        test_count=len(split_ids["test"]),
    )


def _load_splits(path: Path) -> dict[str, list[int]]:
    with open(path) as f:
        raw = json.load(f)
    return {
        "train": [int(x) for x in raw["train"]],
        "test": [int(x) for x in raw["test"]],
    }


def _depth_to_dformerv2_uint8(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth)
    if not valid.any():
        return np.zeros(depth.shape, dtype=np.uint8)

    out = np.zeros(depth.shape, dtype=np.uint8)
    finite = depth[valid]
    d_min = float(finite.min())
    d_max = float(finite.max())
    if abs(d_max - d_min) < 1e-8:
        out[valid] = 255
        return out

    normalized = (depth[valid] - d_min) / (d_max - d_min)
    out[valid] = np.clip(np.round((1.0 - normalized) * 255.0), 0, 255).astype(np.uint8)
    return out


def _depth_to_esanet_uint16(depth: np.ndarray) -> np.ndarray:
    depth_mm = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth_mm = np.clip(np.round(depth_mm * 1000.0), 0, np.iinfo(np.uint16).max)
    return depth_mm.astype(np.uint16)


def _write_index_file(path: Path, ids: list[int], prefix: str, suffix: str) -> None:
    lines = [f"{prefix}/{idx:04d}{suffix}" for idx in ids]
    path.write_text("\n".join(lines) + "\n")


def _write_flat_index(path: Path, ids: list[int]) -> None:
    lines = [f"{idx:04d}" for idx in ids]
    path.write_text("\n".join(lines) + "\n")


def _link_or_copy(src: Path, dst: Path, force: bool = False) -> None:
    if dst.exists() or dst.is_symlink():
        if not force:
            return
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()

    dst.parent.mkdir(parents=True, exist_ok=True)
    rel_src = os.path.relpath(src, start=dst.parent)
    try:
        os.symlink(rel_src, dst)
    except OSError:
        shutil.copy2(src, dst)
