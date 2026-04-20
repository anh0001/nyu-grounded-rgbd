"""Download NYUv2 labeled .mat + Gupta split, extract RGB/depth/labels.

Outputs under data/nyuv2/:
  rgb/{0001..1449}.png        uint8 (480,640,3)
  depth/{0001..1449}.npy      float32 in-painted depth, meters
  depth_raw/{0001..1449}.npy  float32 raw depth (NaN for invalid)
  labels40/{0001..1449}.png   uint8 single-channel, 0=unlabeled, 1..40
  labels13/{0001..1449}.png   uint8, 0..13
  splits/gupta_795_654.json   {"train":[...1-based ids], "test":[...]}

Sources:
  labeled mat: http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
  splits mat:  http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat
  classMapping40: derived from NYU toolbox ClassMapping40.m (Gupta ordering).
                  We also fetch classMapping40.mat if available; else user must supply.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
from tqdm import tqdm

LABELED_URL = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
SPLITS_URL = "http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat"
NYUV2_META_REPO = "https://github.com/ankurhanda/nyuv2-meta-data"


def resolve_meta_mat(meta_path: Path, what: str, alt_names: tuple[str, ...] = ()) -> Path:
    """Resolve required NYUv2 metadata, rejecting zero-byte placeholder files."""
    candidates = (meta_path, *(meta_path.parent / name for name in alt_names))
    empty: list[Path] = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.stat().st_size == 0:
            empty.append(candidate)
            continue
        return candidate

    if empty:
        paths = ", ".join(str(p) for p in empty)
        raise FileNotFoundError(
            f"Found empty/truncated {what} at {paths}. "
            f"Delete and re-download from {NYUV2_META_REPO}."
        )

    looked_for = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Need {what}. Looked for: {looked_for}. "
        f"Fetch from {NYUV2_META_REPO} and place under {meta_path.parent}."
    )


def download(url: str, dst: Path) -> None:
    if dst.exists():
        print(f"exist {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"download {url} -> {dst}")
    with urllib.request.urlopen(url) as r, open(dst, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        bar = tqdm(total=total, unit="B", unit_scale=True)
        while chunk := r.read(1 << 20):
            f.write(chunk)
            bar.update(len(chunk))
        bar.close()


def load_class_mapping_40(meta_path: Path) -> np.ndarray:
    """Load 894-vector mapping raw class idx -> NYU-40 id (0=unlabeled)."""
    import scipy.io as sio

    meta_path = resolve_meta_mat(meta_path, "NYU40 class mapping")
    m = sio.loadmat(meta_path)
    # toolbox field name: mapClass (1x894 or 894x1)
    arr = np.array(m.get("mapClass") if "mapClass" in m else m["mapping"]).squeeze()
    assert arr.shape[0] == 894, f"expected 894 classes, got {arr.shape}"
    return arr.astype(np.int32)


def load_class_mapping_13(meta_path: Path) -> np.ndarray:
    """Load NYU-13 mapping as either raw-894->13 or NYU40->13."""
    import scipy.io as sio

    meta_path = resolve_meta_mat(
        meta_path,
        "NYU13 class mapping",
        alt_names=("class13Mapping.mat",),
    )
    m = sio.loadmat(meta_path)
    class_mapping = m.get("classMapping13")
    if class_mapping is not None and class_mapping.dtype.names and "labels13" in class_mapping.dtype.names:
        arr = np.array(class_mapping["labels13"][0, 0]).squeeze()
    else:
        arr = np.array(class_mapping if class_mapping is not None else m["mapping"]).squeeze()
    assert arr.shape[0] in (40, 894), f"expected 40 or 894, got {arr.shape}"
    return arr.astype(np.int32)


def build_lut13(map13: np.ndarray) -> tuple[np.ndarray, str]:
    """Return LUT plus source space used to compute NYU-13 labels."""
    map13 = np.asarray(map13).squeeze()
    if map13.shape[0] == 894:
        return np.concatenate([[0], map13]).astype(np.int32), "raw"
    if map13.shape[0] == 40:
        return np.concatenate([[0], map13]).astype(np.int32), "nyu40"
    raise AssertionError(f"expected 40 or 894, got {map13.shape}")


def extract(root: Path, mat_path: Path, splits_path: Path) -> None:
    import h5py
    import scipy.io as sio
    from PIL import Image

    (root / "rgb").mkdir(parents=True, exist_ok=True)
    (root / "depth").mkdir(parents=True, exist_ok=True)
    (root / "depth_raw").mkdir(parents=True, exist_ok=True)
    (root / "labels40").mkdir(parents=True, exist_ok=True)
    (root / "labels13").mkdir(parents=True, exist_ok=True)
    (root / "splits").mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir(parents=True, exist_ok=True)

    map40 = load_class_mapping_40(root / "meta" / "classMapping40.mat")
    map13 = load_class_mapping_13(root / "meta" / "classMapping13.mat")
    # ensure 0 stays 0 (unlabeled); toolbox vectors are 1-indexed from raw class.
    # Pre-pend 0 for label 0.
    lut40 = np.concatenate([[0], map40]).astype(np.int32)
    lut13, lut13_space = build_lut13(map13)

    print(f"open {mat_path}")
    with h5py.File(mat_path, "r") as h:
        imgs = h["images"]         # (N,3,W,H) uint8, transposed
        depths = h["depths"]       # (N,W,H) float32, meters (in-painted)
        raw = h["rawDepths"]       # (N,W,H) float32
        labels = h["labels"]       # (N,W,H) uint16 raw class ids
        n = imgs.shape[0]
        print(f"N={n}")

        for i in tqdm(range(n)):
            idx = i + 1
            rgb = np.transpose(imgs[i, ...], (2, 1, 0))  # (H,W,3)
            d = np.transpose(depths[i, ...], (1, 0))      # (H,W)
            dr = np.transpose(raw[i, ...], (1, 0))        # (H,W)
            lab = np.transpose(labels[i, ...], (1, 0)).astype(np.int32)  # (H,W)
            lab40 = lut40[lab].astype(np.uint8)
            lab13 = (lut13[lab] if lut13_space == "raw" else lut13[lab40]).astype(np.uint8)

            Image.fromarray(rgb).save(root / "rgb" / f"{idx:04d}.png")
            np.save(root / "depth" / f"{idx:04d}.npy", d.astype(np.float32))
            np.save(root / "depth_raw" / f"{idx:04d}.npy", dr.astype(np.float32))
            Image.fromarray(lab40).save(root / "labels40" / f"{idx:04d}.png")
            Image.fromarray(lab13).save(root / "labels13" / f"{idx:04d}.png")

    print(f"open {splits_path}")
    s = sio.loadmat(splits_path)
    train_ids = sorted(int(x) for x in np.array(s["trainNdxs"]).squeeze().tolist())
    test_ids = sorted(int(x) for x in np.array(s["testNdxs"]).squeeze().tolist())
    assert len(train_ids) == 795, len(train_ids)
    assert len(test_ids) == 654, len(test_ids)
    with open(root / "splits" / "gupta_795_654.json", "w") as f:
        json.dump({"train": train_ids, "test": test_ids}, f)
    print(f"splits saved: train={len(train_ids)} test={len(test_ids)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/nyuv2"))
    ap.add_argument("--labeled-mat", type=Path, default=None,
                    help="path to nyu_depth_v2_labeled.mat; if missing, download")
    ap.add_argument("--splits-mat", type=Path, default=None,
                    help="path to splits.mat; if missing, download")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--extract-only", action="store_true")
    args = ap.parse_args()

    root: Path = args.root
    root.mkdir(parents=True, exist_ok=True)
    cache = root / "_downloads"
    mat = args.labeled_mat or (cache / "nyu_depth_v2_labeled.mat")
    spl = args.splits_mat or (cache / "splits.mat")

    if not args.skip_download and not args.extract_only:
        download(LABELED_URL, mat)
        download(SPLITS_URL, spl)

    if not mat.exists() or not spl.exists():
        sys.exit(f"missing {mat} or {spl}")

    extract(root, mat, spl)


if __name__ == "__main__":
    main()
