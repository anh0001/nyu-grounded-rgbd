# nyu-grounded-rgbd

Zero-shot RGB-D semantic segmentation on NYUv2 with GroundingDINO + SAM.
Training-free pipeline. 2D detection/segmentation + depth-aware refinement.

## Benchmark

- **Primary**: NYUv2 40-class, Gupta 795-train / 654-test split, **mIoU** primary, pixel-acc + mean-class-acc secondary.
- **Secondary (appendix)**: NYUv2 13-class.
- Single-frame only (no multi-view / SLAM).

## Stacks

| config | detector | segmentor |
|---|---|---|
| `gdino_tiny_mobilesam` (fast) | GroundingDINO-Tiny (HF) | MobileSAM |
| `gdino_tiny_sam` | GroundingDINO-Tiny (HF) | SAM ViT-H |
| `gdino_base_hqsam` (best local) | GroundingDINO-Base (HF) | HQ-SAM ViT-H |

Configure via `configs/experiment/*.yaml`.

## Install

Reuse existing env:

```bash
ENV=~/miniconda3/envs/mobile-sam
$ENV/bin/pip install -r requirements/base.txt  # or base+dev
```

Key deps: `torch>=2.1`, `transformers>=4.51`, `mobile_sam` / `segment_anything` / `segment_anything_hq`, `h5py`, `scikit-image`, `omegaconf`.

Always launch with clean env (ROS pollutes `PYTHONPATH`):

```bash
env -u PYTHONPATH -u AMENT_PREFIX_PATH PYTHONNOUSERSITE=1 $ENV/bin/python ...
```

## Prepare data

```bash
python scripts/prepare_nyuv2.py --root data/nyuv2
```

Downloads `nyu_depth_v2_labeled.mat` + `splits.mat`; extracts RGB / depth / raw-depth / labels40 / labels13 under `data/nyuv2/`. Needs `classMapping40.mat` plus the NYU-13 mapping under `data/nyuv2/meta/`; the upstream metadata repo names that file `class13Mapping.mat`, and `prepare_nyuv2.py` accepts either `class13Mapping.mat` or `classMapping13.mat` (source: [ankurhanda/nyuv2-meta-data](https://github.com/ankurhanda/nyuv2-meta-data)).

## Run inference + eval

```bash
python scripts/run_infer.py --config week1_baseline --limit 10   # smoke
python scripts/run_infer.py --config week1_baseline               # full 654
python scripts/run_eval.py --pred-dir outputs/predictions/week1_baseline
python scripts/make_table.py
```

## Layout

```
src/
  datasets/    NYUv2 loader + metadata (40/13 maps)
  prompts/     alias bank + chunk builders
  models/      GroundingDINO, SAM wrappers
  pipeline/    proposals -> mask_refine -> depth_features -> semantic_fusion -> region_fill
  eval/        mIoU / pixel-acc / mean-class-acc
  utils/       cache, io helpers
configs/       dataset + model + pipeline + experiment YAMLs
scripts/       prepare_nyuv2, run_infer, run_eval, make_table
data/prompts/  nyu40_aliases.json, nyu13_aliases.json
```

## Algorithm

1. **Detect** — GroundingDINO over 7 semantically-grouped prompt chunks (`structural`, `openings`, `large_furniture`, `small_furniture`, `appliances`, `small_props`, `counters`), period-separated aliases per chunk.
2. **Segment** — SAM / HQ-SAM / MobileSAM single-box prompts; cache image embedding per frame.
3. **Score** — `w_box·box + w_mask·quality + w_depth·consistency + w_geo·class_prior`.
4. **Dedup** — class-aware box NMS, then mask NMS (stricter same-class, softer cross-class).
5. **Rasterize** — 3 phases: instance-like → structural (floor/wall/ceiling via depth+gravity) → residual fill (best-overlapping candidate → geometric fallback).
6. **Depth refinement** — raw-depth validity, Sobel edges, back-projected normals, gravity-aligned `up_proj` for floor/ceiling discrimination. Invalid raw-depth downweights geometry.

## Notes

- NYU40 ids: 1-indexed, 0 = unlabeled/ignore.
- Intrinsics from NYU toolbox (color-aligned Kinect v1): fx=518.86, fy=519.47, cx=325.58, cy=253.74.
- Out of scope (v1): SAM2, multi-view fusion, Grounding DINO 1.5/1.6 Pro API, any supervised training.
