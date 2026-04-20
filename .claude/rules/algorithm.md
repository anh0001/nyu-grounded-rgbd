# Pipeline algorithm

<important if="editing src/pipeline/*">
Order matters. Do not reorder without regenerating predictions from scratch.
</important>

1. **Detect** — `src/models/gdino.py` runs GroundingDINO over prompt chunks per image. Chunks are grouped semantically to reduce open-vocab cross-class confusion, with period-separated aliases within each chunk.
2. **Segment** — `src/models/sam_wrapper.py` uses single-box prompts for SAM / HQ-SAM / MobileSAM and caches one image embedding per frame.
3. **Score proposals** — `src/pipeline/proposals.py`:
   `score = w_box·box_conf + w_mask·sam_quality + w_depth·depth_consistency + w_geo·class_prior`
4. **Dedup** — class-aware box NMS first, then mask NMS. Stricter IoU for same-class, softer cross-class.
5. **Rasterize** — `src/pipeline/semantic_fusion.py`, 3 ordered phases:
   - instance-like (counters, furniture, props)
   - structural (floor / wall / ceiling via depth + gravity)
   - residual fill — best-overlapping remaining candidate → geometric fallback (`region_fill.py`, SLIC).
6. **Depth refinement** — `src/pipeline/depth_features.py`. See `.claude/rules/depth_refinement.md`.

## Invariants

- Output label map: dense `H×W` integer array with values in `{0..40}`. In the current code path, `semantic_fusion.py` builds this as `uint8`.
- Predictions save as PNG under `outputs/predictions/<exp>/<frame>.png`.
- Aggregate run metrics save to `outputs/reports/<exp>.json`. The current inference script does not write per-frame timing/proposal JSON sidecars.
