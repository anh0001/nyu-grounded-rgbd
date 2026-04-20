# Pipeline algorithm

<important if="editing src/pipeline/*">
Order matters. Do not reorder without regenerating predictions from scratch.
</important>

1. **Detect** — `models/gdino.py` runs GroundingDINO over 7 prompt chunks per image. Chunks grouped semantically to avoid cross-class confusion in open-vocab scoring. Period-separated aliases within each chunk.
2. **Segment** — `models/sam_wrapper.py` single-box prompts → SAM / HQ-SAM / MobileSAM. Cache image embedding per frame (one forward pass of image encoder per image).
3. **Score proposals** — `pipeline/proposals.py`:
   `score = w_box·box_conf + w_mask·sam_quality + w_depth·depth_consistency + w_geo·class_prior`
4. **Dedup** — class-aware box NMS first, then mask NMS. Stricter IoU for same-class, softer cross-class.
5. **Rasterize** — `pipeline/semantic_fusion.py`, 3 ordered phases:
   - instance-like (counters, furniture, props)
   - structural (floor / wall / ceiling via depth + gravity)
   - residual fill — best-overlapping remaining candidate → geometric fallback (`region_fill.py`, SLIC).
6. **Depth refinement** — `pipeline/depth_features.py`. See @.claude/rules/depth_refinement.md.

## Invariants

- Output label map: `H×W int32`, values in `{0..40}`. 0 = ignore.
- Predictions save as PNG (palette-free int32 → encoded as uint16 PNG) under `outputs/predictions/<exp>/<frame>.png`.
- Per-frame report JSON written alongside, tracks chunk timings + proposal counts.
