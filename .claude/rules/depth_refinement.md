# Depth refinement

`src/pipeline/depth_features.py` computes per-pixel signals from raw + filled depth.

## Signals

- **Raw-depth validity mask** — Kinect v1 missing returns 0. Downweight geometry where invalid.
- **Sobel edges on depth** — mask boundary alignment prior.
- **Back-projected normals** — from filled depth + intrinsics (fx=518.86, fy=519.47, cx=325.58, cy=253.74). Cross-product of image-plane gradients in camera coords.
- **Gravity-aligned `up_proj`** — dot product of normal with the repo's fixed camera-up vector (`[0, -1, 0]`). Discriminates floor vs ceiling vs wall.

## Usage

- Floor: `up_proj > +τ` AND lower image region.
- Ceiling: `up_proj < -τ` AND upper image region.
- Wall: `|up_proj| < τ` AND vertical.

<important if="tuning thresholds">
Raw-depth invalid regions must NOT drive floor/ceiling assignment. Gate all geometric verdicts on `raw_valid`. Otherwise hallucinated normals from fill-interpolation bleed into class decisions.
</important>

## Gotchas

- Depth is in meters (float32), not mm. `h5py` loader casts on load.
- NYU labeled-set depth is filled (Colorization inpainting). Use `raw_depth` for validity, `depth` for geometry.
- The current implementation does not estimate gravity per frame; it uses the fixed camera-up assumption in `compute_features()`.
