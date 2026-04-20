# Depth refinement

`src/pipeline/depth_features.py` computes per-pixel signals from raw + filled depth.

## Signals

- **Raw-depth validity mask** — Kinect v1 missing returns 0. Downweight geometry where invalid.
- **Sobel edges on depth** — mask boundary alignment prior.
- **Back-projected normals** — from filled depth + intrinsics (fx=518.86, fy=519.47, cx=325.58, cy=253.74). Cross-product of image-plane gradients in camera coords.
- **Gravity-aligned `up_proj`** — dot product of normal with estimated up vector. Discriminates floor vs ceiling vs wall.

## Usage

- Floor: `up_proj > +τ` AND low height.
- Ceiling: `up_proj > +τ` AND high height.
- Wall: `|up_proj| < τ` AND vertical.

<important if="tuning thresholds">
Raw-depth invalid regions must NOT drive floor/ceiling assignment. Gate all geometric verdicts on `raw_valid`. Otherwise hallucinated normals from fill-interpolation bleed into class decisions.
</important>

## Gotchas

- Depth is in meters (float32), not mm. `h5py` loader casts on load.
- NYU labeled-set depth is filled (Colorization inpainting). Use `raw_depth` for validity, `depth` for geometry.
- Gravity estimate per-frame is sensitive to clutter; fall back to image-column axis if normal PCA fails.
