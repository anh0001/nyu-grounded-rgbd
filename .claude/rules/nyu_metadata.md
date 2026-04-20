# NYUv2 metadata

## Splits

Gupta split: 795 train / 654 test. Use `splits.mat` from NYU toolbox. Our benchmark is **654-test** only.

## Class mappings

- `classMapping40.mat` is required under `data/nyuv2/meta/`.
- The NYU-13 mapping is also required there; `prepare_nyuv2.py` accepts either `classMapping13.mat` or the upstream alias `class13Mapping.mat`.
- Source: [ankurhanda/nyuv2-meta-data](https://github.com/ankurhanda/nyuv2-meta-data).
- 40-class: primary benchmark. 13-class: appendix only.
- **Class 0 = unlabeled / ignore**. Never include in mIoU / pAcc numerator or denominator for a class.

## Class name → alias bank

- `src/datasets/nyuv2_meta.py` loads prompt banks from `data/prompts/nyu40_aliases.json` and `data/prompts/nyu13_aliases.json`.
- Chunking in `src/prompts/alias_bank.py` groups classes into 7 semantic chunks before concatenating with periods.

## Raw files

`scripts/prepare_nyuv2.py` downloads `nyu_depth_v2_labeled.mat` (~2.8GB) and extracts:
- `rgb/<id>.png`
- `depth/<id>.npy` (filled, float32 meters)
- `depth_raw/<id>.npy` (raw, float32, invalid = 0 or NaN)
- `labels40/<id>.png`
- `labels13/<id>.png`

IDs are 1-indexed to match `.mat` ordering.
