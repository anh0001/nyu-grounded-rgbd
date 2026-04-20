# NYUv2 metadata

## Splits

Gupta split: 795 train / 654 test. Use `splits.mat` from NYU toolbox. Our benchmark is **654-test** only.

## Class mappings

- `classMapping40.mat` + `classMapping13.mat` required under `data/nyuv2/meta/`.
- Source: [ankurhanda/nyuv2-meta-data](https://github.com/ankurhanda/nyuv2-meta-data).
- 40-class: primary benchmark. 13-class: appendix only.
- **Class 0 = unlabeled / ignore**. Never include in mIoU / pAcc numerator or denominator for a class.

## Class name → alias bank

- `data/prompts/nyu40_aliases.json` — each of 40 classes maps to list of surface forms (e.g., `sofa` → `["sofa", "couch", "settee"]`).
- Chunking in `src/prompts/alias_bank.py` groups classes into 7 semantic chunks before concatenating with periods.

## Raw files

`prepare_nyuv2.py` downloads `nyu_depth_v2_labeled.mat` (~2.8GB) and extracts:
- `rgb/<id>.png`
- `depth/<id>.png` (filled, uint16 mm)
- `raw_depth/<id>.png` (raw, uint16 mm, 0 = invalid)
- `labels40/<id>.png`
- `labels13/<id>.png`

IDs are 1-indexed to match `.mat` ordering.
