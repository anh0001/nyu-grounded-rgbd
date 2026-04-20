# nyu-grounded-rgbd

Zero-shot RGB-D semantic segmentation on NYUv2. Training-free. GroundingDINO + SAM + depth refinement.

## Environment

Reuse existing conda env. **Never install into system Python.**

```bash
ENV=~/miniconda3/envs/mobile-sam
```

<important if="running any python command">
ROS pollutes `PYTHONPATH`. Always launch with clean env or imports break:

```bash
env -u PYTHONPATH -u AMENT_PREFIX_PATH PYTHONNOUSERSITE=1 $ENV/bin/python <script>
```
</important>

Install deps: `$ENV/bin/pip install -r requirements/base.txt` (add `requirements/dev.txt` for test tooling).

## Commands

| Task | Command |
|---|---|
| Smoke test (10 imgs) | `$ENV/bin/python scripts/run_infer.py --config week1_baseline --limit 10` |
| Full inference | `$ENV/bin/python scripts/run_infer.py --config week1_baseline` |
| Eval | `$ENV/bin/python scripts/run_eval.py --pred-dir outputs/predictions/<name>` |
| Aggregate table | `$ENV/bin/python scripts/make_table.py` |
| Tests | `$ENV/bin/pytest` |
| Prepare data | `$ENV/bin/python scripts/prepare_nyuv2.py --root data/nyuv2` |

All with ROS-clean env prefix shown above.

## Layout

```
src/
  datasets/    NYUv2 loader, 40/13 class maps
  prompts/     alias bank, chunked prompts
  models/      gdino.py (HF), sam_wrapper.py (SAM/MobileSAM/HQ-SAM)
  pipeline/    proposals → mask_refine → depth_features → semantic_fusion → region_fill
  eval/        ConfusionAccumulator, mIoU / pAcc / mAcc
  utils/       cache, io
configs/{dataset,model,pipeline,experiment}/*.yaml   (OmegaConf)
scripts/       prepare_nyuv2, run_infer, run_eval, make_table
data/nyuv2/    RGB + depth + labels (generated)
data/prompts/  nyu40_aliases.json, nyu13_aliases.json
outputs/       predictions, logs, reports (gitignored)
```

## Conventions

- **Label 0 = ignore** (unlabeled). NYU40 ids 1-40, 1-indexed. Never treat 0 as valid class.
- **Metrics**: primary = mIoU (40-class, Gupta 654-test split). Secondary = pAcc, mAcc. 13-class = appendix only.
- **Intrinsics** (NYU Kinect v1, color-aligned): fx=518.86, fy=519.47, cx=325.58, cy=253.74.
- **Configs** = OmegaConf YAMLs. Experiment YAML composes dataset + model + pipeline refs.
- **Single-frame only** v1. No SAM2, no multi-view, no supervised training.
- **Prompt chunks**: 7 semantic groups (`structural`, `openings`, `large_furniture`, `small_furniture`, `appliances`, `small_props`, `counters`). Period-separated aliases.

## Extended rules (lazy-load)

@.claude/rules/algorithm.md
@.claude/rules/depth_refinement.md
@.claude/rules/nyu_metadata.md

## Out of scope

SAM2, multi-view fusion, Grounding DINO 1.5/1.6 Pro API, any supervised training, SLAM.
