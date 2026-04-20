---
description: Full 654-frame inference + eval + table refresh for an experiment
argument-hint: <experiment_name>
---

Full benchmark run. Long (minutes to hours). Confirm GPU free before starting.

Experiment: $ARGUMENTS

Steps:

1. `nvidia-smi` — verify a GPU idle.
2. Inference:
   ```bash
   env -u PYTHONPATH -u AMENT_PREFIX_PATH PYTHONNOUSERSITE=1 \
     ~/miniconda3/envs/mobile-sam/bin/python scripts/run_infer.py --config $ARGUMENTS
   ```
3. Eval:
   ```bash
   ~/miniconda3/envs/mobile-sam/bin/python scripts/run_eval.py \
     --pred-dir outputs/predictions/$ARGUMENTS
   ```
4. Refresh leaderboard: `~/miniconda3/envs/mobile-sam/bin/python scripts/make_table.py`.

Report: mIoU / pAcc / mAcc + delta vs previous runs in the table. Flag any per-class IoU that regressed >2pp.
