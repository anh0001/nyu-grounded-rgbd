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
   env -u PYTHONPATH -u AMENT_PREFIX_PATH PYTHONNOUSERSITE=1 \
     ~/miniconda3/envs/mobile-sam/bin/python scripts/run_eval.py \
     --pred-dir outputs/predictions/$ARGUMENTS
   ```
4. Refresh leaderboard:
   ```bash
   env -u PYTHONPATH -u AMENT_PREFIX_PATH PYTHONNOUSERSITE=1 \
     ~/miniconda3/envs/mobile-sam/bin/python scripts/make_table.py
   ```

Report: mIoU / pAcc / mAcc and the path to `outputs/reports/$ARGUMENTS.json`. If a prior report exists, compare against it explicitly instead of assuming `make_table.py` computes deltas.
