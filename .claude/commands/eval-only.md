---
description: Re-run eval on existing predictions without re-inferring
argument-hint: <experiment_name>
---

Eval-only pass. Use when prediction PNGs already exist and you changed metrics code or want per-class breakdown.

```bash
env -u PYTHONPATH -u AMENT_PREFIX_PATH PYTHONNOUSERSITE=1 \
  ~/miniconda3/envs/mobile-sam/bin/python scripts/run_eval.py \
  --pred-dir outputs/predictions/$ARGUMENTS
```

Print the metrics table (`mIoU`, `mean class acc`, `pixel acc`) for the available prediction PNGs.
