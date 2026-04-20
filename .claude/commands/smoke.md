---
description: Run 10-image smoke inference on an experiment config
argument-hint: <experiment_name> (default week1_baseline)
---

Run smoke inference to verify pipeline runs end-to-end without OOM or import errors.

Experiment: $ARGUMENTS (fall back to `week1_baseline` if empty).

Execute:

```bash
env -u PYTHONPATH -u AMENT_PREFIX_PATH PYTHONNOUSERSITE=1 \
  ~/miniconda3/envs/mobile-sam/bin/python scripts/run_infer.py \
  --config ${ARGUMENTS:-week1_baseline} --limit 10
```

After completion:
- Report wall time.
- Confirm `outputs/predictions/<exp>/` contains 10 PNGs.
- If any frame errored, show stderr + suggest root cause. Do NOT silently retry.
