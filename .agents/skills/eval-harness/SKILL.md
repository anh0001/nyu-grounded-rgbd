---
name: eval-harness
description: Eval-driven benchmark workflow for NYUv2 RGB-D experiments, including smoke regression, metric regression, and benchmark comparison.
origin: local
---

# Eval Harness

Use this skill when defining success criteria for experiment changes or when comparing behavior before and after a code or config change.

## Environment

Always run repo commands with the local env and ROS-clean prefix:

```bash
ENV=~/miniconda3/envs/mobile-sam
PREFIX='env -u PYTHONPATH -u AMENT_PREFIX_PATH PYTHONNOUSERSITE=1'
```

## Evaluation Ladder

### 1. Capability Check

Goal: confirm the script or config runs at all.

Typical command:

```bash
$PREFIX $ENV/bin/python scripts/run_infer.py --config week1_baseline --limit 1
```

Pass criteria:

- config resolves correctly
- imports succeed
- one frame completes without runtime failure

### 2. Smoke Regression

Goal: confirm the end-to-end pipeline still runs on a small slice.

Command:

```bash
$PREFIX $ENV/bin/python scripts/run_infer.py --config week1_baseline --limit 10
```

Pass criteria:

- no import errors
- no CUDA OOM
- 10 prediction PNGs are written
- no silent per-frame failures

### 3. Metric Regression

Goal: confirm reported metrics are unchanged unless intentionally changed.

Command:

```bash
$PREFIX $ENV/bin/python scripts/run_eval.py --pred-dir outputs/predictions/week1_baseline
```

Report:

- `mIoU`
- `pAcc`
- `mAcc`
- explanation for any expected delta

Use this level for changes in:

- `src/eval/**`
- `scripts/run_eval.py`
- `scripts/make_table.py`
- prediction semantics that affect labels

### 4. Benchmark Comparison

Goal: compare a full experiment against an earlier run when a real benchmark change is intended.

Commands:

```bash
$PREFIX $ENV/bin/python scripts/run_infer.py --config <experiment_name>
$PREFIX $ENV/bin/python scripts/run_eval.py --pred-dir outputs/predictions/<experiment_name>
$PREFIX $ENV/bin/python scripts/make_table.py
```

Report:

- wall time
- `mIoU`, `pAcc`, `mAcc`
- delta versus prior baseline
- notable per-class deltas when metrics materially move

## Standard Comparison Rules

- Primary benchmark is NYUv2 40-class Gupta 654-test split
- `mIoU` is the primary success metric
- `0` is ignore and excluded from valid-class metrics
- If numbers move because formulas changed, call that out explicitly instead of presenting the change as a model improvement

## Default Artifacts

Use repo-native artifacts and paths:

- predictions under `outputs/predictions/<experiment_name>/`
- evaluation reports under `outputs/reports/`
- aggregate tables from `scripts/make_table.py`

## Example Use

For a pipeline change:

1. run capability check
2. run smoke regression
3. rerun metrics if prediction semantics changed
4. run full benchmark only if the smoke result is stable and the change is intended to affect benchmark outcomes

## Output Format

Produce a concise report:

```text
EVAL REPORT

Capability: PASS/FAIL
Smoke: PASS/FAIL
Metrics: PASS/FAIL/SKIPPED
Benchmark: PASS/FAIL/SKIPPED

Primary metric: mIoU = <value or skipped>
Notes: <expected delta / unexpected regression / no change>
```
