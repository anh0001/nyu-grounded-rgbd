---
name: verification-loop
description: Verification workflow for the NYUv2 GroundingDINO + SAM RGB-D pipeline. Runs the right Python, smoke, and eval checks in repo-specific order.
origin: local
---

# Verification Loop

Use this skill after meaningful code changes, before reporting work complete, and before creating a PR.

## When to Use

- After editing Python source, configs, or scripts
- After refactors that might affect imports or runtime wiring
- After any change to metrics, prompts, models, pipeline logic, or experiment configs

## Environment

Always use the repo env and ROS-clean command prefix:

```bash
ENV=~/miniconda3/envs/mobile-sam
PREFIX='env -u PYTHONPATH -u AMENT_PREFIX_PATH PYTHONNOUSERSITE=1'
```

## Verification Order

### 1. Lint

```bash
$PREFIX $ENV/bin/python -m ruff check .
```

Stop on lint failures and report the first actionable errors.

### 2. Tests

```bash
$PREFIX $ENV/bin/pytest
```

Report failing test names and first assertion errors. Do not "fix" tests blindly.

### 3. Import and Syntax Sanity

Use this when edits touch Python entrypoints, import wiring, or packaging:

```bash
$PREFIX $ENV/bin/python -m compileall src scripts
```

### 4. Smoke Inference

Run this when edits touch any of:

- `src/pipeline/**`
- `src/models/**`
- `src/prompts/**`
- `configs/**`
- `scripts/run_infer.py`

Command:

```bash
$PREFIX $ENV/bin/python scripts/run_infer.py --config week1_baseline --limit 10
```

Verification:

- run completes without import failure or CUDA OOM
- `outputs/predictions/week1_baseline/` contains 10 prediction PNGs
- any frame-level failure is surfaced explicitly, not silently ignored

### 5. Eval Rerun

Run this when edits touch metrics or prediction semantics, including:

- `src/eval/**`
- `scripts/run_eval.py`
- `scripts/make_table.py`
- changes that alter label assignment or rasterization behavior

Command:

```bash
$PREFIX $ENV/bin/python scripts/run_eval.py --pred-dir outputs/predictions/week1_baseline
```

Report:

- `mIoU`, `pAcc`, `mAcc`
- whether deltas were expected
- any notable per-class regression if metrics changed materially

## Output Format

Produce a short report:

```text
VERIFICATION REPORT

Lint: PASS/FAIL
Tests: PASS/FAIL
Import sanity: PASS/FAIL/SKIPPED
Smoke: PASS/FAIL/SKIPPED
Eval: PASS/FAIL/SKIPPED

Overall: READY / NOT READY
```

## Repo-Specific Checks

- `0` remains ignore in metrics and label maps
- metric means do not include absent GT classes
- confusion accumulation remains integer-based
- pipeline order remains proposals -> dedup -> rasterize
- raw-depth validity still gates depth-driven class decisions
