# `nyu-grounded-rgbd` Codex Instructions

This repo implements zero-shot RGB-D semantic segmentation on NYUv2 with GroundingDINO + SAM and depth-aware refinement. Treat this file as the Codex root instruction surface. Deeper domain rules already live under `.claude/` and remain the detailed source of truth.

## Environment

- Reuse the existing conda env: `ENV=~/miniconda3/envs/mobile-sam`
- Never use system Python for repo commands.
- ROS can pollute imports. Run repo Python commands with a clean env:

```bash
env -u PYTHONPATH -u AMENT_PREFIX_PATH PYTHONNOUSERSITE=1 $ENV/bin/python <script>
```

- Install deps with `$ENV/bin/pip install -r requirements/base.txt` and add `requirements/dev.txt` only when needed.

## Benchmark Priorities

- Primary benchmark: NYUv2 40-class, Gupta 654-test split
- Primary metric: `mIoU`
- Secondary metrics: `pAcc`, `mAcc`
- `0` is unlabeled / ignore and must never be treated as a valid class
- Single-frame only. No SAM2, no multi-view fusion, no SLAM, no supervised training

## Common Commands

```bash
ENV=~/miniconda3/envs/mobile-sam
PREFIX='env -u PYTHONPATH -u AMENT_PREFIX_PATH PYTHONNOUSERSITE=1'

# smoke run
$PREFIX $ENV/bin/python scripts/run_infer.py --config week1_baseline --limit 10

# full inference
$PREFIX $ENV/bin/python scripts/run_infer.py --config week1_baseline

# eval existing predictions
$PREFIX $ENV/bin/python scripts/run_eval.py --pred-dir outputs/predictions/week1_baseline

# refresh aggregate table
$PREFIX $ENV/bin/python scripts/make_table.py

# tests
$PREFIX $ENV/bin/pytest
```

## Editing Guardrails

### Metrics

- Preserve `ignore_index=0`
- `mIoU` averages only over classes present in GT
- `pAcc` uses only valid GT pixels
- Confusion accumulation must stay `int64`
- If metric semantics change, add or update tests instead of silently changing outputs

### Pipeline

- Preserve the high-level order: proposals -> dedup -> rasterize
- Do not reorder structural stages without expecting fresh predictions and metric deltas
- Keep output label maps within `{0..40}` with `0` as ignore

### Depth Refinement

- Raw-depth validity must gate geometry-driven decisions
- Do not let interpolated depth alone decide floor or ceiling classes
- Keep NYU intrinsics and filled-vs-raw depth semantics aligned with current loader behavior

## Deeper Repo Rules

Use these files when work touches the corresponding subsystem:

- `.claude/rules/algorithm.md`
- `.claude/rules/depth_refinement.md`
- `.claude/rules/nyu_metadata.md`
- `.claude/agents/metrics-auditor.md`
- `.claude/agents/experiment-runner.md`
- `.claude/commands/smoke.md`
- `.claude/commands/test.md`

## Codex Workflow

- Prefer repo-local skills when relevant: `verification-loop`, `eval-harness`
- Use the local Codex roles for exploration, review, and documentation verification
- When a task depends on PyTorch, Transformers, Hugging Face, SAM, or GroundingDINO behavior, verify against primary docs instead of relying on memory
