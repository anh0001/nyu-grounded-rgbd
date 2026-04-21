---
name: experiment-runner
description: Launch long-running NYUv2 experiment inference with GPU preflight, ROS-clean env, streamed progress, and post-run eval plus table refresh. Use when the user wants a full run, long experiment, or unattended benchmark execution.
origin: local
---

# Experiment Runner

Use this skill for long-running inference or benchmark jobs in this repo.

## Environment

Always use the repo env and ROS-clean prefix:

```bash
ENV=~/miniconda3/envs/mobile-sam
PREFIX='env -u PYTHONPATH -u AMENT_PREFIX_PATH PYTHONNOUSERSITE=1'
```

Never use system Python.

## Scope

This skill is for:

- full inference over an experiment config
- long smoke or benchmark runs that may take minutes to hours
- post-run eval and aggregate table refresh

This skill is not for:

- changing metric semantics
- deleting partial outputs during debugging
- retrying blindly after CUDA OOM

## Preflight

Before launching:

1. Check GPU state with `nvidia-smi`.
2. Require the target GPU to be reasonably idle before a full run.
3. Confirm the config exists at `configs/experiment/<name>.yaml` unless the user gave a different explicit path or config family.
4. Confirm NYUv2 data is present by checking for extracted RGB frames under `data/nyuv2/`.
5. If prior predictions or reports exist for the same experiment, note that they exist before starting.

If the GPU is busy or inputs are missing, stop and report the blocker instead of starting a doomed run.

## Launch Strategy

Prefer a PTY-backed long-running command so progress can be streamed and revisited.

Inference command:

```bash
$PREFIX $ENV/bin/python scripts/run_infer.py --config <name>
```

Codex-specific guidance:

- start long runs in a TTY session when possible
- poll output through the existing session instead of blind sleep loops
- if the user explicitly wants a detached run, launch it intentionally and report the log path and PID
- do not pretend a detached run is still being actively monitored unless you are actually checking it

## Post-Run Steps

After inference succeeds:

```bash
$PREFIX $ENV/bin/python scripts/run_eval.py --pred-dir outputs/predictions/<name>
$PREFIX $ENV/bin/python scripts/make_table.py
```

Report:

- wall time
- `mIoU`
- `pAcc`
- `mAcc`
- report path, typically under `outputs/reports/`

If a prior report exists, compute and state deltas explicitly instead of implying improvement from the refreshed table alone.

## Failure Handling

- On CUDA OOM: stop, capture the error, and suggest the lightest credible next step
- On per-frame runtime failures: surface them explicitly
- On partial output directories: preserve them unless the user asks otherwise
- On eval failure after successful inference: keep predictions and report eval as a separate failure

## Repo Guardrails

- keep label maps in `{0..40}`
- preserve `0` as ignore
- do not present metric changes caused by formula changes as model improvements
- do not reorder the pipeline structure without explicitly calling out benchmark impact

## Output Format

Use a concise report:

```text
EXPERIMENT REPORT

Config: <name>
Preflight: PASS/FAIL
Inference: PASS/FAIL
Eval: PASS/FAIL/SKIPPED
Table refresh: PASS/FAIL/SKIPPED

Primary metric: mIoU = <value or skipped>
Artifacts: <pred dir>, <report path>
Notes: <GPU status, wall time, deltas, blockers>
```
