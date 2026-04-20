# Codex Supplement

This file supplements the root `AGENTS.md` with Codex-specific behavior for this repo.

## Local Skills

Codex should discover and use these repo-local skills when relevant:

- `verification-loop` - run lint, tests, import sanity, smoke inference, and eval checks in the right order for this Python/CUDA pipeline
- `eval-harness` - define and compare capability, smoke, metric, and benchmark regressions for experiments

These skills are repo-specific rewrites. Do not fall back to generic Node or web-app verification patterns.

## Local Roles

The repo defines three Codex multi-agent roles:

- `explorer` - read-only codebase tracing and evidence gathering
- `reviewer` - correctness, regression, benchmark-integrity, and missing-test review
- `docs_researcher` - primary-doc verification for library and model behavior

Use these roles when the task benefits from separation between exploration, review, and documentation verification.

## Enforcement Model

This repo does not rely on Claude-style hooks for Codex.

- enforcement is instruction-based
- benchmark invariants come from `AGENTS.md` and `.claude/rules/*`
- verification should be done explicitly through the local `verification-loop` skill and targeted commands

## MCP Guidance

The repo-local Codex config keeps only a light MCP baseline:

- `github` for repository and PR context
- `context7` for current library and framework documentation

Do not assume heavier MCP servers are available in-repo.
