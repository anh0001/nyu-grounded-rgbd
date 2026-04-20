---
description: Scaffold a new experiment YAML from an existing one
argument-hint: <source_exp> <new_exp>
---

Create `configs/experiment/$2.yaml` copying from `configs/experiment/$1.yaml`. Set `name: $2`. Leave dataset/model/pipeline refs intact — user tunes what they want.

Then:
1. Show the new YAML.
2. Ask user which fields to override (model stack, pipeline variant, limit, weights).
3. After edits: run `/smoke $2`.
