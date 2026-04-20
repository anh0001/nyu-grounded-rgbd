---
description: Run pytest suite with clean env
---

```bash
env -u PYTHONPATH -u AMENT_PREFIX_PATH PYTHONNOUSERSITE=1 \
  ~/miniconda3/envs/mobile-sam/bin/pytest
```

On fail: show failing test name + first assertion error. Do not auto-edit tests to make them pass — diagnose root cause first.
