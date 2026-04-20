"""Simple disk cache keyed by (stage, image_id, config_hash)."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any


def config_hash(cfg: dict) -> str:
    payload = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.md5(payload).hexdigest()[:10]


def cache_path(root: Path | str, stage: str, image_id: str, chash: str,
               ext: str = "pkl") -> Path:
    p = Path(root) / stage / image_id
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{chash}.{ext}"


def save_pkl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pkl(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)
