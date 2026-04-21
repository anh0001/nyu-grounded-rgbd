"""NYUv2 label metadata: class names, 40→13 mapping, prompt-bank loader."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = REPO_ROOT / "data" / "prompts"

NYU40_NAMES: list[str] = [
    "unlabeled",
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
    "window", "bookshelf", "picture", "counter", "blinds", "desk", "shelves",
    "curtain", "dresser", "pillow", "mirror", "floor mat", "clothes",
    "ceiling", "books", "refrigerator", "television", "paper", "towel",
    "shower curtain", "box", "whiteboard", "person", "night stand", "toilet",
    "sink", "lamp", "bathtub", "bag", "otherstructure", "otherfurniture",
    "otherprop",
]
assert len(NYU40_NAMES) == 41  # 0..40

NYU13_NAMES: list[str] = [
    "unlabeled",
    "bed", "books", "ceiling", "chair", "floor", "furniture", "objects",
    "picture", "sofa", "table", "tv", "wall", "window",
]
assert len(NYU13_NAMES) == 14


def load_nyu40_bank(filename: str = "nyu40_aliases.json") -> dict[str, Any]:
    path = PROMPTS_DIR / filename
    with open(path) as f:
        return json.load(f)


def load_nyu40_descriptions(filename: str = "nyu40_descriptions.json") -> dict[int, str]:
    path = PROMPTS_DIR / filename
    with open(path) as f:
        raw = json.load(f)
    return {int(k): str(v) for k, v in raw.items()}


def load_nyu13_bank() -> dict[str, Any]:
    with open(PROMPTS_DIR / "nyu13_aliases.json") as f:
        return json.load(f)


def nyu40_to_nyu13_lut() -> list[int]:
    """Return LUT[0..40] → 0..13 (0 = unlabeled)."""
    bank = load_nyu13_bank()
    lut = [0] * 41
    for k, v in bank["map_from_nyu40"].items():
        lut[int(k)] = int(v)
    return lut


def classes_in_chunk(chunk_name: str) -> list[int]:
    bank = load_nyu40_bank()
    return bank["chunks"][chunk_name]


def chunk_names() -> list[str]:
    return list(load_nyu40_bank()["chunks"].keys())
