"""Load NYUv2 alias banks and turn them into prompt chunks."""
from __future__ import annotations

from dataclasses import dataclass

from src.datasets.nyuv2_meta import load_nyu13_bank, load_nyu40_bank


@dataclass
class PromptClass:
    class_id: int          # 1..40 (or 1..13)
    name: str              # canonical class name
    aliases: list[str]     # phrases to feed GroundingDINO


@dataclass
class PromptChunk:
    name: str                      # chunk name (e.g. "structural")
    classes: list[PromptClass]     # classes in this chunk

    @property
    def class_ids(self) -> list[int]:
        return [c.class_id for c in self.classes]


def build_chunks(protocol: str = "nyu40") -> list[PromptChunk]:
    bank = load_nyu40_bank() if protocol == "nyu40" else load_nyu13_bank()
    by_id: dict[int, PromptClass] = {
        c["id"]: PromptClass(c["id"], c["name"], c["aliases"]) for c in bank["classes"]
    }
    if "chunks" in bank:
        return [
            PromptChunk(name=k, classes=[by_id[i] for i in v])
            for k, v in bank["chunks"].items()
        ]
    # no chunks defined (nyu13): one chunk of 5-6 per pass
    all_cls = list(by_id.values())
    chunks: list[PromptChunk] = []
    step = 6
    for i in range(0, len(all_cls), step):
        chunks.append(PromptChunk(name=f"chunk{i // step}", classes=all_cls[i:i + step]))
    return chunks


def flatten_aliases(chunk: PromptChunk) -> tuple[list[str], list[int]]:
    """Flatten chunk into (aliases, class_id_per_alias) for GroundingDINO."""
    phrases: list[str] = []
    ids: list[int] = []
    for c in chunk.classes:
        for a in c.aliases:
            phrases.append(a.strip())
            ids.append(c.class_id)
    return phrases, ids
