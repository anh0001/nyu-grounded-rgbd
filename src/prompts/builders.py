"""Build GroundingDINO prompt text from chunks.

GroundingDINO expects lowercase class names separated by periods.
Each alias becomes one "label candidate"; we keep a mapping from alias
string back to (class_id, canonical_name) so we can recover the NYUv2 class
for each detection box.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.prompts.alias_bank import PromptChunk


@dataclass
class BuiltPrompt:
    text: str                              # "chair. office chair. desk. table."
    alias_list: list[str]                  # deduped aliases in order
    alias_to_class: dict[str, int]         # alias -> class_id


def build_prompt(chunk: PromptChunk) -> BuiltPrompt:
    aliases: list[str] = []
    a2c: dict[str, int] = {}
    for c in chunk.classes:
        for a in c.aliases:
            a_low = a.strip().lower()
            if a_low not in a2c:
                aliases.append(a_low)
                a2c[a_low] = c.class_id
    text = ". ".join(aliases) + "."
    return BuiltPrompt(text=text, alias_list=aliases, alias_to_class=a2c)


def alias_for_label(label_text: str, a2c: dict[str, int]) -> int | None:
    """HF GroundingDINO returns a label string; match it back to a class_id.
    Label is usually a substring of one of our aliases (GroundingDINO may merge
    adjacent tokens). Prefer exact, else longest-containing, else token-overlap."""
    s = label_text.strip().lower()
    if not s:
        return None
    if s in a2c:
        return a2c[s]
    # longest alias that is contained in s, or s contained in alias
    best = None
    best_len = 0
    for a, cid in a2c.items():
        if a in s or s in a:
            if len(a) > best_len:
                best = cid
                best_len = len(a)
    if best is not None:
        return best
    # token overlap
    s_tok = set(s.split())
    for a, cid in a2c.items():
        if s_tok & set(a.split()):
            return cid
    return None
