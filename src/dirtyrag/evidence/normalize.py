from __future__ import annotations

import re

from dirtyrag.evaluation.normalize import normalize_text


UNKNOWN_VALUES = {"", "unknown", "none", "n/a", "not mentioned", "not provided"}


def normalize_answer_candidate(answer: str) -> str:
    normalized = normalize_text(answer)
    if normalized in UNKNOWN_VALUES:
        return "unknown"
    return normalized


def relevance_score(relevance: str) -> int:
    value = relevance.strip().lower()
    if value == "high":
        return 2
    if value == "medium":
        return 1
    return 0


def text_fingerprint(text: str) -> set[str]:
    normalized = normalize_text(text)
    return set(re.findall(r"[a-z0-9]+", normalized))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

