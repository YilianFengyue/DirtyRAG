from __future__ import annotations

import re
import string


def normalize_text(text: str) -> str:
    lowered = text.lower()
    lowered = lowered.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    no_punct = lowered.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", no_punct).strip()


def contains_answer(answer: str, candidates: list[str]) -> bool:
    normalized_answer = normalize_text(answer)
    return any(normalize_text(candidate) in normalized_answer for candidate in candidates if candidate)

