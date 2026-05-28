from __future__ import annotations

from typing import Any

from dirtyrag.evidence.normalize import normalize_answer_candidate
from dirtyrag.evidence.schemas import EvidenceCard
from dirtyrag.llm_client import LLMClient
from dirtyrag.prompts import evidence_card_prompt
from dirtyrag.schemas import Document


def extract_evidence_card(llm: LLMClient, question: str, doc: Document) -> tuple[EvidenceCard, dict[str, Any]]:
    try:
        payload, response = llm.json_chat(evidence_card_prompt(question, doc), max_tokens=900)
    except Exception as exc:
        payload = heuristic_card_payload(question, doc, f"card_error: {exc}")
        response = None
    card = normalize_card_payload(doc.doc_id, payload)
    usage = {
        "latency_sec": response.latency_sec if response is not None else 0.0,
        "prompt_tokens": response.prompt_tokens if response is not None else None,
        "completion_tokens": response.completion_tokens if response is not None else None,
        "total_tokens": response.total_tokens if response is not None else None,
    }
    return card, usage


def normalize_card_payload(doc_id: str, payload: dict[str, Any]) -> EvidenceCard:
    relevance = str(payload.get("relevance", "low")).strip().lower()
    if relevance not in {"high", "medium", "low"}:
        relevance = "low"
    confidence = payload.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    answer = str(payload.get("answer_candidate", "unknown")).strip() or "unknown"
    return EvidenceCard(
        doc_id=doc_id,
        relevance=relevance,
        answer_candidate=answer,
        normalized_answer=normalize_answer_candidate(answer),
        claim=str(payload.get("claim", ""))[:800],
        temporal_status=normalize_temporal(str(payload.get("temporal_status", "unknown"))),
        time_cue=str(payload.get("time_cue", "unknown"))[:200],
        confidence=confidence,
        raw_quote=str(payload.get("raw_quote", ""))[:500],
        rationale=str(payload.get("rationale", ""))[:500],
    )


def normalize_temporal(value: str) -> str:
    value = value.strip().lower()
    if value in {"current", "outdated", "unknown"}:
        return value
    return "unknown"


def fallback_payload(doc_id: str, reason: str) -> dict[str, Any]:
    return {
        "doc_id": doc_id,
        "relevance": "low",
        "answer_candidate": "unknown",
        "claim": "",
        "temporal_status": "unknown",
        "time_cue": "unknown",
        "confidence": 0.0,
        "raw_quote": "",
        "rationale": reason,
    }


def heuristic_card_payload(question: str, doc: Document, reason: str) -> dict[str, Any]:
    text = doc.text
    candidate = extract_candidate_from_text(question, text)
    relevance = "high" if candidate != "unknown" else "low"
    confidence = 0.55 if candidate != "unknown" else 0.0
    return {
        "doc_id": doc.doc_id,
        "relevance": relevance,
        "answer_candidate": candidate,
        "claim": f"The document may support the answer '{candidate}'." if candidate != "unknown" else "",
        "temporal_status": "unknown",
        "time_cue": extract_time_cue(text),
        "confidence": confidence,
        "raw_quote": candidate if candidate != "unknown" else "",
        "rationale": reason + "; heuristic fallback used",
    }


def extract_candidate_from_text(question: str, text: str) -> str:
    import re

    question_lower = question.lower()
    if "population" in question_lower:
        matches = re.findall(r"\b\d{1,3}(?:,\d{3})+\b|\b\d{4,7}\b", text)
        if matches:
            return normalize_population_candidate(matches)
    if "born" in question_lower or "birth" in question_lower:
        month_pattern = (
            r"\b(?:January|February|March|April|May|June|July|August|September|October|"
            r"November|December)\s+\d{1,2},\s+\d{4}\b"
        )
        match = re.search(month_pattern, text)
        if match:
            return match.group(0)
        year_match = re.search(r"\b(?:born|birth)[^\.\n]{0,80}?(\d{4})\b", text, flags=re.IGNORECASE)
        if year_match:
            return year_match.group(1)
    quoted = re.search(r"\banswer\s+(?:is|:)\s*([A-Za-z0-9,\- ]{1,80})", text, flags=re.IGNORECASE)
    if quoted:
        return quoted.group(1).strip()
    return "unknown"


def normalize_population_candidate(matches: list[str]) -> str:
    cleaned = []
    for match in matches:
        numeric = int(match.replace(",", ""))
        if numeric >= 1000:
            cleaned.append((numeric, match))
    if not cleaned:
        return matches[0]
    # Prefer explicitly comma-formatted population figures over years.
    comma_matches = [match for _, match in cleaned if "," in match]
    return comma_matches[0] if comma_matches else cleaned[0][1]


def extract_time_cue(text: str) -> str:
    import re

    match = re.search(r"\b(19|20)\d{2}\b", text)
    return match.group(0) if match else "unknown"
