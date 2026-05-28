from __future__ import annotations

from itertools import combinations

from dirtyrag.evidence.normalize import relevance_score
from dirtyrag.evidence.schemas import ConflictEdge, DuplicateGroup, EvidenceCard


def build_conflict_edges(
    cards: list[EvidenceCard],
    duplicate_groups: list[DuplicateGroup],
) -> list[ConflictEdge]:
    duplicate_pairs = set()
    for group in duplicate_groups:
        if len(group.doc_ids) > 1:
            for src, dst in combinations(group.doc_ids, 2):
                duplicate_pairs.add(tuple(sorted((src, dst))))

    edges: list[ConflictEdge] = []
    for src, dst in combinations(cards, 2):
        pair = tuple(sorted((src.doc_id, dst.doc_id)))
        if pair in duplicate_pairs:
            edges.append(
                ConflictEdge(
                    src=src.doc_id,
                    dst=dst.doc_id,
                    relation="duplicate",
                    reason="The two documents support the same answer with highly similar text.",
                    confidence=0.95,
                )
            )
            continue
        if src.normalized_answer == "unknown" or dst.normalized_answer == "unknown":
            continue
        if src.normalized_answer == dst.normalized_answer:
            edges.append(
                ConflictEdge(
                    src=src.doc_id,
                    dst=dst.doc_id,
                    relation="support",
                    reason="The two documents support the same answer candidate.",
                    confidence=0.9,
                )
            )
            continue
        if relevance_score(src.relevance) >= 1 and relevance_score(dst.relevance) >= 1:
            edges.append(
                ConflictEdge(
                    src=src.doc_id,
                    dst=dst.doc_id,
                    relation="contradict",
                    reason=(
                        f"{src.doc_id} supports '{src.answer_candidate}', "
                        f"while {dst.doc_id} supports '{dst.answer_candidate}'."
                    ),
                    confidence=0.85,
                )
            )
    return edges

