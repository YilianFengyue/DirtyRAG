from __future__ import annotations

from collections import defaultdict

from dirtyrag.evidence.normalize import jaccard, text_fingerprint
from dirtyrag.evidence.schemas import DuplicateGroup, EvidenceCard
from dirtyrag.schemas import Document


def build_duplicate_groups(cards: list[EvidenceCard], documents: list[Document]) -> list[DuplicateGroup]:
    doc_text = {doc.doc_id: doc.text for doc in documents}
    groups: list[list[EvidenceCard]] = []
    fingerprints: dict[str, set[str]] = {
        card.doc_id: text_fingerprint(doc_text.get(card.doc_id, "")) for card in cards
    }

    for card in cards:
        placed = False
        for group in groups:
            representative = group[0]
            same_answer = card.normalized_answer == representative.normalized_answer
            similar_text = jaccard(
                fingerprints.get(card.doc_id, set()),
                fingerprints.get(representative.doc_id, set()),
            ) >= 0.82
            if same_answer and similar_text:
                group.append(card)
                placed = True
                break
        if not placed:
            groups.append([card])

    duplicate_groups = []
    for idx, group in enumerate(groups, start=1):
        duplicate_groups.append(
            DuplicateGroup(
                group_id=f"G{idx}",
                doc_ids=[card.doc_id for card in group],
                representative_doc_id=group[0].doc_id,
                answer_candidate=group[0].answer_candidate,
            )
        )
    return duplicate_groups


def group_id_by_doc(duplicate_groups: list[DuplicateGroup]) -> dict[str, str]:
    mapping = {}
    for group in duplicate_groups:
        for doc_id in group.doc_ids:
            mapping[doc_id] = group.group_id
    return mapping


def cards_by_answer(cards: list[EvidenceCard]) -> dict[str, list[EvidenceCard]]:
    grouped: dict[str, list[EvidenceCard]] = defaultdict(list)
    for card in cards:
        if card.normalized_answer != "unknown":
            grouped[card.normalized_answer].append(card)
    return dict(grouped)

