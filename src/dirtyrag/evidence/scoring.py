from __future__ import annotations

from statistics import mean

from dirtyrag.evidence.clustering import cards_by_answer, group_id_by_doc
from dirtyrag.evidence.normalize import relevance_score
from dirtyrag.evidence.schemas import AnswerCluster, BoardDecision, ConflictEdge, DuplicateGroup, EvidenceCard


def build_answer_clusters(
    cards: list[EvidenceCard],
    duplicate_groups: list[DuplicateGroup],
    conflict_edges: list[ConflictEdge],
) -> list[AnswerCluster]:
    doc_to_group = group_id_by_doc(duplicate_groups)
    grouped = cards_by_answer(cards)
    clusters = []
    for normalized_answer, answer_cards in grouped.items():
        doc_ids = [card.doc_id for card in answer_cards]
        duplicate_group_ids = sorted({doc_to_group.get(card.doc_id, card.doc_id) for card in answer_cards})
        high_relevance_count = sum(1 for card in answer_cards if relevance_score(card.relevance) == 2)
        explicit_support_count = sum(1 for card in answer_cards if card.entity_explicitness == "explicit")
        missing_entity_count = sum(1 for card in answer_cards if card.entity_explicitness == "missing")
        conflict_count = sum(
            1
            for edge in conflict_edges
            if edge.relation == "contradict" and (edge.src in doc_ids or edge.dst in doc_ids)
        )
        mean_confidence = mean([card.confidence for card in answer_cards]) if answer_cards else 0.0
        outdated_count = sum(1 for card in answer_cards if card.temporal_status == "outdated")
        unique_support_count = len(duplicate_group_ids)
        score = (
            unique_support_count * 1.15
            + explicit_support_count * 0.7
            + high_relevance_count * 0.25
            + mean_confidence * 0.25
            - missing_entity_count * 0.35
            - conflict_count * 0.12
            - outdated_count * 0.05
        )
        clusters.append(
            AnswerCluster(
                answer=answer_cards[0].answer_candidate,
                normalized_answer=normalized_answer,
                doc_ids=doc_ids,
                duplicate_group_ids=duplicate_group_ids,
                unique_support_count=unique_support_count,
                high_relevance_count=high_relevance_count,
                explicit_support_count=explicit_support_count,
                missing_entity_count=missing_entity_count,
                mean_confidence=round(mean_confidence, 4),
                conflict_count=conflict_count,
                score=round(score, 4),
            )
        )
    return sorted(clusters, key=lambda item: item.score, reverse=True)


def make_candidate_decision(
    clusters: list[AnswerCluster],
    conflict_edges: list[ConflictEdge],
) -> BoardDecision:
    if not clusters:
        return BoardDecision(mode="unknown", answer="unknown", reason="No non-unknown answer cluster was found.")

    best = clusters[0]
    second = clusters[1] if len(clusters) > 1 else None
    scores = {cluster.answer: cluster.score for cluster in clusters}
    rejected_doc_ids = sorted(
        {
            doc_id
            for edge in conflict_edges
            if edge.relation == "contradict"
            for doc_id in (edge.src, edge.dst)
            if doc_id not in best.doc_ids
        }
    )

    if second is None:
        return BoardDecision(
            mode="answer",
            answer=best.answer,
            supporting_doc_ids=best.doc_ids,
            rejected_doc_ids=rejected_doc_ids,
            reason="Only one supported answer cluster was found.",
            scores=scores,
        )

    margin = best.score - second.score
    if best.unique_support_count >= 2 and margin >= 0.2:
        return BoardDecision(
            mode="answer",
            answer=best.answer,
            supporting_doc_ids=best.doc_ids,
            rejected_doc_ids=rejected_doc_ids,
            reason=(
                "The top answer has stronger independent support than competing "
                "conflicting clusters."
            ),
            scores=scores,
        )
    if best.explicit_support_count > second.explicit_support_count and margin >= 0.05:
        return BoardDecision(
            mode="answer",
            answer=best.answer,
            supporting_doc_ids=best.doc_ids,
            rejected_doc_ids=rejected_doc_ids,
            reason="The top answer has stronger explicit entity support than competing clusters.",
            scores=scores,
        )
    if margin < 0.35 and second.unique_support_count >= 1:
        return BoardDecision(
            mode="conflict",
            answer="conflict",
            supporting_doc_ids=best.doc_ids + second.doc_ids,
            rejected_doc_ids=[],
            reason="Multiple answer clusters have similar support.",
            scores=scores,
        )
    return BoardDecision(
        mode="answer",
        answer=best.answer,
        supporting_doc_ids=best.doc_ids,
        rejected_doc_ids=rejected_doc_ids,
        reason="The top answer cluster has the highest evidence score.",
        scores=scores,
    )
