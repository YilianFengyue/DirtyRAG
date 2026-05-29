from __future__ import annotations

from dirtyrag.evidence.schemas import AnswerCluster, BoardDecision, ConflictEdge, EvidenceCard


def summarize_board_for_verifier(
    cards: list[EvidenceCard],
    clusters: list[AnswerCluster],
    edges: list[ConflictEdge],
    decision: BoardDecision,
) -> str:
    lines = ["Evidence cards:"]
    for card in cards:
        lines.append(
            f"- {card.doc_id}: relevance={card.relevance}, answer={card.answer_candidate}, "
            f"confidence={card.confidence}, entity={card.entity_explicitness}, "
            f"role={card.answer_role}, risk={card.contamination_risk}, "
            f"cues={card.domain_cues}, claim={card.claim}"
        )
    lines.append("\nAnswer clusters:")
    for cluster in clusters[:5]:
        lines.append(
            f"- answer={cluster.answer}, docs={cluster.doc_ids}, "
            f"unique_support={cluster.unique_support_count}, score={cluster.score}, "
            f"explicit_support={cluster.explicit_support_count}, "
            f"primary_support={cluster.primary_support_count}, "
            f"risk={cluster.mean_contamination_risk}, conflicts={cluster.conflict_count}"
        )
    lines.append("\nConflict edges:")
    for edge in edges[:10]:
        if edge.relation in {"contradict", "duplicate"}:
            lines.append(f"- {edge.src}->{edge.dst}: {edge.relation}, {edge.reason}")
    lines.append(
        f"\nCandidate decision: mode={decision.mode}, answer={decision.answer}, "
        f"supporting_docs={decision.supporting_doc_ids}, reason={decision.reason}"
    )
    return "\n".join(lines)
