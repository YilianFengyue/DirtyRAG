from dirtyrag.evidence.graph import build_conflict_edges
from dirtyrag.evidence.schemas import DuplicateGroup, EvidenceCard
from dirtyrag.evidence.scoring import build_answer_clusters, make_candidate_decision
from dirtyrag.evidence.cards import heuristic_card_payload
from dirtyrag.schemas import Document


def test_evidence_scoring_prefers_independent_support_over_isolated_conflict() -> None:
    cards = [
        EvidenceCard(
            doc_id="D1",
            relevance="high",
            answer_candidate="3,559 people",
            normalized_answer="3559 people",
            confidence=0.8,
        ),
        EvidenceCard(
            doc_id="D2",
            relevance="high",
            answer_candidate="3,559 people",
            normalized_answer="3559 people",
            confidence=0.8,
        ),
        EvidenceCard(
            doc_id="D3",
            relevance="high",
            answer_candidate="10,000 people",
            normalized_answer="10000 people",
            confidence=0.8,
        ),
    ]
    groups = [
        DuplicateGroup(group_id="G1", doc_ids=["D1"], representative_doc_id="D1", answer_candidate="3,559 people"),
        DuplicateGroup(group_id="G2", doc_ids=["D2"], representative_doc_id="D2", answer_candidate="3,559 people"),
        DuplicateGroup(group_id="G3", doc_ids=["D3"], representative_doc_id="D3", answer_candidate="10,000 people"),
    ]

    edges = build_conflict_edges(cards, groups)
    clusters = build_answer_clusters(cards, groups, edges)
    decision = make_candidate_decision(clusters, edges)

    assert decision.mode == "answer"
    assert decision.answer == "3,559 people"
    assert decision.supporting_doc_ids == ["D1", "D2"]


def test_heuristic_card_payload_recovers_population_from_truncated_json_failure() -> None:
    doc = Document(
        doc_id="D1",
        text="As of the census of 2010, there were 3,559 people residing in the city.",
    )

    payload = heuristic_card_payload("What is the population of Broken Bow?", doc, "card_error")

    assert payload["relevance"] == "high"
    assert payload["answer_candidate"] == "3,559"
