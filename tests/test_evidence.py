from dirtyrag.evidence.graph import build_conflict_edges
from dirtyrag.evidence.schemas import DuplicateGroup, EvidenceCard
from dirtyrag.evidence.scoring import build_answer_clusters, make_candidate_decision
from dirtyrag.evidence.cards import (
    calibrate_candidate_with_domain,
    estimate_contamination_risk,
    heuristic_card_payload,
    infer_sport_from_text,
)
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
    assert payload["answer_candidate"] == "3,559 people"


def test_scoring_penalizes_missing_entity_even_with_high_confidence() -> None:
    cards = [
        EvidenceCard(
            doc_id="D1",
            relevance="high",
            answer_candidate="3,559 people",
            normalized_answer="3559 people",
            confidence=0.95,
            temporal_status="outdated",
            entity_explicitness="explicit",
        ),
        EvidenceCard(
            doc_id="D3",
            relevance="high",
            answer_candidate="10,000 people",
            normalized_answer="10000 people",
            confidence=1.0,
            temporal_status="current",
            entity_explicitness="missing",
        ),
    ]
    groups = [
        DuplicateGroup(group_id="G1", doc_ids=["D1"], representative_doc_id="D1", answer_candidate="3,559 people"),
        DuplicateGroup(group_id="G2", doc_ids=["D3"], representative_doc_id="D3", answer_candidate="10,000 people"),
    ]

    edges = build_conflict_edges(cards, groups)
    clusters = build_answer_clusters(cards, groups, edges)
    decision = make_candidate_decision(clusters, edges)

    assert decision.mode == "answer"
    assert decision.answer == "3,559 people"


def test_sport_heuristic_prefers_primary_football_over_secondary_golf() -> None:
    text = (
        "Justin Thomas started all 14 games including the Orange Bowl victory. "
        "He had 125 passing yards, a passing touchdown, and 121 rushing yards. "
        "Outside of football, Thomas is also known for golf."
    )

    inferred = infer_sport_from_text(text)
    calibrated = calibrate_candidate_with_domain(
        "What sport is Justin Thomas associated with?",
        text,
        "golf",
    )
    risk = estimate_contamination_risk(
        "What sport is Justin Thomas associated with?",
        text,
        "golf",
        "secondary",
    )

    assert inferred == "American football"
    assert calibrated == "American football"
    assert risk >= 0.5


def test_sport_scoring_rejects_high_risk_golf_cluster() -> None:
    cards = [
        EvidenceCard(
            doc_id="D2",
            relevance="high",
            answer_candidate="golf",
            normalized_answer="golf",
            confidence=0.95,
            entity_explicitness="explicit",
            answer_role="secondary",
            contamination_risk=0.75,
        ),
        EvidenceCard(
            doc_id="D3",
            relevance="high",
            answer_candidate="golf",
            normalized_answer="golf",
            confidence=0.95,
            entity_explicitness="explicit",
            answer_role="primary",
            contamination_risk=0.65,
        ),
    ]
    groups = [
        DuplicateGroup(group_id="G1", doc_ids=["D2"], representative_doc_id="D2", answer_candidate="golf"),
        DuplicateGroup(group_id="G2", doc_ids=["D3"], representative_doc_id="D3", answer_candidate="golf"),
    ]

    clusters = build_answer_clusters(cards, groups, [])
    decision = make_candidate_decision(clusters, [])

    assert decision.mode == "conflict"
    assert "high-risk evidence" in decision.reason
