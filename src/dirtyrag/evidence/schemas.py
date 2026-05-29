from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EvidenceCard(BaseModel):
    doc_id: str
    relevance: str = "low"
    answer_candidate: str = "unknown"
    normalized_answer: str = "unknown"
    claim: str = ""
    temporal_status: str = "unknown"
    time_cue: str = "unknown"
    confidence: float = 0.0
    raw_quote: str = ""
    rationale: str = ""
    entity_explicitness: str = "unknown"
    answer_role: str = "primary"
    contamination_risk: float = 0.0
    domain_cues: list[str] = Field(default_factory=list)


class DuplicateGroup(BaseModel):
    group_id: str
    doc_ids: list[str]
    representative_doc_id: str
    answer_candidate: str


class ConflictEdge(BaseModel):
    src: str
    dst: str
    relation: str
    reason: str
    confidence: float = 1.0


class AnswerCluster(BaseModel):
    answer: str
    normalized_answer: str
    doc_ids: list[str]
    duplicate_group_ids: list[str]
    unique_support_count: int
    high_relevance_count: int
    explicit_support_count: int = 0
    missing_entity_count: int = 0
    primary_support_count: int = 0
    mean_contamination_risk: float = 0.0
    mean_confidence: float
    conflict_count: int
    score: float


class BoardDecision(BaseModel):
    mode: str
    answer: str
    supporting_doc_ids: list[str] = Field(default_factory=list)
    rejected_doc_ids: list[str] = Field(default_factory=list)
    reason: str = ""
    scores: dict[str, float] = Field(default_factory=dict)


class EvidenceBoard(BaseModel):
    qid: str
    question: str
    cards: list[EvidenceCard]
    duplicate_groups: list[DuplicateGroup]
    conflict_edges: list[ConflictEdge]
    answer_clusters: list[AnswerCluster]
    candidate_decision: BoardDecision
    verifier_decision: dict[str, Any] = Field(default_factory=dict)
    final_answer: str
    decision_trace: list[str] = Field(default_factory=list)
