from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    doc_id: str
    text: str
    source_type: str | None = None
    source_answer: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class QAExample(BaseModel):
    qid: str
    dataset: str
    question: str
    documents: list[Document]
    gold_answers: list[str] = Field(default_factory=list)
    wrong_answers: list[str] = Field(default_factory=list)
    task_type: str = "conflict_qa"
    metadata: dict[str, Any] = Field(default_factory=dict)


class MethodResult(BaseModel):
    qid: str
    method: str
    answer: str
    raw_response: str
    used_doc_ids: list[str] = Field(default_factory=list)
    evidence_board_path: str | None = None
    latency_sec: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

