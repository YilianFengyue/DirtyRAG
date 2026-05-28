from __future__ import annotations

from typing import Any

from dirtyrag.methods.base import BaseMethod
from dirtyrag.prompts import relevance_judge_prompt, vanilla_rag_prompt
from dirtyrag.schemas import Document, MethodResult, QAExample


class RelevanceFilterRAG(BaseMethod):
    name = "relevance_filter_rag"

    def run(self, example: QAExample) -> MethodResult:
        judgments = [self.judge_document(example.question, doc) for doc in example.documents]
        kept_docs = [
            doc
            for doc, judgment in zip(example.documents, judgments, strict=True)
            if int(judgment.get("relevance", 0)) >= 1
        ]
        if not kept_docs:
            kept_docs = []
            answer = "unknown"
            raw_response = answer
            latency_sec = sum(float(item.get("latency_sec") or 0) for item in judgments)
            prompt_tokens = sum_int(judgment.get("prompt_tokens") for judgment in judgments)
            completion_tokens = sum_int(judgment.get("completion_tokens") for judgment in judgments)
            total_tokens = sum_int(judgment.get("total_tokens") for judgment in judgments)
        else:
            response = self.llm.chat(
                vanilla_rag_prompt(example.question, kept_docs),
                max_tokens=self.max_tokens,
            )
            answer = response.content
            raw_response = response.content
            latency_sec = response.latency_sec + sum(
                float(item.get("latency_sec") or 0) for item in judgments
            )
            prompt_tokens = sum_int(
                [response.prompt_tokens, *[item.get("prompt_tokens") for item in judgments]]
            )
            completion_tokens = sum_int(
                [response.completion_tokens, *[item.get("completion_tokens") for item in judgments]]
            )
            total_tokens = sum_int(
                [response.total_tokens, *[item.get("total_tokens") for item in judgments]]
            )

        return MethodResult(
            qid=example.qid,
            method=self.name,
            answer=answer,
            raw_response=raw_response,
            used_doc_ids=[doc.doc_id for doc in kept_docs],
            latency_sec=latency_sec,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            metadata={
                "filtered_doc_ids": [doc.doc_id for doc in kept_docs],
                "dropped_doc_ids": [
                    doc.doc_id
                    for doc, judgment in zip(example.documents, judgments, strict=True)
                    if int(judgment.get("relevance", 0)) < 1
                ],
                "relevance_judgments": judgments,
            },
        )

    def judge_document(self, question: str, doc: Document) -> dict[str, Any]:
        try:
            payload, response = self.llm.json_chat(
                relevance_judge_prompt(question, doc),
                max_tokens=400,
            )
        except Exception as exc:
            payload = {"relevance": 1, "reason": f"judge_error: {exc}"}
            response = None
        relevance = payload.get("relevance", 1)
        try:
            relevance = int(relevance)
        except (TypeError, ValueError):
            relevance = 1
        relevance = max(0, min(2, relevance))
        return {
            "doc_id": doc.doc_id,
            "relevance": relevance,
            "reason": str(payload.get("reason", ""))[:500],
            "latency_sec": response.latency_sec if response is not None else None,
            "prompt_tokens": response.prompt_tokens if response is not None else None,
            "completion_tokens": response.completion_tokens if response is not None else None,
            "total_tokens": response.total_tokens if response is not None else None,
        }


def sum_int(values) -> int | None:
    present = [int(value) for value in values if value is not None]
    if not present:
        return None
    return sum(present)
