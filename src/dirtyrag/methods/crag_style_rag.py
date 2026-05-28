from __future__ import annotations

from dirtyrag.methods.base import BaseMethod
from dirtyrag.methods.relevance_filter_rag import RelevanceFilterRAG, sum_int
from dirtyrag.prompts import crag_conservative_answer_prompt, crag_retrieval_eval_prompt
from dirtyrag.schemas import MethodResult, QAExample


class CRAGStyleRAG(BaseMethod):
    name = "crag_style_rag"

    def run(self, example: QAExample) -> MethodResult:
        eval_payload, eval_response = self.evaluate_retrieval(example)
        verdict = normalize_verdict(str(eval_payload.get("retrieval_verdict", "insufficient")))
        action = normalize_action(str(eval_payload.get("action", "filter_then_answer")))
        reason = str(eval_payload.get("reason", ""))

        selected_docs = list(example.documents)
        relevance_metadata = None
        answer_latency = 0.0
        answer_prompt_tokens = None
        answer_completion_tokens = None
        answer_total_tokens = None

        if action == "abstain" or verdict in {"incorrect", "insufficient"} and action != "filter_then_answer":
            answer = "unknown"
            raw_response = answer
        else:
            if action == "filter_then_answer" or verdict in {"incorrect", "insufficient"}:
                filter_method = RelevanceFilterRAG(self.llm, max_tokens=self.max_tokens)
                judgments = [
                    filter_method.judge_document(example.question, doc) for doc in example.documents
                ]
                selected_docs = [
                    doc
                    for doc, judgment in zip(example.documents, judgments, strict=True)
                    if int(judgment.get("relevance", 0)) >= 1
                ]
                relevance_metadata = judgments
                if not selected_docs:
                    answer = "unknown"
                    raw_response = answer
                else:
                    response = self.llm.chat(
                        crag_conservative_answer_prompt(
                            example.question,
                            selected_docs,
                            retrieval_verdict=verdict,
                            reason=reason,
                        ),
                        max_tokens=self.max_tokens,
                    )
                    answer = response.content
                    raw_response = response.content
                    answer_latency = response.latency_sec
                    answer_prompt_tokens = response.prompt_tokens
                    answer_completion_tokens = response.completion_tokens
                    answer_total_tokens = response.total_tokens
            else:
                response = self.llm.chat(
                    crag_conservative_answer_prompt(
                        example.question,
                        selected_docs,
                        retrieval_verdict=verdict,
                        reason=reason,
                    ),
                    max_tokens=self.max_tokens,
                )
                answer = response.content
                raw_response = response.content
                answer_latency = response.latency_sec
                answer_prompt_tokens = response.prompt_tokens
                answer_completion_tokens = response.completion_tokens
                answer_total_tokens = response.total_tokens

        filter_latency = sum(float(item.get("latency_sec") or 0) for item in relevance_metadata or [])
        filter_prompt_tokens = [item.get("prompt_tokens") for item in relevance_metadata or []]
        filter_completion_tokens = [item.get("completion_tokens") for item in relevance_metadata or []]
        filter_total_tokens = [item.get("total_tokens") for item in relevance_metadata or []]

        return MethodResult(
            qid=example.qid,
            method=self.name,
            answer=answer,
            raw_response=raw_response,
            used_doc_ids=[doc.doc_id for doc in selected_docs],
            latency_sec=(eval_response.latency_sec if eval_response else 0) + answer_latency + filter_latency,
            prompt_tokens=sum_int([eval_response.prompt_tokens, answer_prompt_tokens, *filter_prompt_tokens]),
            completion_tokens=sum_int(
                [eval_response.completion_tokens, answer_completion_tokens, *filter_completion_tokens]
            ),
            total_tokens=sum_int([eval_response.total_tokens, answer_total_tokens, *filter_total_tokens]),
            metadata={
                "retrieval_verdict": verdict,
                "action": action,
                "reason": reason,
                "filtered_doc_ids": [doc.doc_id for doc in selected_docs],
                "relevance_judgments": relevance_metadata or [],
            },
        )

    def evaluate_retrieval(self, example: QAExample):
        try:
            return self.llm.json_chat(
                crag_retrieval_eval_prompt(example.question, example.documents),
                max_tokens=180,
            )
        except Exception as exc:
            return {
                "retrieval_verdict": "insufficient",
                "action": "filter_then_answer",
                "reason": f"retrieval_eval_error: {exc}",
            }, None


def normalize_verdict(value: str) -> str:
    value = value.strip().lower()
    if value in {"correct", "ambiguous", "incorrect", "insufficient"}:
        return value
    return "insufficient"


def normalize_action(value: str) -> str:
    value = value.strip().lower()
    if value in {"answer", "filter_then_answer", "abstain"}:
        return value
    return "filter_then_answer"

