from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dirtyrag.evidence.board import summarize_board_for_verifier
from dirtyrag.evidence.cards import extract_evidence_card
from dirtyrag.evidence.clustering import build_duplicate_groups
from dirtyrag.evidence.graph import build_conflict_edges
from dirtyrag.evidence.schemas import BoardDecision, EvidenceBoard
from dirtyrag.evidence.scoring import build_answer_clusters, make_candidate_decision
from dirtyrag.methods.base import BaseMethod
from dirtyrag.methods.relevance_filter_rag import sum_int
from dirtyrag.prompts import evidence_verifier_prompt
from dirtyrag.schemas import MethodResult, QAExample


class EvidenceBoardRAG(BaseMethod):
    name = "evidenceboard_rag"

    def run(self, example: QAExample) -> MethodResult:
        cards = []
        usages: list[dict[str, Any]] = []
        trace = []
        for doc in example.documents:
            card, usage = extract_evidence_card(self.llm, example.question, doc)
            cards.append(card)
            usages.append(usage)
        trace.append(f"Extracted {len(cards)} evidence cards.")

        duplicate_groups = build_duplicate_groups(cards, example.documents)
        trace.append(f"Built {len(duplicate_groups)} duplicate groups.")

        conflict_edges = build_conflict_edges(cards, duplicate_groups)
        trace.append(f"Built {len(conflict_edges)} evidence graph edges.")

        answer_clusters = build_answer_clusters(cards, duplicate_groups, conflict_edges)
        trace.append(f"Built {len(answer_clusters)} answer clusters.")

        candidate_decision = make_candidate_decision(answer_clusters, conflict_edges)
        trace.append(
            f"Candidate decision: {candidate_decision.mode} -> {candidate_decision.answer}."
        )

        verifier_payload, verifier_usage = self.verify(
            example.question,
            candidate_decision,
            cards,
            answer_clusters,
            conflict_edges,
        )
        final_answer = final_answer_from_verifier(candidate_decision, verifier_payload)
        trace.append(f"Verifier decision: {verifier_payload.get('verdict', 'unknown')} -> {final_answer}.")

        board = EvidenceBoard(
            qid=example.qid,
            question=example.question,
            cards=cards,
            duplicate_groups=duplicate_groups,
            conflict_edges=conflict_edges,
            answer_clusters=answer_clusters,
            candidate_decision=candidate_decision,
            verifier_decision=verifier_payload,
            final_answer=final_answer,
            decision_trace=trace,
        )
        board_path = self.save_board(board)

        all_usages = [*usages, verifier_usage]
        return MethodResult(
            qid=example.qid,
            method=self.name,
            answer=final_answer,
            raw_response=json.dumps(verifier_payload, ensure_ascii=False),
            used_doc_ids=supporting_doc_ids(candidate_decision, verifier_payload),
            evidence_board_path=str(board_path) if board_path is not None else None,
            latency_sec=sum(float(item.get("latency_sec") or 0) for item in all_usages),
            prompt_tokens=sum_int(item.get("prompt_tokens") for item in all_usages),
            completion_tokens=sum_int(item.get("completion_tokens") for item in all_usages),
            total_tokens=sum_int(item.get("total_tokens") for item in all_usages),
            metadata={
                "candidate_mode": candidate_decision.mode,
                "candidate_answer": candidate_decision.answer,
                "verifier_verdict": verifier_payload.get("verdict"),
                "supporting_doc_ids": supporting_doc_ids(candidate_decision, verifier_payload),
                "rejected_doc_ids": verifier_payload.get("rejected_doc_ids", candidate_decision.rejected_doc_ids),
                "num_cards": len(cards),
                "num_conflict_edges": len([e for e in conflict_edges if e.relation == "contradict"]),
                "num_answer_clusters": len(answer_clusters),
            },
        )

    def verify(
        self,
        question: str,
        candidate_decision: BoardDecision,
        cards,
        answer_clusters,
        conflict_edges,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if candidate_decision.mode in {"unknown", "conflict"}:
            payload = {
                "verdict": candidate_decision.mode,
                "final_answer": candidate_decision.answer,
                "supporting_doc_ids": candidate_decision.supporting_doc_ids,
                "rejected_doc_ids": candidate_decision.rejected_doc_ids,
                "reason": candidate_decision.reason,
            }
            return payload, {
                "latency_sec": 0.0,
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            }

        board_summary = summarize_board_for_verifier(
            cards,
            answer_clusters,
            conflict_edges,
            candidate_decision,
        )
        try:
            payload, response = self.llm.json_chat(
                evidence_verifier_prompt(question, candidate_decision.answer, board_summary),
                max_tokens=600,
            )
            payload = normalize_verifier_payload(payload, candidate_decision)
            return payload, {
                "latency_sec": response.latency_sec,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "total_tokens": response.total_tokens,
            }
        except Exception as exc:
            return {
                "verdict": "supported",
                "final_answer": candidate_decision.answer,
                "supporting_doc_ids": candidate_decision.supporting_doc_ids,
                "rejected_doc_ids": candidate_decision.rejected_doc_ids,
                "reason": f"verifier_error_fallback: {exc}",
            }, {
                "latency_sec": 0.0,
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            }

    def save_board(self, board: EvidenceBoard) -> Path | None:
        if self.run_dir is None:
            return None
        board_dir = self.run_dir / "evidence_boards"
        board_dir.mkdir(parents=True, exist_ok=True)
        board_path = board_dir / f"{board.qid}_{self.name}.json"
        board_path.write_text(
            json.dumps(board.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return board_path


def normalize_verifier_payload(payload: dict[str, Any], candidate_decision: BoardDecision) -> dict[str, Any]:
    verdict = str(payload.get("verdict", "supported")).strip().lower()
    if verdict not in {"supported", "revise", "conflict", "unknown"}:
        verdict = "supported"
    final_answer = str(payload.get("final_answer", "")).strip()
    if not final_answer:
        final_answer = candidate_decision.answer
    supporting_doc_ids = payload.get("supporting_doc_ids")
    if not isinstance(supporting_doc_ids, list):
        supporting_doc_ids = candidate_decision.supporting_doc_ids
    rejected_doc_ids = payload.get("rejected_doc_ids")
    if not isinstance(rejected_doc_ids, list):
        rejected_doc_ids = candidate_decision.rejected_doc_ids
    return {
        "verdict": verdict,
        "final_answer": final_answer,
        "supporting_doc_ids": [str(item) for item in supporting_doc_ids],
        "rejected_doc_ids": [str(item) for item in rejected_doc_ids],
        "reason": str(payload.get("reason", ""))[:800],
    }


def final_answer_from_verifier(candidate_decision: BoardDecision, verifier_payload: dict[str, Any]) -> str:
    verdict = verifier_payload.get("verdict")
    if verdict in {"supported", "revise"}:
        return str(verifier_payload.get("final_answer") or candidate_decision.answer)
    if verdict == "conflict":
        return "conflict"
    if verdict == "unknown":
        return "unknown"
    return candidate_decision.answer


def supporting_doc_ids(candidate_decision: BoardDecision, verifier_payload: dict[str, Any]) -> list[str]:
    doc_ids = verifier_payload.get("supporting_doc_ids")
    if isinstance(doc_ids, list) and doc_ids:
        return [str(item) for item in doc_ids]
    return candidate_decision.supporting_doc_ids
