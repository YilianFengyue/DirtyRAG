from __future__ import annotations

from dirtyrag.methods.base import BaseMethod
from dirtyrag.prompts import direct_prompt
from dirtyrag.schemas import MethodResult, QAExample


class DirectLLM(BaseMethod):
    name = "direct_llm"

    def run(self, example: QAExample) -> MethodResult:
        response = self.llm.chat(direct_prompt(example.question), max_tokens=self.max_tokens)
        return MethodResult(
            qid=example.qid,
            method=self.name,
            answer=response.content,
            raw_response=response.content,
            used_doc_ids=[],
            latency_sec=response.latency_sec,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
        )

