from __future__ import annotations

from abc import ABC, abstractmethod

from dirtyrag.llm_client import LLMClient
from dirtyrag.schemas import MethodResult, QAExample


class BaseMethod(ABC):
    name: str

    def __init__(self, llm: LLMClient, *, max_tokens: int = 512) -> None:
        self.llm = llm
        self.max_tokens = max_tokens

    @abstractmethod
    def run(self, example: QAExample) -> MethodResult:
        raise NotImplementedError

