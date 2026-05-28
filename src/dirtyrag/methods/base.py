from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from dirtyrag.llm_client import LLMClient
from dirtyrag.schemas import MethodResult, QAExample


class BaseMethod(ABC):
    name: str

    def __init__(self, llm: LLMClient, *, max_tokens: int = 512, run_dir: Path | None = None) -> None:
        self.llm = llm
        self.max_tokens = max_tokens
        self.run_dir = run_dir

    @abstractmethod
    def run(self, example: QAExample) -> MethodResult:
        raise NotImplementedError
