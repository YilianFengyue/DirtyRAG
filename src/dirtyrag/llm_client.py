from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class LLMResponse:
    content: str
    raw: dict[str, Any]
    latency_sec: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class LLMClient:
    def __init__(
        self,
        *,
        provider: str = "openai_compatible",
        model: str = "",
        base_url: str = "",
        api_key: str = "",
        temperature: float = 0,
        timeout: int = 60,
        retry: int = 3,
        cache_path: Path | None = None,
    ) -> None:
        self.provider = provider
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.temperature = temperature
        self.timeout = timeout
        self.retry = retry
        self.cache_path = cache_path
        self.cache: dict[str, dict[str, Any]] = {}
        if cache_path is not None and cache_path.exists():
            self._load_cache(cache_path)

    @classmethod
    def from_config(cls, config: dict[str, Any], *, cache_path: Path | None = None) -> "LLMClient":
        return cls(
            provider=str(config.get("provider", "openai_compatible")),
            model=str(config.get("model", "")),
            base_url=str(config.get("base_url", "")),
            api_key=str(config.get("api_key", "")),
            temperature=float(config.get("temperature", 0)),
            timeout=int(config.get("timeout", 60)),
            retry=int(config.get("retry", 3)),
            cache_path=cache_path,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float | None = None,
    ) -> LLMResponse:
        payload = {
            "provider": self.provider,
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
        }
        cache_key = self._cache_key(payload)
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            return LLMResponse(
                content=str(cached["content"]),
                raw=dict(cached.get("raw", {})),
                latency_sec=0.0,
                prompt_tokens=cached.get("prompt_tokens"),
                completion_tokens=cached.get("completion_tokens"),
                total_tokens=cached.get("total_tokens"),
            )

        if self.provider == "mock":
            response = self._mock_chat(messages)
        else:
            response = self._openai_chat(payload)

        self.cache[cache_key] = {
            "content": response.content,
            "raw": response.raw,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "total_tokens": response.total_tokens,
        }
        self._append_cache(cache_key, self.cache[cache_key])
        return response

    def json_chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
    ) -> tuple[dict[str, Any], LLMResponse]:
        response = self.chat(messages, max_tokens=max_tokens)
        return parse_json_object(response.content), response

    def _openai_chat(self, payload: dict[str, Any]) -> LLMResponse:
        if not self.api_key:
            raise RuntimeError("LLM_API_KEY is empty. Set it or use llm.provider=mock.")

        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        last_error: Exception | None = None
        for attempt in range(1, self.retry + 1):
            start = time.perf_counter()
            try:
                completion = client.chat.completions.create(
                    model=payload["model"],
                    messages=payload["messages"],
                    temperature=payload["temperature"],
                    max_tokens=payload["max_tokens"],
                )
                latency = time.perf_counter() - start
                message = completion.choices[0].message.content or ""
                usage = completion.usage
                return LLMResponse(
                    content=message.strip(),
                    raw=completion.model_dump(mode="json"),
                    latency_sec=latency,
                    prompt_tokens=getattr(usage, "prompt_tokens", None),
                    completion_tokens=getattr(usage, "completion_tokens", None),
                    total_tokens=getattr(usage, "total_tokens", None),
                )
            except Exception as exc:  # pragma: no cover - exercised by live API only
                last_error = exc
                if attempt < self.retry:
                    time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"LLM call failed after {self.retry} attempts: {last_error}")

    def _mock_chat(self, messages: list[dict[str, str]]) -> LLMResponse:
        start = time.perf_counter()
        joined = "\n".join(message.get("content", "") for message in messages)
        answer = mock_response_for_prompt(joined)
        return LLMResponse(
            content=answer,
            raw={"provider": "mock"},
            latency_sec=time.perf_counter() - start,
            prompt_tokens=len(joined.split()),
            completion_tokens=len(answer.split()),
            total_tokens=len(joined.split()) + len(answer.split()),
        )

    def _load_cache(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                self.cache[str(row["key"])] = dict(row["value"])

    def _append_cache(self, key: str, value: dict[str, Any]) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")

    @staticmethod
    def _cache_key(payload: dict[str, Any]) -> str:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            return json.loads(stripped[start : end + 1])
        raise


def mock_response_for_prompt(prompt: str) -> str:
    if "Extract one structured evidence card" in prompt:
        answer = "mock_answer"
        if "3,559" in prompt:
            answer = "3,559 people"
        elif "10,000" in prompt:
            answer = "10,000 people"
        return json.dumps(
            {
                "doc_id": "D1",
                "relevance": "high",
                "answer_candidate": answer,
                "claim": f"The document supports {answer}.",
                "temporal_status": "unknown",
                "time_cue": "unknown",
                "confidence": 0.8,
                "raw_quote": answer,
                "rationale": "mock evidence card",
            },
            ensure_ascii=False,
        )
    if "strict answer verifier" in prompt:
        final_answer = "mock_answer"
        if "Candidate answer:" in prompt:
            final_answer = prompt.split("Candidate answer:", 1)[1].split("Evidence board summary:", 1)[0].strip()
        return json.dumps(
            {
                "verdict": "supported",
                "final_answer": final_answer,
                "supporting_doc_ids": [],
                "rejected_doc_ids": [],
                "reason": "mock verifier",
            },
            ensure_ascii=False,
        )
    if "Output JSON only." in prompt and '"relevance"' in prompt:
        return json.dumps({"relevance": 2, "reason": "mock relevance"}, ensure_ascii=False)
    if "Output JSON only." in prompt and '"retrieval_verdict"' in prompt:
        return json.dumps(
            {
                "retrieval_verdict": "correct",
                "action": "answer",
                "reason": "mock retrieval evaluation",
            },
            ensure_ascii=False,
        )
    if "Final answer:" in prompt or "Answer:" in prompt:
        return "mock_answer"
    return "unknown"
