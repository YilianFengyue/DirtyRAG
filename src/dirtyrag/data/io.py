from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def write_jsonl(path: Path, rows: Iterable[BaseModel | dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            if isinstance(row, BaseModel):
                payload = row.model_dump(mode="json")
            else:
                payload = row
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    return count


def append_jsonl(path: Path, row: BaseModel | dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(row, BaseModel):
        payload = row.model_dump(mode="json")
    else:
        payload = row
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
