from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path) -> dict[str, Any]:
    load_env_files()
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return expand_env(config)


def load_env_files() -> None:
    # Prefer the real local env file. Fall back to .env.example only to unblock
    # local experiments when a key was put there by mistake.
    for name in (".env", ".env.example"):
        env_path = PROJECT_ROOT / name
        if env_path.exists():
            load_dotenv(env_path, override=False, encoding="utf-8")


def expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    if isinstance(value, str):
        return ENV_PATTERN.sub(lambda match: os.getenv(match.group(1), ""), value)
    return value
