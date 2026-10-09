"""Configuration loaded from environment variables.

Per user story #14, we deliberately avoid `pydantic-settings` to keep the
dependency surface minimal. Every read goes through `os.getenv` and is wrapped
by `get_config()` which caches the result so tests can `monkeypatch.setenv`
*before* the first call.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Config:
    """Frozen view of the environment-driven config."""

    openai_api_key: str
    openai_base_url: str
    llm_model: str

    dashscope_api_key: str
    dashscope_base_url: str

    redis_url: str

    log_level: str


def _getenv(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise RuntimeError(
            f"Required environment variable {name!r} is not set. "
            "Copy .env.example to .env and fill in the values."
        )
    return value


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Read configuration from the environment.

    Cached for the lifetime of the process — tests should call
    ``get_config.cache_clear()`` after ``monkeypatch.setenv`` to pick up changes.
    """
    return Config(
        openai_api_key=_getenv("OPENAI_API_KEY", ""),
        openai_base_url=_getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        llm_model=_getenv("LLM_MODEL", "gpt-4o"),
        dashscope_api_key=_getenv("DASHSCOPE_API_KEY", ""),
        dashscope_base_url=_getenv(
            "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        redis_url=_getenv("REDIS_URL", "redis://localhost:6379"),
        log_level=_getenv("LOG_LEVEL", "INFO"),
    )