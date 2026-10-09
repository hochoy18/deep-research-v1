"""LLM client wrapper (ticket #2).

Provides the single boundary every node uses to call the OpenAI chat completion
API. Tests `monkeypatch` ``openai.OpenAI`` (the underlying SDK that
``ChatOpenAI`` delegates to) to swap in a fake client without ever hitting the
real API.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from app.config import get_config


def get_chat_model() -> ChatOpenAI:
    """Build a ``ChatOpenAI`` from environment-driven config.

    Returns a fresh instance per call so tests can safely patch the constructor
    without polluting other callers.
    """
    cfg = get_config()
    return ChatOpenAI(
        model=cfg.llm_model,
        api_key=cfg.openai_api_key,
        base_url=cfg.openai_base_url,
    )