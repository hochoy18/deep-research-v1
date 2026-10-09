"""Shared pytest fixtures — mock every external boundary so tests never hit
the network.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

# Ensure config has some baseline env values before any module imports it.
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("DASHSCOPE_API_KEY", "test-dashscope-key")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")


@pytest.fixture(autouse=True)
def _reset_config_cache(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """``get_config`` is ``lru_cache``d; clear between tests so env patches stick.

    Also noop ``load_dotenv`` so tests never read the project's real
    ``.env`` — that would override ``setdefault`` values and (worse) leak
    real credentials into test assertions. ``test_config.py`` re-enables
    the call with its own ``monkeypatch.setattr`` to verify the contract.
    """
    from app import config
    from app.config import get_config

    monkeypatch.setattr(config, "load_dotenv", lambda **kwargs: None)
    get_config.cache_clear()
    yield
    get_config.cache_clear()


@pytest.fixture
def fake_chat_model():
    """Stub ChatOpenAI-like object.

    Tests push strings onto ``fake_chat_model.responses``; ``invoke`` pops the
    next one each call. If the list is empty, ``invoke`` returns an empty
    JSON object — that's the expected "no more scripted responses" signal.
    """

    class _StubMessage:
        def __init__(self, content: str) -> None:
            self.content = content

    class _StubChatModel:
        def __init__(self) -> None:
            self.responses: list[str] = []
            self.calls: list[list] = []
            self.structured_responses: list = []

        def invoke(self, messages, **_kwargs):
            self.calls.append(messages)
            content = self.responses.pop(0) if self.responses else "{}"
            return _StubMessage(content)

        def with_structured_output(self, schema, **kwargs):
            """Return a chainable whose ``.invoke`` yields the next structured payload."""
            outer = self

            class _StructuredChat:
                def invoke(self, messages, **_inner):
                    outer.calls.append(messages)
                    if outer.structured_responses:
                        payload = outer.structured_responses.pop(0)
                        if isinstance(payload, schema):
                            return payload
                        return schema(**payload)
                    return schema()

            return _StructuredChat()

    return _StubChatModel()


@pytest.fixture
def fake_search_tool(monkeypatch):
    """Patch ``get_web_search`` so research_run never hits the network."""

    from app.tools import web_search

    class _StubTool:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def invoke(self, payload, **_kwargs):
            self.calls.append(payload)
            topic_id = payload["topic_id"]
            return {
                "topic_id": topic_id,
                "url": f"https://example.com/{topic_id}",
                "title": f"Mock result for {topic_id}",
                "snippet": "A canned snippet.",
                "score": 0.9,
            }

    stub = _StubTool()
    monkeypatch.setattr(web_search, "get_web_search", lambda: stub)
    return stub


@pytest.fixture
def patched_chat_model(monkeypatch, fake_chat_model):
    """Replace ``get_chat_model`` with one that returns ``fake_chat_model``."""
    from app.tools import llm

    monkeypatch.setattr(llm, "get_chat_model", lambda: fake_chat_model)
    return fake_chat_model


@pytest.fixture
def memory_checkpointer():
    """In-memory checkpointer suitable for tests — no Redis required."""
    from langgraph.checkpoint.memory import MemorySaver

    return MemorySaver()