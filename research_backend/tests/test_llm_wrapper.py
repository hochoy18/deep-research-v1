"""Ticket #2 — LLM client wrapper."""

from __future__ import annotations

from unittest.mock import patch

import pytest


def test_get_chat_model_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key-XYZ")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("LLM_MODEL", "gpt-4.1-mini")

    # Import lazily so env patches are picked up.
    from app.config import get_config
    from app.tools.llm import get_chat_model

    get_config.cache_clear()

    captured: dict = {}

    class _StubChatOpenAI:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    with patch("app.tools.llm.ChatOpenAI", _StubChatOpenAI):
        model = get_chat_model()

    assert isinstance(model, _StubChatOpenAI)
    assert captured["api_key"] == "openai-key-XYZ"
    assert captured["base_url"] == "https://example.com/v1"
    assert captured["model"] == "gpt-4.1-mini"


def test_get_chat_model_defaults(monkeypatch) -> None:
    # Clear all related env so defaults surface.
    for key in ("OPENAI_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "any-key")

    from app.config import get_config
    from app.tools.llm import get_chat_model

    get_config.cache_clear()

    captured: dict = {}

    class _StubChatOpenAI:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    with patch("app.tools.llm.ChatOpenAI", _StubChatOpenAI):
        get_chat_model()

    assert captured["base_url"] == "https://api.openai.com/v1"
    assert captured["model"] == "gpt-4o"


def test_no_real_openai_call_during_construction(monkeypatch) -> None:
    """Sanity: building the wrapper must not hit any HTTP endpoint."""
    monkeypatch.setenv("OPENAI_API_KEY", "any-key")
    from app.config import get_config
    from app.tools.llm import get_chat_model

    get_config.cache_clear()

    # If the real ChatOpenAI ran, it would attempt a network connection in
    # some environments. We swap it with a sentinel that records construction
    # but rejects any non-construction call.
    class _Sentinel:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def __getattr__(self, name):
            raise AssertionError(f"Real ChatOpenAI.{name} was called")

    with patch("app.tools.llm.ChatOpenAI", _Sentinel):
        model = get_chat_model()
    assert model.kwargs["api_key"] == "any-key"


@pytest.mark.usefixtures("_reset_config_cache")
def test_get_chat_model_returns_new_instance(monkeypatch) -> None:
    """Each call returns a fresh ChatOpenAI so tests don't share state."""
    from unittest.mock import MagicMock

    monkeypatch.setenv("OPENAI_API_KEY", "any-key")
    from app.config import get_config
    from app.tools.llm import get_chat_model

    get_config.cache_clear()

    factory = MagicMock(side_effect=lambda **kwargs: object())
    with patch("app.tools.llm.ChatOpenAI", factory):
        a = get_chat_model()
        b = get_chat_model()
        assert a is not b
        assert factory.call_count == 2