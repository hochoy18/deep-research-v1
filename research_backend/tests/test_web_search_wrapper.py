"""Ticket #4 — Web search tool wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def _make_fake_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def test_web_search_sends_enable_search_true(monkeypatch) -> None:
    """Tool must call the DashScope-compatible client with enable_search=True."""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-XYZ")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://example.com/compatible-mode/v1")

    from app.config import get_config
    from app.tools import web_search

    get_config.cache_clear()

    captured: dict = {}

    class _FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _make_fake_response(
                '{"url": "https://x.test/a", "title": "Hello", "snippet": "world", "score": 0.8}'
            )

    class _FakeClient:
        def __init__(self, **kwargs) -> None:
            captured["client_kwargs"] = kwargs
            self.chat = MagicMock()
            self.chat.completions = _FakeCompletions()

    with patch("app.tools.web_search.OpenAI", _FakeClient):
        result = web_search.web_search.invoke({"query": "what is X?", "topic_id": "topic-1"})

    assert captured["client_kwargs"]["base_url"] == "https://example.com/compatible-mode/v1"
    assert captured["client_kwargs"]["api_key"] == "dashscope-XYZ"
    assert captured["extra_body"] == {"enable_search": True}
    assert captured["model"]  # some default model is set
    # Response parsed to SearchResult dict
    assert result["topic_id"] == "topic-1"
    assert result["url"] == "https://x.test/a"
    assert result["title"] == "Hello"
    assert result["snippet"] == "world"
    assert result["score"] == 0.8


def test_web_search_strips_markdown_fences(monkeypatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-XYZ")

    from app.config import get_config
    from app.tools import web_search

    get_config.cache_clear()

    class _FakeCompletions:
        def create(self, **kwargs):
            return _make_fake_response(
                "```json\n"
                '{"url": "https://x.test/b", "title": "Fenced", "snippet": "ok", "score": 0.5}\n'
                "```"
            )

    class _FakeClient:
        def __init__(self, **kwargs) -> None:
            self.chat = MagicMock()
            self.chat.completions = _FakeCompletions()

    with patch("app.tools.web_search.OpenAI", _FakeClient):
        result = web_search.web_search.invoke({"query": "q", "topic_id": "topic-2"})

    assert result["title"] == "Fenced"
    assert result["topic_id"] == "topic-2"


def test_web_search_no_real_call_during_import(monkeypatch) -> None:
    """Importing the module must not instantiate an OpenAI client."""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "any")

    class _BoomClient:
        def __init__(self, **kwargs) -> None:
            raise AssertionError("OpenAI client must not be instantiated at import time")

    with patch("app.tools.web_search.OpenAI", _BoomClient):
        # Reloading triggers re-import; should not call OpenAI.
        import importlib

        from app.tools import web_search as mod

        importlib.reload(mod)