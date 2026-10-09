"""Ticket #4 / ADR-0010 — Web search tool wrapper.

Wraps ``dashscope.Application.call`` (a Bailian single-agent application
with a web-search plugin) and projects every page of its structured
output into the canonical ``SearchResult``. Tests ``monkeypatch`` the
module-level ``dashscope.Application.call`` so no real network call
happens, then assert the ``app_id`` comes from config and the nested
``output.text`` JSON is parsed correctly.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


def _make_fake_response(pages: list[dict]) -> MagicMock:
    """Build a stub ``dashscope.Application.call`` return value.

    The real Bailian response wraps the agent's JSON twice:
      output.text = '{"result":{"content":[{"text":"<inner>"}]}}'
      <inner>     = '{"pages":[{"url":...,"title":...,"snippet":...}]}'
    """
    inner = json.dumps({"pages": pages})
    outer = json.dumps({"result": {"content": [{"text": inner}]}})
    resp = MagicMock()
    resp.output = MagicMock()
    resp.output.text = outer
    return resp


def test_web_search_calls_bailian_app_with_configured_app_id(monkeypatch) -> None:
    """Tool must call ``dashscope.Application.call`` with the configured app_id."""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-XYZ")
    monkeypatch.setenv("DASHSCOPE_MCP_APP_ID", "app-abc")

    from app.config import get_config
    from app.tools import web_search

    get_config.cache_clear()

    captured: dict = {}

    def _fake_call(**kwargs):
        captured.update(kwargs)
        return _make_fake_response(
            [
                {
                    "url": "https://x.test/a",
                    "title": "Hello",
                    "snippet": "world",
                    "hostname": "x.test",
                    "hostlogo": "",
                }
            ]
        )

    with patch("app.tools.web_search.dashscope.Application.call", _fake_call):
        result = web_search.web_search.invoke({"query": "what is X?", "topic_id": "topic-1"})

    assert captured["app_id"] == "app-abc"
    assert captured["prompt"] == "what is X?"
    # Tool returns one SearchResult-shaped dict per Bailian page.
    assert isinstance(result, list)
    assert len(result) == 1
    page = result[0]
    assert page["topic_id"] == "topic-1"
    assert page["url"] == "https://x.test/a"
    assert page["title"] == "Hello"
    assert page["snippet"] == "world"
    assert page["score"] is None


def test_web_search_dashscope_api_key_is_set_from_config(monkeypatch) -> None:
    """The tool must seed ``dashscope.api_key`` from the configured DashScope key."""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-XYZ")
    monkeypatch.setenv("DASHSCOPE_MCP_APP_ID", "app-abc")

    from app.config import get_config
    from app.tools import web_search

    get_config.cache_clear()

    def _fake_call(**kwargs):
        return _make_fake_response(
            [{"url": "https://x.test/a", "title": "t", "snippet": "s", "hostname": "x", "hostlogo": ""}]
        )

    with patch("app.tools.web_search.dashscope.Application.call", _fake_call):
        web_search.web_search.invoke({"query": "q", "topic_id": "topic-2"})

    assert web_search.dashscope.api_key == "dashscope-XYZ"


def test_web_search_no_real_call_during_import(monkeypatch) -> None:
    """Importing the module must not invoke ``dashscope.Application.call``."""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "any")
    monkeypatch.setenv("DASHSCOPE_MCP_APP_ID", "any")

    def _boom_call(**kwargs):
        raise AssertionError(
            "dashscope.Application.call must not be invoked at import time"
        )

    with patch("app.tools.web_search.dashscope.Application.call", _boom_call):
        # Reloading triggers re-import; should not call Application.call.
        import importlib

        from app.tools import web_search as mod

        importlib.reload(mod)