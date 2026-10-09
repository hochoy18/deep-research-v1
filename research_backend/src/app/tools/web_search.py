"""Web search tool wrapper (ticket #4 / ADR-0010).

Wraps ``dashscope.Application.call`` against a Bailian single-agent
application (configured server-side with a web-search plugin). The app
returns a structured double-wrapped JSON::

    output.text = '{"result":{"content":[{"text":"<inner>"}]}}'
    <inner>     = '{"pages":[{"snippet","hostname","hostlogo","title","url"}, ...]}'

We project ``pages[0]`` into the canonical ``SearchResult``. Tests
``monkeypatch`` ``app.tools.web_search.dashscope.Application.call`` and
capture the kwargs to assert the configured ``app_id`` is used without
hitting the real DashScope endpoint.
"""

from __future__ import annotations

import json
from typing import Any

import dashscope
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.config import get_config
from app.models import SearchResult


class WebSearchInput(BaseModel):
    """Schema for the ``web_search`` tool's input."""

    query: str = Field(..., description="The search query")
    topic_id: str = Field(..., description="Owning ResearchTopic.id for grouping")


def _run_web_search(query: str, topic_id: str) -> dict:
    """Call the Bailian single-agent app and project ``pages[0]`` to ``SearchResult``.

    Returns a dict matching ``SearchResult`` so ``DynamicStructuredTool`` can
    type-validate the tool output before returning to the agent.
    """
    cfg = get_config()
    dashscope.api_key = cfg.dashscope_api_key

    response = dashscope.Application.call(
        app_id=cfg.dashscope_mcp_app_id,
        prompt=query,
    )

    outer_text: str | None = getattr(getattr(response, "output", None), "text", None)
    if outer_text is None:
        raise RuntimeError(f"Unexpected Bailian response shape: {response!r}")

    try:
        outer = json.loads(outer_text)
        inner = json.loads(outer["result"]["content"][0]["text"])
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"Unexpected Bailian response outer shape: {outer_text!r}"
        ) from exc

    pages: list[dict[str, Any]] = inner.get("pages") or []
    if not pages:
        raise RuntimeError(f"Bailian returned no pages: {outer_text!r}")

    page = pages[0]
    return SearchResult(
        topic_id=topic_id,
        url=page["url"],
        title=page["title"],
        snippet=page["snippet"],
    ).model_dump()


# A single module-level instance accessed via ``get_web_search()``. We hide
# the instance behind a getter so tests can ``monkeypatch.setattr`` the
# module's ``get_web_search`` to return a stub — StructuredTool is a frozen
# pydantic model and ``invoke`` cannot be reassigned on the instance itself.
web_search = StructuredTool(
    name="web_search",
    description=(
        "Issue a web search via the configured Bailian single-agent application "
        "and return the first page's url / title / snippet."
    ),
    func=_run_web_search,
    args_schema=WebSearchInput,
)


def get_web_search() -> StructuredTool:
    """Return the module-level ``web_search`` tool.

    Tests swap this entire function for a stub.
    """
    return web_search