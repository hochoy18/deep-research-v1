"""Web search tool wrapper (ticket #4).

Wraps DashScope's OpenAI-compatible chat completion endpoint with
``extra_body={"enable_search": True}`` as a ``DynamicStructuredTool``. This is
the single boundary every node uses to issue a web search — tests
``monkeypatch`` the underlying ``openai.OpenAI`` constructor and capture the
payload to assert it carried ``enable_search=True`` without hitting the real API.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import StructuredTool
from openai import OpenAI
from pydantic import BaseModel, Field

from app.config import get_config
from app.models import SearchResult

DEFAULT_SEARCH_MODEL = "qwen-plus"


class WebSearchInput(BaseModel):
    """Schema for the ``web_search`` tool's input."""

    query: str = Field(..., description="The search query")
    topic_id: str = Field(..., description="Owning ResearchTopic.id for grouping")


def _extract_message_content(response: Any) -> str:
    """Pull the text content out of a DashScope/OpenAI chat completion response."""
    try:
        return response.choices[0].message.content or ""
    except (AttributeError, IndexError, KeyError) as exc:
        raise RuntimeError(f"Unexpected search response shape: {response!r}") from exc


def _run_web_search(query: str, topic_id: str) -> dict:
    """Call DashScope with ``enable_search=True`` and parse the first message.

    Returns a dict matching ``SearchResult`` so ``DynamicStructuredTool`` can
    type-validate the tool output before returning to the agent.
    """
    cfg = get_config()
    client = OpenAI(api_key=cfg.dashscope_api_key, base_url=cfg.dashscope_base_url)

    response = client.chat.completions.create(
        model=DEFAULT_SEARCH_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a web search assistant. When the user asks a "
                    "question, run a web search and respond with a JSON object "
                    "of the form "
                    '{"url": "...", "title": "...", "snippet": "...", "score": 0.0}.'
                ),
            },
            {"role": "user", "content": query},
        ],
        extra_body={"enable_search": True},
    )

    raw = _extract_message_content(response).strip()

    # DashScope sometimes wraps the JSON in markdown code fences; strip them.
    if raw.startswith("```"):
        raw = raw.strip("`").removeprefix("json").strip()

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Search response was not valid JSON: {raw!r}"
        ) from exc

    payload.setdefault("score", None)
    payload["topic_id"] = topic_id

    # Validate through the Pydantic schema so downstream code can trust the shape.
    return SearchResult(**payload).model_dump()


# A single module-level instance accessed via ``get_web_search()``. We hide
# the instance behind a getter so tests can ``monkeypatch.setattr`` the
# module's ``get_web_search`` to return a stub — StructuredTool is a frozen
# pydantic model and ``invoke`` cannot be reassigned on the instance itself.
web_search = StructuredTool(
    name="web_search",
    description=(
        "Issue a web search via DashScope and return a single best-result "
        "SearchResult (url, title, snippet, score)."
    ),
    func=_run_web_search,
    args_schema=WebSearchInput,
)


def get_web_search() -> StructuredTool:
    """Return the module-level ``web_search`` tool.

    Tests swap this entire function for a stub.
    """
    return web_search