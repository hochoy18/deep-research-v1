"""Research node — turn research_topics into search_results (ticket #5).

For each topic, fires one web-search call via the ``web_search`` tool and
collects the results.

The hard guardrail is ``RECURSION_LIMIT`` declared here. The graph layer wires
``recursion_limit`` into ``RunnableConfig`` (per LangGraph docs); we assert
it's set on entry so a misconfigured caller fails fast (ADR-0008).
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from app.models import SearchResult
from app.state import State
from app.tools import web_search as search_module

# Hard guardrail: never visit this node more than this many times per thread.
# LangGraph's checkpointer + recursion_limit will stop runaway loops before
# this matters in practice, but we surface it as a constant for documentation.
RECURSION_LIMIT = 25


def _format_query(topic) -> str:
    return f"{topic.title} — {topic.description}"


def research_run(state: State, config: RunnableConfig) -> dict:
    """Run one web search per topic and collect the results into ``search_results``.

    v0.3 — issue #26: the searches run on the user's final list
    (``confirmed_research_plan``, written by ``confirm_plan_gate``)
    when present, falling back to the LLM draft ``research_topics``
    only when the gate has not run (ADR-0005 §6 — the Research stage
    must reflect the user's confirmed intent, never the pre-gate
    draft, once the gate has produced a final list).
    """
    topics = state.get("confirmed_research_plan") or state.get("research_topics") or []
    if not topics:
        raise RuntimeError(
            "Research node: state has no confirmed_research_plan "
            "and no research_topics"
        )
    if config is None or "recursion_limit" not in config:
        raise RuntimeError(
            "Research node: caller must pass a RunnableConfig with "
            "'recursion_limit' set (see app.graph.build_graph)"
        )

    results: list[SearchResult] = []
    for topic in topics:
        payload = search_module.get_web_search().invoke(
            {"query": _format_query(topic), "topic_id": topic.id}
        )
        # ``web_search`` returns one SearchResult per Bailian page (typically ~5).
        results.extend(SearchResult(**p) for p in payload)

    return {
        "search_results": results,
        "current_step": "write",
    }