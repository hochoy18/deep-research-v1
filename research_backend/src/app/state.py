"""Global LangGraph state — a TypedDict that every node reads and writes.

``total=False`` because the graph starts with just ``user_query`` and accumulates
the rest as it advances through the pipeline.
"""

from __future__ import annotations

from typing import Literal

from typing_extensions import TypedDict

from app.models import Outline, ResearchTopic, SearchResult

# Pipeline stages; mirrored by the LangGraph node names.
Step = Literal["plan", "research", "write", "final", "failed"]


class State(TypedDict, total=False):
    user_query: str

    # Populated by plan_decompose (ticket #3)
    research_topics: list[ResearchTopic]

    # Populated by research_run (ticket #5)
    search_results: list[SearchResult]

    # Populated by write_compose stage 1 (ticket #6)
    outline: Outline
    draft: str

    # Populated by write_compose stage 2 (ticket #7)
    report: str

    # Always updated by every node to reflect progress
    current_step: Step
