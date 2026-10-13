"""Global LangGraph state — a TypedDict that every node reads and writes.

``total=False`` because the graph starts with just ``user_query`` and accumulates
the rest as it advances through the pipeline.

Subgraph IO contracts (v0.2 — issue #10):
- ``PlanInput`` / ``PlanOutput`` — the plan subgraph's IO (wraps v0.1 ``plan_decompose``)
- ``ResearchInput`` / ``ResearchOutput`` — the research subgraph's IO (wraps v0.1 ``research_run``)
- ``WriteInput`` / ``WriteOutput`` — the write subgraph's IO (wraps v0.1 ``write_compose``)

The global ``State`` is the union of the subgraph outputs plus ``user_query``
(the pipeline's only true input). Each subgraph declares its own subset so the
stage boundaries are typed.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from app.models import (
    Outline,
    PlanQuestion,
    PlanUserAnswer,
    ResearchTopic,
    SearchResult,
)

# Pipeline stages; mirrored by the LangGraph node names.
Step = Literal["plan", "research", "write", "final", "failed"]


class State(TypedDict, total=False):
    """Union of every subgraph's IO fields — the main graph's state."""

    user_query: str

    # Populated by plan_decompose (ticket #3) via plan_subgraph (issue #10)
    research_topics: list[ResearchTopic]

    # Populated by research_run (ticket #5) via research_subgraph (issue #10)
    search_results: list[SearchResult]

    # Populated by write_compose stage 1 (ticket #6) via write_subgraph (issue #10)
    outline: Outline
    draft: str

    # Populated by write_compose stage 2 (ticket #7) via write_subgraph (issue #10)
    report: str

    # Always updated by every node to reflect progress
    current_step: Step


# ─────────────────────────────────────────────────────────────────────────────
# Subgraph IO contracts (v0.2 — issue #10)
# Each is a strict subset of ``State`` so the main graph can compose subgraphs
# without leaking one stage's internals into another.
# ─────────────────────────────────────────────────────────────────────────────


class PlanInput(TypedDict, total=False):
    """What the plan subgraph reads from the main state.

    The plan stage is the first stage and only consumes ``user_query`` —
    it has no upstream fields to read.
    """

    user_query: str


class PlanOutput(TypedDict, total=False):
    """What the plan subgraph writes back to the main state."""

    research_topics: list[ResearchTopic]
    current_step: Step


class ResearchInput(TypedDict, total=False):
    """What the research subgraph reads from the main state."""

    user_query: str
    research_topics: list[ResearchTopic]


class ResearchOutput(TypedDict, total=False):
    """What the research subgraph writes back to the main state."""

    search_results: list[SearchResult]
    current_step: Step


class WriteInput(TypedDict, total=False):
    """What the write subgraph reads from the main state.

    The write subgraph runs stage 1 (outline + draft) then stage 2
    (citations + References) in one node transition; both stages read
    from this same input set. ``draft`` and ``outline`` are written by
    stage 1 inside the subgraph, so they do not appear here.
    """

    user_query: str
    research_topics: list[ResearchTopic]
    search_results: list[SearchResult]


class WriteOutput(TypedDict, total=False):
    """What the write subgraph writes back to the main state."""

    outline: Outline
    draft: str
    report: str
    current_step: Step


# ─────────────────────────────────────────────────────────────────────────────
# Plan subgraph-local state (v0.2 — issue #11)
#
# ``PlanState`` is the schema the plan subgraph operates on once the HITL
# rewrite (issue #9) replaces the v0.1 single-shot ``plan_decompose`` node. It
# accumulates one round at a time: the questioner writes ``last_questions``,
# the user reply is folded into ``last_answers``, and once
# ``ready_to_proceed`` is True the plan subgraph produces
# ``research_topics`` and the main graph moves on.
# ─────────────────────────────────────────────────────────────────────────────


class PlanState(TypedDict, total=False):
    """Subgraph-local state for the plan HITL loop (issue #9 / #11)."""

    user_query: str
    messages: list[BaseMessage]
    round: int
    last_questions: list[PlanQuestion]
    last_answers: list[PlanUserAnswer]
    ready_to_proceed: bool
    research_topics: list[ResearchTopic]
    current_step: Step
