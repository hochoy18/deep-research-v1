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

v0.3 — issue #22: Plan→Research transition gate state additions.
``confirmed_research_plan`` holds the user's final topic list once the gate
resumes; ``plan_edit_diff`` is the structured audit trail of what they added /
removed / modified. Both are written by the gate node (issue #23) and live
alongside the LLM draft ``research_topics`` (ADR-0005 §6 calls this field
``research_plan`` — the name has not been migrated yet, so the docstring
keeps the pre-existing v0.1 spelling) so post-hoc "report drift" analysis
can attribute changes to either side (ADR-0005 §6 / §7).
``Step`` widens with ``"confirm"`` so the SSE detector can distinguish a
gate-open interrupt from a plan-stage interrupt (issue #25).
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from app.models import (
    Outline,
    PlanEditDiff,
    PlanQuestion,
    PlanUserAnswer,
    ResearchTopic,
    SearchResult,
)

# Pipeline stages; mirrored by the LangGraph node names.
# v0.3 — issue #22: ``"confirm"`` marks the gate-open state. The plan
# subgraph writes ``current_step="confirm"`` when it pauses for the user to
# review the topic list, and resumes to ``"research"`` once the user submits
# their final list via ``POST /runs/{id}/confirm-plan``.
Step = Literal["plan", "research", "write", "final", "failed", "confirm"]


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

    # Populated by confirm_plan_gate (issue #23) — the user's final topic list
    # after editing at the Plan→Research transition gate. Lives alongside
    # the LLM draft ``research_topics`` (ADR-0005 §6 names the draft field
    # ``research_plan``; the v0.1 spelling sticks until a future rename) so
    # the API can return both for post-hoc drift analysis. The Research
    # subgraph reads from here, never from ``research_topics`` directly.
    confirmed_research_plan: list[ResearchTopic]

    # Populated by confirm_plan_gate (issue #23) — structured audit trail of
    # the user's edits at the gate (added / removed / modified topics; see
    # ADR-0005 §7 and the ``PlanEditDiff`` model from issue #21).
    plan_edit_diff: PlanEditDiff


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
    """What the plan subgraph writes back to the main state.

    v0.3 — issue #22: gains the two transition-gate fields so the gate's
    subgraph-local writes (``PlanState.confirmed_research_plan`` /
    ``plan_edit_diff``) actually reach the main ``State``. Without these
    entries the subgraph's ``output_schema`` would silently drop them at
    the subgraph boundary and the main graph would never see the user's
    final topic list or the audit trail (ADR-0005 §1).
    """

    research_topics: list[ResearchTopic]
    current_step: Step
    confirmed_research_plan: list[ResearchTopic]
    plan_edit_diff: PlanEditDiff


class ResearchInput(TypedDict, total=False):
    """What the research subgraph reads from the main state.

    v0.3 — issue #26: gains ``confirmed_research_plan`` so the user's
    final topic list (written by ``confirm_plan_gate``) flows into the
    research stage. ``research_run`` prefers it over the LLM draft
    ``research_topics`` — the searches must run on what the user
    confirmed, not what the LLM first proposed (ADR-0005 §6).
    """

    user_query: str
    research_topics: list[ResearchTopic]
    confirmed_research_plan: list[ResearchTopic]


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
    """Subgraph-local state for the plan HITL loop (issue #9 / #11).

    v0.3 — issue #22: gains the same two transition-gate fields as the main
    ``State`` (``confirmed_research_plan`` and ``plan_edit_diff``). The gate
    lives inside the plan subgraph (ADR-0005 §1), so it writes here first;
    the main graph reads them via ``PlanOutput`` once the subgraph completes.
    """

    user_query: str
    messages: list[BaseMessage]
    round: int
    last_questions: list[PlanQuestion]
    last_answers: list[PlanUserAnswer]
    ready_to_proceed: bool
    research_topics: list[ResearchTopic]
    current_step: Step

    # Populated by confirm_plan_gate (issue #23) — same fields as on
    # ``State``. Mirrored here because the gate writes inside the subgraph
    # and the subgraph must declare them on its own schema so the typed
    # node signature compiles. ``PlanOutput`` is the upstream channel
    # that surfaces these writes to the main ``State``.
    confirmed_research_plan: list[ResearchTopic]
    plan_edit_diff: PlanEditDiff
