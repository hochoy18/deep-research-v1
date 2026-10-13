"""Main graph wiring (v0.2 — issue #10 / ADR-0006).

Composes the plan / research / write subgraphs as compiled subgraphs,
plus an explicit ``finalize`` terminal that owns the
``current_step: "final"`` write so the main graph doesn't depend on
``write_compose`` writing it as a side effect:

    plan_subgraph → research_subgraph → write_subgraph → finalize → END

Each subgraph declares its own IO contract (``PlanInput``/``PlanOutput``,
``ResearchInput``/``ResearchOutput``, ``WriteInput``/``WriteOutput``) so
the stage boundaries are typed. The v0.1 nodes inside each subgraph are
unchanged — this is the no-behaviour-change wrap that sets the stage for
the HITL rewrite in issue #9.

The caller picks the checkpointer — the API passes a ``RedisSaver`` while
tests pass an in-memory ``MemorySaver``.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.graph.subgraphs.plan import build_plan_subgraph
from app.graph.subgraphs.research import build_research_subgraph
from app.graph.subgraphs.write import build_write_subgraph
from app.state import State


def build_graph(checkpointer) -> CompiledStateGraph:
    """Return a compiled main graph bound to the supplied checkpointer.

    The compiled graph has the three subgraph nodes plus an explicit
    ``finalize`` terminal, with edges
    ``plan_subgraph → research_subgraph → write_subgraph → finalize → END``.
    The checkpointer is mandatory so that ``GET /runs/{thread_id}``
    always reads state from a durable source. Callers pass
    ``recursion_limit`` in their own ``RunnableConfig`` (see
    ``app.api.routes._build_config``).
    """
    plan_subgraph = build_plan_subgraph()
    research_subgraph = build_research_subgraph()
    write_subgraph = build_write_subgraph()

    builder = StateGraph(State)
    builder.add_node("plan_subgraph", plan_subgraph)
    builder.add_node("research_subgraph", research_subgraph)
    builder.add_node("write_subgraph", write_subgraph)
    builder.add_node("finalize", _finalize)

    builder.set_entry_point("plan_subgraph")
    builder.add_edge("plan_subgraph", "research_subgraph")
    builder.add_edge("research_subgraph", "write_subgraph")
    builder.add_edge("write_subgraph", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer)


def _finalize(state: State) -> dict:
    """Terminal node — assert we reached the end. (v0.2 — issue #10)

    Owns the ``current_step: "final"`` write so the main graph's terminal
    state is not borrowed from ``write_compose``'s implicit side effect.
    If a future refactor removes the ``"final"`` write from
    ``write_compose``, the E2E test ``test_graph_invoke_end_to_end``
    starts failing at line 96 — but the failure points at this contract,
    not at the v0.2 terminal-node regression.
    """
    return {"current_step": "final"}


__all__ = ["build_graph"]