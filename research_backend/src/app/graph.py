"""LangGraph wiring (ticket #8).

Single flat graph: ``plan_decompose → research_run → write_compose → finalize``.
The caller picks the checkpointer — the API passes a ``RedisSaver`` while
tests pass an in-memory ``MemorySaver``.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.nodes.plan_decompose import plan_decompose
from app.nodes.research_run import research_run
from app.nodes.write_compose import write_compose
from app.state import State


def _finalize(state: State) -> dict:
    """Trivial terminal node — just records that we reached the end."""
    return {"current_step": "final"}


def build_graph(checkpointer) -> CompiledStateGraph:
    """Return a compiled graph bound to the supplied checkpointer.

    The checkpointer is mandatory so that ``GET /runs/{thread_id}`` always
    reads state from a durable source. Callers pass ``recursion_limit`` in
    their own ``RunnableConfig`` (see ``app.api.routes._build_config``).
    """
    builder = StateGraph(State)
    builder.add_node("plan_decompose", plan_decompose)
    builder.add_node("research_run", research_run)
    builder.add_node("write_compose", write_compose)
    builder.add_node("finalize", _finalize)

    builder.set_entry_point("plan_decompose")
    builder.add_edge("plan_decompose", "research_run")
    builder.add_edge("research_run", "write_compose")
    builder.add_edge("write_compose", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile(
        checkpointer=checkpointer,
    )