"""Plan subgraph — wraps ``plan_decompose`` (v0.2 — issue #10).

The subgraph is a one-node ``StateGraph`` whose state schema is ``State``
(matches what ``plan_decompose`` reads) and whose IO is constrained by
``PlanInput`` / ``PlanOutput`` so the main graph can compose it without
leaking the plan stage's internals.

This is the **no-behaviour-change** wrap: the v0.1 single-shot
``plan_decompose`` node still does the work. The HITL rewrite that
replaces it lives in issue #9's later steps.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.nodes.plan_decompose import plan_decompose
from app.state import PlanInput, PlanOutput, State


def build_plan_subgraph() -> CompiledStateGraph:
    """Return a compiled plan subgraph that wraps the v0.1 ``plan_decompose`` node.

    The subgraph declares its IO via ``PlanInput`` / ``PlanOutput`` so the
    main graph can constrain which fields flow in and out at this stage.
    """
    builder = StateGraph(State, input_schema=PlanInput, output_schema=PlanOutput)
    builder.add_node("plan_decompose", plan_decompose)
    builder.set_entry_point("plan_decompose")
    builder.add_edge("plan_decompose", END)
    return builder.compile()