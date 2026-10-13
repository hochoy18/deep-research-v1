"""Research subgraph — wraps ``research_run`` (v0.2 — issue #10).

The subgraph is a one-node ``StateGraph`` whose state schema is ``State``
(matches what ``research_run`` reads) and whose IO is constrained by
``ResearchInput`` / ``ResearchOutput`` so the main graph can compose it
without leaking the research stage's internals.

The v0.1 node is reused unchanged — same recursion-limit guardrail, same
tool-call contract, same output shape.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.nodes.research_run import research_run
from app.state import ResearchInput, ResearchOutput, State


def build_research_subgraph() -> CompiledStateGraph:
    """Return a compiled research subgraph that wraps the v0.1 ``research_run`` node.

    The subgraph declares its IO via ``ResearchInput`` / ``ResearchOutput`` so
    the main graph can constrain which fields flow in and out at this stage.
    """
    builder = StateGraph(State, input_schema=ResearchInput, output_schema=ResearchOutput)
    builder.add_node("research_run", research_run)
    builder.set_entry_point("research_run")
    builder.add_edge("research_run", END)
    return builder.compile()