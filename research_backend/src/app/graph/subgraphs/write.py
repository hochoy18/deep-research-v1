"""Write subgraph — wraps ``write_compose`` (v0.2 — issue #10).

The subgraph is a one-node ``StateGraph`` whose state schema is ``State``
(matches what ``write_compose`` reads) and whose IO is constrained by
``WriteInput`` / ``WriteOutput`` so the main graph can compose it without
leaking the write stage's internals.

The v0.1 node is reused unchanged — both stage 1 (outline + draft) and
stage 2 (citations + References) still run in one node transition, so
the output carries ``outline``, ``draft``, ``report``, and ``current_step``.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.nodes.write_compose import write_compose
from app.state import State, WriteInput, WriteOutput


def build_write_subgraph() -> CompiledStateGraph:
    """Return a compiled write subgraph that wraps the v0.1 ``write_compose`` node.

    The subgraph declares its IO via ``WriteInput`` / ``WriteOutput`` so the
    main graph can constrain which fields flow in and out at this stage.
    """
    builder = StateGraph(State, input_schema=WriteInput, output_schema=WriteOutput)
    builder.add_node("write_compose", write_compose)
    builder.set_entry_point("write_compose")
    builder.add_edge("write_compose", END)
    return builder.compile()