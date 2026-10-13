"""Main graph package — re-exports ``build_graph`` for callers (issue #10).

The actual wiring lives in ``app.graph.main``; this ``__init__`` exists so
that ``from app.graph import build_graph`` continues to work (matches the
v0.1 import path) and the package layout matches ADR-0006.
"""

from app.graph.main import build_graph

__all__ = ["build_graph"]