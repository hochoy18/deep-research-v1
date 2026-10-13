"""Compiled subgraphs (issue #10).

Each subgraph wraps one v0.1 node and exposes it via a typed IO contract.
The main graph composes the three compiled subgraphs as a flat
``plan → research → write → END`` pipeline.
"""