"""Ticket #8 — End-to-end graph test (full pipeline with mocks)."""

from __future__ import annotations

import json
import time
import uuid


def test_graph_invoke_end_to_end(monkeypatch, memory_checkpointer) -> None:
    """Run the whole pipeline via ``graph.invoke`` with every external mocked."""
    plan_payload = {
        "topics": [
            {
                "id": "topic-1",
                "title": "History",
                "description": "history of X",
                "expected_finding_type": "factual",
            },
            {
                "id": "topic-2",
                "title": "Current state",
                "description": "current state of X",
                "expected_finding_type": "data",
            },
        ]
    }
    write_payload = {
        "outline": [
            {"title": "History", "summary": "s", "key_points": ["p"]},
            {"title": "Current state", "summary": "s", "key_points": ["p"]},
        ],
        "draft": (
            "Intro paragraph.\n\n"
            "## History\n\nFirst historical fact. Second historical fact.\n\n"
            "## Current state\n\nFirst current fact. Second current fact."
        ),
    }

    from app.tools import llm

    class _StubMessage:
        def __init__(self, content: str) -> None:
            self.content = content

    class _StubChat:
        def __init__(self) -> None:
            self.responses = [json.dumps(plan_payload), json.dumps(write_payload)]

        def invoke(self, messages, **_kwargs):
            content = self.responses.pop(0) if self.responses else "{}"
            return _StubMessage(content)

    stub_chat = _StubChat()
    monkeypatch.setattr(llm, "get_chat_model", lambda: stub_chat)

    from app.tools import web_search

    class _StubSearch:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def invoke(self, payload, **_kwargs):
            self.calls.append(payload)
            topic_id = payload["topic_id"]
            return [
                {
                    "topic_id": topic_id,
                    "url": f"https://example.com/{topic_id}",
                    "title": f"Mock result {topic_id}",
                    "snippet": "snippet",
                    "score": 0.9,
                }
            ]

    stub_search = _StubSearch()
    monkeypatch.setattr(web_search, "get_web_search", lambda: stub_search)

    from app.graph import build_graph
    from app.models import Outline
    from app.nodes.research_run import RECURSION_LIMIT

    graph = build_graph(memory_checkpointer)

    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
    }
    start = time.perf_counter()
    state = graph.invoke({"user_query": "Tell me about X"}, config=config)
    elapsed = time.perf_counter() - start

    assert isinstance(state, dict)
    assert "report" in state
    assert state["current_step"] == "final"
    assert len(state["research_topics"]) == 2
    assert len(state["search_results"]) == 2
    assert isinstance(state["outline"], Outline)
    assert len(state["outline"].sections) == 2

    final = state["report"]
    assert "[1]" in final
    assert "[2]" in final
    assert "## References" in final
    refs_section = final.split("## References", 1)[1]
    assert "[1] Mock result topic-1" in refs_section
    assert "[2] Mock result topic-2" in refs_section

    assert elapsed < 30, f"e2e took {elapsed:.2f}s, must be < 30s"


def test_graph_node_chain_is_correct(memory_checkpointer) -> None:
    """Sanity: graph has the expected 4 nodes (plus LangGraph's __start__)."""
    from app.graph import build_graph

    graph = build_graph(memory_checkpointer)
    nodes = list(graph.nodes.keys())
    assert "plan_decompose" in nodes
    assert "research_run" in nodes
    assert "write_compose" in nodes
    assert "finalize" in nodes