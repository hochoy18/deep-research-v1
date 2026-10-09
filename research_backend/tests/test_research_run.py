"""Ticket #5 — Research node."""

from __future__ import annotations

import pytest

from app.models import ResearchTopic
from app.nodes.research_run import RECURSION_LIMIT, research_run

_RUN_CONFIG = {"recursion_limit": RECURSION_LIMIT}


def test_research_run_collects_search_results(fake_search_tool) -> None:
    topics = [
        ResearchTopic(
            id="topic-1",
            title="A",
            description="dA",
            expected_finding_type="factual",
        ),
        ResearchTopic(
            id="topic-2",
            title="B",
            description="dB",
            expected_finding_type="data",
        ),
    ]
    delta = research_run({"research_topics": topics}, config=_RUN_CONFIG)

    assert delta["current_step"] == "write"
    results = delta["search_results"]
    assert isinstance(results, list)
    assert len(results) == 2

    for r in results:
        assert r.topic_id in {"topic-1", "topic-2"}
        assert r.url.startswith("https://")
        assert r.title
        assert r.snippet
        assert r.score == 0.9

    # Tool was called once per topic with the right query.
    assert len(fake_search_tool.calls) == 2
    assert fake_search_tool.calls[0]["topic_id"] == "topic-1"
    assert fake_search_tool.calls[1]["topic_id"] == "topic-2"


def test_research_run_rejects_empty_topics(fake_search_tool) -> None:
    with pytest.raises(RuntimeError, match="no research_topics"):
        research_run({}, config=_RUN_CONFIG)


def test_research_run_propagates_results_in_order(fake_search_tool) -> None:
    topics = [
        ResearchTopic(id=f"topic-{i + 1}", title=f"T{i + 1}", description=f"D{i + 1}", expected_finding_type="x")
        for i in range(3)
    ]
    delta = research_run({"research_topics": topics}, config=_RUN_CONFIG)
    assert [r.topic_id for r in delta["search_results"]] == ["topic-1", "topic-2", "topic-3"]


def test_research_run_search_payload_includes_query(fake_search_tool) -> None:
    topic = ResearchTopic(
        id="topic-1",
        title="climate",
        description="long-term temperature",
        expected_finding_type="data",
    )
    research_run({"research_topics": [topic]}, config=_RUN_CONFIG)
    payload = fake_search_tool.calls[0]
    assert "climate" in payload["query"]
    assert "long-term temperature" in payload["query"]


def test_research_run_requires_recursion_limit(fake_search_tool) -> None:
    """Without recursion_limit in config, the node fails fast."""
    topic = ResearchTopic(id="topic-1", title="A", description="a", expected_finding_type="x")
    with pytest.raises(RuntimeError, match="recursion_limit"):
        research_run({"research_topics": [topic]}, config={"configurable": {}})