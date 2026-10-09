"""Ticket #3 — Plan node."""

from __future__ import annotations

import json

import pytest


def test_plan_decompose_produces_research_topics(patched_chat_model) -> None:
    canned = {
        "topics": [
            {
                "id": "topic-1",
                "title": "Background",
                "description": "What is the history of X?",
                "expected_finding_type": "factual context",
            },
            {
                "id": "topic-2",
                "title": "Current state",
                "description": "What is the current state of X?",
                "expected_finding_type": "current data",
            },
        ]
    }
    patched_chat_model.responses.append(json.dumps(canned))

    from app.nodes.plan_decompose import plan_decompose

    state = {"user_query": "Tell me about X"}
    delta = plan_decompose(state)

    assert delta["current_step"] == "research"
    topics = delta["research_topics"]
    assert isinstance(topics, list)
    assert len(topics) == 2
    for topic in topics:
        assert hasattr(topic, "id")
        assert hasattr(topic, "title")
        assert hasattr(topic, "description")
        assert hasattr(topic, "expected_finding_type")
    assert topics[0].id == "topic-1"
    assert topics[1].expected_finding_type == "current data"


def test_plan_decompose_strips_markdown_fences(patched_chat_model) -> None:
    canned = {
        "topics": [
            {
                "id": "topic-1",
                "title": "A",
                "description": "B",
                "expected_finding_type": "C",
            }
        ]
    }
    patched_chat_model.responses.append(f"```json\n{json.dumps(canned)}\n```")

    from app.nodes.plan_decompose import plan_decompose

    delta = plan_decompose({"user_query": "Q"})
    assert len(delta["research_topics"]) == 1
    assert delta["research_topics"][0].title == "A"


def test_plan_decompose_truncates_to_five(patched_chat_model) -> None:
    topics = [
        {
            "id": f"topic-{i}",
            "title": f"T{i}",
            "description": f"D{i}",
            "expected_finding_type": "F",
        }
        for i in range(1, 8)  # 7 topics
    ]
    patched_chat_model.responses.append(json.dumps({"topics": topics}))

    from app.nodes.plan_decompose import plan_decompose

    delta = plan_decompose({"user_query": "Q"})
    assert len(delta["research_topics"]) == 5


def test_plan_decompose_rejects_empty_query(patched_chat_model) -> None:
    from app.nodes.plan_decompose import plan_decompose

    with pytest.raises(RuntimeError, match="user_query is empty"):
        plan_decompose({"user_query": ""})


def test_plan_decompose_rejects_non_json_output(patched_chat_model) -> None:
    from app.nodes.plan_decompose import plan_decompose

    patched_chat_model.responses.append("not json at all")
    with pytest.raises(RuntimeError, match="non-JSON"):
        plan_decompose({"user_query": "Q"})


def test_plan_decompose_rejects_missing_topics_key(patched_chat_model) -> None:
    """A JSON object without the 'topics' key fails Pydantic validation."""
    from pydantic import ValidationError

    from app.nodes.plan_decompose import plan_decompose

    patched_chat_model.responses.append(json.dumps({"oops": []}))
    with pytest.raises(ValidationError):
        plan_decompose({"user_query": "Q"})