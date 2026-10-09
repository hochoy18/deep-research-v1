"""Ticket #6 — Write node stage 1 (outline + draft, NO reference markers)."""

from __future__ import annotations

import json

import pytest

from app.models import Outline, OutlineSection, ResearchTopic, SearchResult
from app.nodes.write_compose import compose_outline_and_draft


def _state_with_results() -> dict:
    return {
        "user_query": "Q?",
        "research_topics": [
            ResearchTopic(id="topic-1", title="A", description="a", expected_finding_type="x"),
        ],
        "search_results": [
            SearchResult(
                topic_id="topic-1",
                url="https://x.test/a",
                title="A1",
                snippet="snippet a",
                score=0.9,
            )
        ],
    }


def test_stage1_produces_outline_and_draft(patched_chat_model) -> None:
    canned = {
        "outline": [
            {
                "title": "Background",
                "summary": "What came before.",
                "key_points": ["Point 1", "Point 2"],
            }
        ],
        "draft": "Intro paragraph.\n\n## Background\n\nSentence one. Sentence two.",
    }
    patched_chat_model.responses.append(json.dumps(canned))

    delta = compose_outline_and_draft(_state_with_results())

    assert delta["current_step"] == "write"
    assert "outline" in delta
    assert "draft" in delta
    outline = delta["outline"]
    assert isinstance(outline, Outline)
    assert len(outline.sections) == 1
    assert isinstance(outline.sections[0], OutlineSection)
    assert outline.sections[0].title == "Background"
    assert outline.sections[0].key_points == ["Point 1", "Point 2"]

    draft = delta["draft"]
    assert isinstance(draft, str) and draft.strip()
    assert "## Background" in draft


def test_stage1_does_not_insert_reference_markers_or_references(patched_chat_model) -> None:
    patched_chat_model.responses.append(
        json.dumps(
            {
                "outline": [
                    {
                        "title": "Section A",
                        "summary": "s",
                        "key_points": ["p"],
                    }
                ],
                "draft": "Intro.\n\n## Section A\n\nA body.",
            }
        )
    )

    delta = compose_outline_and_draft(_state_with_results())
    assert "[1]" not in delta["draft"]
    assert "[N]" not in delta["draft"]
    assert "## References" not in delta["draft"]


def test_stage1_rejects_draft_with_citation(patched_chat_model) -> None:
    """If the LLM tries to add citations, stage 1 must reject the output."""
    patched_chat_model.responses.append(
        json.dumps(
            {
                "outline": [
                    {"title": "A", "summary": "s", "key_points": ["p"]}
                ],
                "draft": "## A\n\nSome claim [1].",
            }
        )
    )
    with pytest.raises(RuntimeError, match=r"\[N\] markers"):
        compose_outline_and_draft(_state_with_results())


def test_stage1_rejects_empty_outline(patched_chat_model) -> None:
    patched_chat_model.responses.append(json.dumps({"outline": [], "draft": "body"}))
    with pytest.raises(RuntimeError, match="'outline' must be a non-empty list"):
        compose_outline_and_draft(_state_with_results())


def test_stage1_rejects_empty_draft(patched_chat_model) -> None:
    patched_chat_model.responses.append(
        json.dumps({"outline": [{"title": "A", "summary": "s", "key_points": ["p"]}], "draft": "   "})
    )
    with pytest.raises(RuntimeError, match="'draft' must be a non-empty string"):
        compose_outline_and_draft(_state_with_results())