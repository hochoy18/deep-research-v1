"""Ticket #7 — Write node stage 2 ([N] citations + References list)."""

from __future__ import annotations

import json

from app.models import ResearchTopic, SearchResult
from app.nodes.write_compose import compose_report, write_compose


def _state() -> dict:
    return {
        "user_query": "Q?",
        "research_topics": [
            ResearchTopic(id="topic-1", title="A", description="a", expected_finding_type="x"),
            ResearchTopic(id="topic-2", title="B", description="b", expected_finding_type="y"),
        ],
        "search_results": [
            SearchResult(
                topic_id="topic-1",
                url="https://x.test/a1",
                title="A1 result",
                snippet="snippet a",
                score=0.9,
            ),
            SearchResult(
                topic_id="topic-2",
                url="https://x.test/b1",
                title="B1 result",
                snippet="snippet b",
                score=0.7,
            ),
        ],
        "outline": [],
        "draft": (
            "Intro paragraph.\n\n"
            "## Background\n\nThe history of X goes back a long time. It began in 1990.\n\n"
            "## Current State\n\nToday, X is widely used. Recent numbers are encouraging."
        ),
    }


def test_stage2_appends_inline_citations() -> None:
    delta = compose_report(_state())
    assert delta["current_step"] == "final"
    final = delta["report"]
    assert "[1]" in final or "[2]" in final
    # Both sections should pick up citations since there are matching search results.
    assert final.count("[1]") >= 1
    assert final.count("[2]") >= 1


def test_stage2_cites_section_once_not_per_sentence() -> None:
    """Regression for Spec #8 — a section is sourced by [N] as a whole.

    The deterministic inserter has no per-claim source mapping, so each
    section must receive exactly one [N] marker, placed at the end of
    the section body. Stamping the same [N] on every sentence falsely
    claims each sentence is independently backed by the same single
    source.
    """
    state = _state()
    # Strip the multi-sentence body in section 1 down to three sentences.
    state["draft"] = (
        "Intro paragraph.\n\n"
        "## Background\n\n"
        "First factual claim. Second factual claim. Third factual claim."
    )
    delta = compose_report(state)
    final = delta["report"]
    bg = final.split("## Background", 1)[1].split("## References", 1)[0]
    assert bg.count("[1]") == 1, (
        f"expected 1 inline citation per section, got {bg.count('[1]')}"
    )


def test_stage2_appends_references_section() -> None:
    delta = compose_report(_state())
    final = delta["report"]
    assert "## References" in final
    refs_section = final.split("## References", 1)[1]
    assert "[1] A1 result — https://x.test/a1" in refs_section
    assert "[2] B1 result — https://x.test/b1" in refs_section


def test_stage2_references_match_search_results() -> None:
    """Every entry in the references block maps 1-to-1 to a search result."""
    state = _state()
    delta = compose_report(state)
    final = delta["report"]
    refs_section = final.split("## References", 1)[1]
    for result in state["search_results"]:
        assert f"— {result.url}" in refs_section
        assert result.title in refs_section


def test_stage2_rejects_missing_draft() -> None:
    import pytest

    state = _state()
    state["draft"] = ""
    with pytest.raises(RuntimeError, match="no draft"):
        compose_report(state)


def test_stage2_rejects_missing_search_results() -> None:
    import pytest

    state = _state()
    state["search_results"] = []
    with pytest.raises(RuntimeError, match="no search_results"):
        compose_report(state)


def test_full_write_compose_runs_both_stages(patched_chat_model) -> None:
    """``write_compose`` runs stage 1 then stage 2 in one go."""
    patched_chat_model.responses.append(
        json.dumps(
            {
                "outline": [
                    {"title": "Background", "summary": "s", "key_points": ["p"]},
                    {"title": "Current State", "summary": "s", "key_points": ["p"]},
                ],
                "draft": (
                    "Intro.\n\n"
                    "## Background\n\nFirst body sentence. Second body sentence.\n\n"
                    "## Current State\n\nThird body sentence. Fourth body sentence."
                ),
            }
        )
    )

    delta = write_compose(_state())
    assert "report" in delta
    final = delta["report"]
    assert "## References" in final
    assert "[1]" in final
    assert "[2]" in final