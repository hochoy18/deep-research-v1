"""Ticket #21 — Plan→Research transition gate data shapes.

The gate's typed contracts are pinned here:

- ``ConfirmedResearchTopics`` — body of ``POST /runs/{id}/confirm-plan``
- ``ModifiedTopic`` — one entry in ``PlanEditDiff.modified``
- ``PlanEditDiff`` — state-side audit trail of the user's edits
- ``PlanConfirmInterruptPayload`` — API/SSE payload for ``plan_confirm_required``

Plus the tightened ``ResearchTopic.id`` validator (rejects empty /
whitespace-only ids). Each new model must:

- round-trip through JSON without losing fields;
- be frozen, so the typed payloads the API hands the user are immutable;
- fail closed on missing required fields (no defaults where they shouldn't
  exist).

The discriminator side of the AC ("discriminator / required fields fail
closed") is already exercised for ``PlanQuestion`` in
``test_models_plan.py``; the new models don't introduce a new union, so
required-field failure is what these tests pin.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.models import (
    ConfirmedResearchTopics,
    ModifiedTopic,
    PlanConfirmInterruptPayload,
    PlanEditDiff,
    ResearchTopic,
)


def _topic(i: int = 1, **overrides) -> dict:
    """Return a valid ResearchTopic JSON payload, optionally tweaked."""
    base = {
        "id": f"topic-{i}",
        "title": f"Topic {i}",
        "description": f"What topic {i} investigates",
        "expected_finding_type": "evidence",
    }
    base.update(overrides)
    return base


def _topic_obj(i: int = 1, **overrides) -> ResearchTopic:
    """Return a valid ``ResearchTopic`` instance, optionally tweaked."""
    return ResearchTopic(**_topic(i, **overrides))


# ─────────────────────────────────────────────────────────────────────────────
# ResearchTopic.id — tightened validator (ticket #21)
# ─────────────────────────────────────────────────────────────────────────────


def test_research_topic_accepts_normal_id() -> None:
    topic = ResearchTopic(**_topic())
    assert topic.id == "topic-1"


def test_research_topic_rejects_empty_id() -> None:
    with pytest.raises(ValidationError) as exc:
        ResearchTopic(**_topic(id=""))
    assert "ResearchTopic.id" in str(exc.value) or "non-empty" in str(exc.value)


def test_research_topic_rejects_whitespace_only_id() -> None:
    for blank in (" ", "   ", "\t", "\n", " \t\n "):
        with pytest.raises(ValidationError):
            ResearchTopic(**_topic(id=blank))


def test_research_topic_rejects_empty_id_via_json() -> None:
    payload = _topic()
    payload["id"] = ""
    with pytest.raises(ValidationError):
        ResearchTopic.model_validate(payload)


def test_research_topic_accepts_id_with_inner_whitespace() -> None:
    """Leading/trailing whitespace is allowed — only all-whitespace is rejected.

    Pinning the boundary: ``" topic-1 "`` is a legitimate id string even
    though naive callers would consider it "blank-ish". The validator
    rejects only fully-blank strings.
    """
    topic = ResearchTopic(**_topic(id=" topic-1 "))
    assert topic.id == " topic-1 "


# ─────────────────────────────────────────────────────────────────────────────
# ConfirmedResearchTopics — the API resume payload
# ─────────────────────────────────────────────────────────────────────────────


def test_confirmed_research_topics_round_trips() -> None:
    payload = ConfirmedResearchTopics(
        topics=[_topic_obj(1), _topic_obj(2, title="Topic two")],
    )
    raw = json.loads(payload.model_dump_json())
    assert len(raw["topics"]) == 2
    assert raw["topics"][0]["id"] == "topic-1"
    assert raw["topics"][1]["title"] == "Topic two"
    parsed = ConfirmedResearchTopics.model_validate(raw)
    assert parsed == payload


def test_confirmed_research_topics_requires_topics_field() -> None:
    """Missing ``topics`` fails closed — the API layer can't proceed without it."""
    with pytest.raises(ValidationError):
        ConfirmedResearchTopics()
    with pytest.raises(ValidationError):
        ConfirmedResearchTopics.model_validate({})


def test_confirmed_research_topics_is_frozen() -> None:
    payload = ConfirmedResearchTopics(topics=[_topic_obj()])
    with pytest.raises(ValidationError):
        payload.topics = []  # type: ignore[misc]


def test_confirmed_research_topics_accepts_empty_list_explicitly() -> None:
    """API-layer rejects empty lists with EMPTY_PLAN — the model itself accepts them."""
    payload = ConfirmedResearchTopics.model_validate({"topics": []})
    assert payload.topics == []


def test_confirmed_research_topics_rejects_blank_topic_id() -> None:
    """The ResearchTopic validator still fires inside the wrapped list."""
    with pytest.raises(ValidationError):
        ConfirmedResearchTopics.model_validate(
            {"topics": [_topic(id="   ")]},
        )


def test_confirmed_research_topics_constructs_from_json() -> None:
    raw = json.dumps({"topics": [_topic(1), _topic(2)]})
    payload = ConfirmedResearchTopics.model_validate_json(raw)
    assert [t.id for t in payload.topics] == ["topic-1", "topic-2"]


# ─────────────────────────────────────────────────────────────────────────────
# ModifiedTopic — one entry in PlanEditDiff.modified
# ─────────────────────────────────────────────────────────────────────────────


def test_modified_topic_round_trips() -> None:
    before = _topic_obj(1, title="Old", description="Old desc")
    after = _topic_obj(1, title="New", description="New desc")
    entry = ModifiedTopic(id="topic-1", before=before, after=after)
    raw = json.loads(entry.model_dump_json())
    assert raw["id"] == "topic-1"
    assert raw["before"]["title"] == "Old"
    assert raw["after"]["title"] == "New"
    parsed = ModifiedTopic.model_validate(raw)
    assert parsed == entry


def test_modified_topic_is_frozen() -> None:
    entry = ModifiedTopic(
        id="topic-1", before=_topic_obj(1), after=_topic_obj(1),
    )
    with pytest.raises(ValidationError):
        entry.id = "topic-2"  # type: ignore[misc]


def test_modified_topic_requires_id() -> None:
    with pytest.raises(ValidationError):
        ModifiedTopic.model_validate({"before": _topic(1), "after": _topic(1)})


def test_modified_topic_requires_before() -> None:
    with pytest.raises(ValidationError):
        ModifiedTopic.model_validate({"id": "topic-1", "after": _topic(1)})


def test_modified_topic_requires_after() -> None:
    with pytest.raises(ValidationError):
        ModifiedTopic.model_validate({"id": "topic-1", "before": _topic(1)})


def test_modified_topic_rejects_blank_id_inside_before() -> None:
    """Nested ``ResearchTopic`` validators still fire — the diff trail must be clean."""
    bad = _topic(id="")
    with pytest.raises(ValidationError):
        ModifiedTopic.model_validate(
            {"id": "topic-1", "before": bad, "after": _topic(1)},
        )


# ─────────────────────────────────────────────────────────────────────────────
# PlanEditDiff — state-side audit trail
# ─────────────────────────────────────────────────────────────────────────────


def test_plan_edit_diff_defaults_to_empty_lists() -> None:
    diff = PlanEditDiff()
    assert diff.added == []
    assert diff.removed == []
    assert diff.modified == []


def test_plan_edit_diff_round_trips_full_payload() -> None:
    diff = PlanEditDiff(
        added=[_topic_obj(3)],
        removed=["topic-2"],
        modified=[
            ModifiedTopic(
                id="topic-1",
                before=_topic_obj(1, title="Old"),
                after=_topic_obj(1, title="New"),
            ),
        ],
    )
    raw = json.loads(diff.model_dump_json())
    assert [t["id"] for t in raw["added"]] == ["topic-3"]
    assert raw["removed"] == ["topic-2"]
    assert raw["modified"][0]["id"] == "topic-1"
    parsed = PlanEditDiff.model_validate(raw)
    assert parsed == diff


def test_plan_edit_diff_is_frozen() -> None:
    diff = PlanEditDiff(added=[_topic_obj(1)])
    with pytest.raises(ValidationError):
        diff.added = []  # type: ignore[misc]


def test_plan_edit_diff_accepts_empty_payload_via_json() -> None:
    parsed = PlanEditDiff.model_validate_json("{}")
    assert parsed.added == []
    assert parsed.removed == []
    assert parsed.modified == []


def test_plan_edit_diff_rejects_blank_added_topic_id() -> None:
    with pytest.raises(ValidationError):
        PlanEditDiff.model_validate({"added": [_topic(id="")]})
    with pytest.raises(ValidationError):
        PlanEditDiff.model_validate({"added": [_topic(id=" ")]})


def test_plan_edit_diff_removed_is_list_of_strings() -> None:
    """``removed`` is intentionally just topic ids (not full ResearchTopics)."""
    parsed = PlanEditDiff.model_validate({"removed": ["topic-1", "topic-2"]})
    assert parsed.removed == ["topic-1", "topic-2"]


# ─────────────────────────────────────────────────────────────────────────────
# PlanConfirmInterruptPayload — the API/SSE event payload
# ─────────────────────────────────────────────────────────────────────────────


def test_confirm_interrupt_payload_round_trips() -> None:
    payload = PlanConfirmInterruptPayload(
        topics=[_topic_obj(1), _topic_obj(2)],
        reasoning="Two complementary angles",
    )
    raw = json.loads(payload.model_dump_json())
    assert len(raw["topics"]) == 2
    assert raw["reasoning"] == "Two complementary angles"
    parsed = PlanConfirmInterruptPayload.model_validate(raw)
    assert parsed == payload


def test_confirm_interrupt_payload_reasoning_defaults_to_empty_string() -> None:
    payload = PlanConfirmInterruptPayload(topics=[_topic_obj(1)])
    assert payload.reasoning == ""


def test_confirm_interrupt_payload_is_frozen() -> None:
    payload = PlanConfirmInterruptPayload(topics=[_topic_obj(1)])
    with pytest.raises(ValidationError):
        payload.topics = []  # type: ignore[misc]


def test_confirm_interrupt_payload_requires_topics() -> None:
    with pytest.raises(ValidationError):
        PlanConfirmInterruptPayload.model_validate({"reasoning": "?"})


def test_confirm_interrupt_payload_accepts_empty_topics() -> None:
    """An empty draft is a real LLM failure mode — the model accepts it so the
    gate node / API can flag it with the right error path downstream."""
    payload = PlanConfirmInterruptPayload.model_validate(
        {"topics": [], "reasoning": ""},
    )
    assert payload.topics == []


def test_confirm_interrupt_payload_rejects_blank_topic_id() -> None:
    with pytest.raises(ValidationError):
        PlanConfirmInterruptPayload.model_validate({"topics": [_topic(id="")]})
    with pytest.raises(ValidationError):
        PlanConfirmInterruptPayload.model_validate({"topics": [_topic(id="\t")]})


def test_confirm_interrupt_payload_constructs_from_json() -> None:
    raw = json.dumps({"topics": [_topic(1)], "reasoning": "n/a"})
    payload = PlanConfirmInterruptPayload.model_validate_json(raw)
    assert payload.topics[0].id == "topic-1"
    assert payload.reasoning == "n/a"
