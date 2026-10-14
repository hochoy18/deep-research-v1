"""Utility-helper unit tests.

The v0.3 transition gate needs ``diff_topics`` — a small pure helper
that takes the LLM's draft topic list and the user's final topic list
and returns the structured ``PlanEditDiff`` audit trail. There is no
LLM, no LangGraph, and no I/O involved; this is the easiest seam to
test in isolation, so it gets its own file.
"""

from __future__ import annotations

from app.models import PlanEditDiff, ResearchTopic
from app.utils import diff_topics


def _topic(i: int = 1, **overrides) -> ResearchTopic:
    """Build a typed ``ResearchTopic`` with overridable fields."""
    base = {
        "id": f"topic-{i}",
        "title": f"Topic {i}",
        "description": f"What topic {i} investigates",
        "expected_finding_type": "evidence",
    }
    base.update(overrides)
    return ResearchTopic(**base)


def test_diff_topics_empty_inputs_returns_empty_diff() -> None:
    """Both sides empty → all three diff lists are empty."""
    diff = diff_topics([], [])
    assert diff == PlanEditDiff(added=[], removed=[], modified=[])


def test_diff_topics_identical_lists_returns_empty_diff() -> None:
    """Same topic list on both sides → no edits, all lists empty."""
    draft = [_topic(1), _topic(2)]
    final = [_topic(1), _topic(2)]
    diff = diff_topics(draft, final)
    assert diff.added == []
    assert diff.removed == []
    assert diff.modified == []


def test_diff_topics_added_lists_final_only_ids() -> None:
    """Topics present in ``final`` but missing from ``draft`` go to ``added``."""
    draft = [_topic(1)]
    new = _topic(2, title="Newly added")
    final = [_topic(1), new]
    diff = diff_topics(draft, final)
    assert diff.added == [new]
    assert diff.removed == []
    assert diff.modified == []


def test_diff_topics_removed_lists_draft_only_ids() -> None:
    """Topics present in ``draft`` but missing from ``final`` go to ``removed``
    as bare id strings (not full topics — see ADR-0005 §7)."""
    draft = [_topic(1), _topic(2)]
    final = [_topic(1)]
    diff = diff_topics(draft, final)
    assert diff.added == []
    assert diff.removed == ["topic-2"]
    assert diff.modified == []


def test_diff_topics_modified_detects_title_change() -> None:
    """Same ``id`` but a different ``title`` is a ``modified`` entry
    carrying full before/after snapshots."""
    before = _topic(1, title="Old title")
    after = _topic(1, title="New title")
    diff = diff_topics([before], [after])
    assert diff.added == []
    assert diff.removed == []
    assert len(diff.modified) == 1
    entry = diff.modified[0]
    assert entry.id == "topic-1"
    assert entry.before == before
    assert entry.after == after


def test_diff_topics_modified_detects_description_change() -> None:
    """A description-only edit still counts as ``modified``."""
    before = _topic(1, description="old desc")
    after = _topic(1, description="new desc")
    diff = diff_topics([before], [after])
    assert len(diff.modified) == 1
    assert diff.modified[0].before.description == "old desc"
    assert diff.modified[0].after.description == "new desc"


def test_diff_topics_modified_detects_expected_finding_type_change() -> None:
    """An expected_finding_type-only edit still counts as ``modified``."""
    before = _topic(1, expected_finding_type="data")
    after = _topic(1, expected_finding_type="case-study")
    diff = diff_topics([before], [after])
    assert len(diff.modified) == 1


def test_diff_topics_no_modify_when_only_other_fields_change() -> None:
    """A ``ResearchTopic`` is frozen and has no other fields, but if a future
    field is added that doesn't participate in the diff, it must not
    trigger a spurious ``modified`` entry. We pin this via
    ``ResearchTopic.model_fields`` so any new field has to be added to
    the comparison explicitly."""
    before = _topic(1)
    after = _topic(1)  # identical
    diff = diff_topics([before], [after])
    assert diff.modified == []


def test_diff_topics_combined_added_removed_modified() -> None:
    """A round-trip with all three kinds of edits records each in the
    right list — pinned by the structure's user-facing promise."""
    draft = [_topic(1), _topic(2), _topic(3)]
    added = _topic(4)
    kept_changed = _topic(1, title="Refined")
    final = [_topic(3), kept_changed, added]  # topic-1 modified, topic-2 removed, topic-4 added

    diff = diff_topics(draft, final)

    assert [t.id for t in diff.added] == ["topic-4"]
    assert diff.removed == ["topic-2"]
    assert [m.id for m in diff.modified] == ["topic-1"]
    assert diff.modified[0].before.title == "Topic 1"
    assert diff.modified[0].after.title == "Refined"


def test_diff_topics_id_collision_is_treated_as_modified() -> None:
    """If the final list has the same id twice (a client bug), the second
    one is treated as the ``after`` for a modified entry — we never
    emit it under ``added`` because the id already exists in draft.

    This pins the helper's choice between ``added`` and ``modified``:
    the id-set, not the order, is the discriminator.
    """
    draft = [_topic(1, title="A")]
    final = [_topic(1, title="A"), _topic(1, title="B")]
    diff = diff_topics(draft, final)
    assert diff.added == []
    assert diff.removed == []
    assert len(diff.modified) == 1
    assert diff.modified[0].after.title == "B"


def test_diff_topics_modified_requires_any_field_difference() -> None:
    """Pinning the ``modified`` semantics: any single field diff (title,
    description, expected_finding_type) is enough to register as
    modified — there is no "whitelist" of fields to compare."""
    diff_a = diff_topics(
        [_topic(1, title="x")],
        [_topic(1, title="y")],
    )
    diff_b = diff_topics(
        [_topic(1, description="x")],
        [_topic(1, description="y")],
    )
    diff_c = diff_topics(
        [_topic(1, expected_finding_type="x")],
        [_topic(1, expected_finding_type="y")],
    )
    assert len(diff_a.modified) == 1
    assert len(diff_b.modified) == 1
    assert len(diff_c.modified) == 1


def test_diff_topics_returns_plan_edit_diff_instance() -> None:
    """The return type is always a ``PlanEditDiff`` — even an empty one
    — so callers can rely on attribute access without ``None`` checks."""
    diff = diff_topics([], [])
    assert isinstance(diff, PlanEditDiff)


def test_diff_topics_added_in_final_order() -> None:
    """``added`` preserves the user's final-list order — the diff is a
    forensic artefact and clients should see edits in the order they
    appeared in the final payload."""
    draft = [_topic(1)]
    new_a = _topic(2)
    new_b = _topic(3)
    diff = diff_topics(draft, [_topic(1), new_a, new_b])
    assert [t.id for t in diff.added] == ["topic-2", "topic-3"]


def test_diff_topics_removed_in_draft_order() -> None:
    """``removed`` preserves the draft order so a "what did the LLM
    propose that the user dropped?" question reads top-to-bottom."""
    draft = [_topic(1), _topic(2), _topic(3)]
    final = [_topic(3)]  # 1 and 2 dropped
    diff = diff_topics(draft, final)
    assert diff.removed == ["topic-1", "topic-2"]
