"""Small utilities shared between nodes."""

from __future__ import annotations

import json
import re
from typing import Any

from app.models import ModifiedTopic, PlanEditDiff, ResearchTopic

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

# Fields that participate in the "modified" comparison for the transition
# gate diff. ``ResearchTopic`` is the join contract (ticket #21) so any
# new field added there is *not* automatically compared — this constant
# is the explicit allow-list. Pinning it here means the diff's
# semantics stay stable when the model grows new optional fields
# (e.g. ``priority``) that don't represent semantic edits.
_MODIFIED_FIELDS: tuple[str, ...] = ("title", "description", "expected_finding_type")


def strip_code_fences(raw: str) -> str:
    """Strip Markdown code fences (``` or ```json) from LLM output."""
    if raw.startswith("```"):
        raw = raw.strip("`").removeprefix("json").strip()
    return raw


def parse_llm_json(raw: str, node_name: str) -> dict[str, Any]:
    """Strip fences and parse the LLM's output as JSON. Raise on failure."""
    cleaned = strip_code_fences(raw).strip()
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{node_name}: LLM returned non-JSON: {raw!r}") from exc
    if not isinstance(result, dict):
        raise TypeError(
            f"{node_name}: LLM JSON must be an object, got {type(result).__name__}"
        )
    return result


def diff_topics(
    draft: list[ResearchTopic], final: list[ResearchTopic]
) -> PlanEditDiff:
    """Compute the structured audit trail between an LLM draft and the user's
    final topic list (ADR-0005 §7).

    Keying by ``ResearchTopic.id``:

    - ``added``    — topics in ``final`` whose id is not in ``draft`` (full
      ``ResearchTopic`` snapshots, in final order so the diff reads like
      the user's payload).
    - ``removed``  — topic ids in ``draft`` whose id is not in ``final``
      (bare strings per the model contract — issue #21).
    - ``modified`` — same id with a different ``title`` / ``description`` /
      ``expected_finding_type`` (full ``ModifiedTopic(id, before, after)``).

    Order choices:

    - ``added`` follows ``final`` order so it mirrors the user's edit.
    - ``removed`` follows ``draft`` order so a "what did the LLM propose
      that got dropped?" read walks top-to-bottom in the LLM draft.
    - ``modified`` follows the first occurrence in ``final`` of each
      edited id.

    Edge case: if the same id appears more than once in ``final`` (a
    client bug), the *second* occurrence is treated as the ``after``
    of a single ``modified`` entry — we never emit the duplicate under
    ``added`` because the id already exists in ``draft``. This is
    pinned by ``test_diff_topics_id_collision_is_treated_as_modified``.
    """
    draft_by_id: dict[str, ResearchTopic] = {t.id: t for t in draft}
    final_by_id: dict[str, ResearchTopic] = {}

    added: list[ResearchTopic] = []
    modified: list[ModifiedTopic] = []

    for topic in final:
        prior = final_by_id.get(topic.id)
        if prior is not None:
            # Duplicate id in final — treat the later occurrence as the
            # ``after`` of a single ``modified`` entry keyed by this id.
            if topic.id in draft_by_id:
                before = draft_by_id[topic.id]
                if _topic_differs(before, topic):
                    modified.append(
                        ModifiedTopic(id=topic.id, before=before, after=topic)
                    )
            else:
                # No draft match — it's an "added" topic id-wise, but we
                # already saw this id earlier in ``final`` so treat the
                # later occurrence as the "after" for a modified entry
                # whose ``before`` is the earlier final occurrence. This
                # keeps the audit trail consistent without ever emitting
                # the same id twice in ``added``.
                if _topic_differs(prior, topic):
                    modified.append(
                        ModifiedTopic(id=topic.id, before=prior, after=topic)
                    )
            continue

        final_by_id[topic.id] = topic
        if topic.id not in draft_by_id:
            added.append(topic)
            continue

        before = draft_by_id[topic.id]
        if _topic_differs(before, topic):
            modified.append(ModifiedTopic(id=topic.id, before=before, after=topic))

    removed: list[str] = [t.id for t in draft if t.id not in final_by_id]

    return PlanEditDiff(added=added, removed=removed, modified=modified)


def _topic_differs(before: ResearchTopic, after: ResearchTopic) -> bool:
    """True iff any ``_MODIFIED_FIELDS`` field differs between ``before`` and ``after``."""
    for field_name in _MODIFIED_FIELDS:
        if getattr(before, field_name) != getattr(after, field_name):
            return True
    return False
