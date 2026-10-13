"""Ticket #22 — Plan→Research transition gate state schema additions.

The transition gate writes two new fields into the state (``confirmed_research_plan``
and ``plan_edit_diff``) and widens the ``Step`` literal so the SSE detector
can tell a gate-open interrupt from a plan-stage interrupt. These tests pin
the structural AC from issue #22:

- ``State``, ``PlanState``, and ``PlanOutput`` all gain the two new field
  annotations (``PlanOutput`` is the subgraph→main-state channel — without
  it the gate's writes wouldn't propagate to the main ``State``).
- The new annotations reference the canonical models from ticket #21
  (``ResearchTopic``, ``PlanEditDiff``) — no rebinding to a parallel type.
- ``Step`` widens to also include ``"confirm"``.
- The state additions are additive — every v0.1 / v0.2 annotation is still
  present and unchanged.

TypedDict annotations are not enforced at runtime, so these tests introspect
``__annotations__`` directly. That keeps the contract visible to any future
refactor that might silently drop a field without mypy in CI.
"""

from __future__ import annotations

import sys
import typing

import pytest

from app.models import PlanEditDiff, ResearchTopic
from app.state import PlanOutput, PlanState, State, Step


def _annotation_type(typed_dict_cls: type, field_name: str) -> object:
    """Return the resolved runtime type for one ``TypedDict`` field.

    ``TypedDict.__annotations__`` stores annotations as ``ForwardRef``
    strings (PEP 563 — ``from __future__ import annotations``). We use
    ``typing.get_type_hints`` with the module globals as ``localns`` so
    names like ``ResearchTopic`` and ``PlanEditDiff`` (imported with
    ``from app.models import ...``) resolve against ``app.state``'s
    namespace.
    """
    module = sys.modules[typed_dict_cls.__module__]
    hints = typing.get_type_hints(typed_dict_cls, localns=vars(module))
    return hints[field_name]


def _is_list_of(field_type: object, elem_type: type) -> bool:
    """True if ``field_type`` is ``list[elem_type]`` at runtime."""
    if typing.get_origin(field_type) is not list:
        return False
    (actual,) = typing.get_args(field_type)
    return actual is elem_type


# TypedDicts that should declare both transition-gate fields. The matrix is
# 3×2: ``State`` (main), ``PlanState`` (subgraph-local), ``PlanOutput``
# (subgraph→main channel).
_GATE_FIELDS = ("confirmed_research_plan", "plan_edit_diff")


# ─────────────────────────────────────────────────────────────────────────────
# Step — widening with "confirm"
# ─────────────────────────────────────────────────────────────────────────────


def test_step_includes_confirm_literal() -> None:
    """``Step`` widens so the SSE detector can distinguish gate-open from
    plan-stage interrupts (issue #22 AC). All v0.2 values must still be there."""
    args = set(typing.get_args(Step))
    assert args == {"plan", "research", "write", "final", "failed", "confirm"}


# ─────────────────────────────────────────────────────────────────────────────
# Field presence + canonical types on State / PlanState / PlanOutput
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("typed_dict_cls", [State, PlanState, PlanOutput])
@pytest.mark.parametrize("field_name", _GATE_FIELDS)
def test_gate_field_is_declared(
    typed_dict_cls: type, field_name: str,
) -> None:
    """AC #1 + #2: every typed schema involved in the gate declares the field."""
    assert field_name in typed_dict_cls.__annotations__


@pytest.mark.parametrize("typed_dict_cls", [State, PlanState, PlanOutput])
def test_confirmed_research_plan_uses_research_topic_list(
    typed_dict_cls: type,
) -> None:
    """The new field's type is ``list[ResearchTopic]`` — pinned to the canonical
    model from ``app.models`` so the diff and the state agree on the topic
    shape. Holds across ``State``, ``PlanState``, and ``PlanOutput``."""
    assert _is_list_of(
        _annotation_type(typed_dict_cls, "confirmed_research_plan"),
        ResearchTopic,
    )


@pytest.mark.parametrize("typed_dict_cls", [State, PlanState, PlanOutput])
def test_plan_edit_diff_uses_plan_edit_diff_type(typed_dict_cls: type) -> None:
    """The new field's type is the canonical ``PlanEditDiff`` model — no
    parallel re-declaration that would drift from the model-layer contract.
    Holds across ``State``, ``PlanState``, and ``PlanOutput``."""
    assert _annotation_type(typed_dict_cls, "plan_edit_diff") is PlanEditDiff


# ─────────────────────────────────────────────────────────────────────────────
# Additive invariant — every v0.1 / v0.2 field is still on each TypedDict
# ─────────────────────────────────────────────────────────────────────────────


def test_state_preserves_v0_1_v0_2_fields() -> None:
    """AC #4: state additions are additive; no v0.1 / v0.2 field is dropped."""
    annotations = State.__annotations__
    for field in (
        "user_query",
        "research_topics",
        "search_results",
        "outline",
        "draft",
        "report",
        "current_step",
    ):
        assert field in annotations, f"State must still carry {field!r}"


def test_plan_state_preserves_v0_2_fields() -> None:
    """AC #4: ``PlanState`` additions are additive; the v0.2 HITL fields stay."""
    annotations = PlanState.__annotations__
    for field in (
        "user_query",
        "messages",
        "round",
        "last_questions",
        "last_answers",
        "ready_to_proceed",
        "research_topics",
        "current_step",
    ):
        assert field in annotations, f"PlanState must still carry {field!r}"


def test_plan_output_preserves_v0_2_fields() -> None:
    """AC #4 (transitive): ``PlanOutput`` additions are additive; the v0.2
    subgraph→main fields stay. Without this the gate's writes would lose
    the v0.2 channel contract at the same time as they gain v0.3 fields."""
    annotations = PlanOutput.__annotations__
    for field in ("research_topics", "current_step"):
        assert field in annotations, f"PlanOutput must still carry {field!r}"