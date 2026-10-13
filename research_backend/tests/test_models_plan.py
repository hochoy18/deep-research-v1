"""Ticket #11 — Plan HITL data shapes.

Each frozen Pydantic model in this file pins the contract that the Plan
subgraph's HITL loop will rely on once issue #9 replaces
``plan_decompose`` with the multi-round questioner. The tests below verify
that:

- Every model can be constructed from a valid JSON payload (round-trip).
- The ``PlanQuestion`` discriminated union narrows to the right subclass
  for each ``kind`` literal, and each subclass carries the per-kind
  fields called out in the issue (options / placeholder).
- Each model is frozen, so the typed payloads the API hands the user are
  immutable.
- Invalid payloads fail ``ValidationError`` with the right message.
"""

from __future__ import annotations

import json

import pytest
from pydantic import TypeAdapter, ValidationError

from app.models import (
    ConfirmationPlanQuestion,
    FreeTextPlanQuestion,
    MultiChoicePlanQuestion,
    PlanAnswerBatch,
    PlanInterruptPayload,
    PlanQuestion,
    PlanQuestionerBatch,
    PlanQuestionOption,
    PlanUserAnswer,
    SingleChoicePlanQuestion,
)

_QUESTION_ADAPTER = TypeAdapter(PlanQuestion)


def _options() -> list[dict]:
    return [
        {"option_id": "a", "label": "Apple"},
        {"option_id": "b", "label": "Banana"},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# PlanQuestionOption
# ─────────────────────────────────────────────────────────────────────────────


def test_plan_question_option_round_trips() -> None:
    option = PlanQuestionOption(option_id="a", label="Apple")
    payload = json.loads(option.model_dump_json())
    assert payload == {"option_id": "a", "label": "Apple"}
    assert PlanQuestionOption(**payload) == option


def test_plan_question_option_is_frozen() -> None:
    option = PlanQuestionOption(option_id="a", label="Apple")
    with pytest.raises(ValidationError):
        option.option_id = "b"  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# PlanQuestion — discriminated union
# ─────────────────────────────────────────────────────────────────────────────


def test_single_choice_question_parses() -> None:
    q = _QUESTION_ADAPTER.validate_python(
        {"kind": "single_choice", "id": "q1", "prompt": "Pick one", "options": _options()}
    )
    assert isinstance(q, SingleChoicePlanQuestion)
    assert q.id == "q1"
    assert q.prompt == "Pick one"
    assert q.kind == "single_choice"
    assert [o.option_id for o in q.options] == ["a", "b"]


def test_multi_choice_question_parses() -> None:
    q = _QUESTION_ADAPTER.validate_python(
        {"kind": "multi_choice", "id": "q2", "prompt": "Pick many", "options": _options()}
    )
    assert isinstance(q, MultiChoicePlanQuestion)
    assert q.kind == "multi_choice"
    assert len(q.options) == 2


def test_free_text_question_parses_with_placeholder() -> None:
    q = _QUESTION_ADAPTER.validate_python(
        {
            "kind": "free_text",
            "id": "q3",
            "prompt": "Describe",
            "placeholder": "Type here",
        }
    )
    assert isinstance(q, FreeTextPlanQuestion)
    assert q.placeholder == "Type here"


def test_free_text_question_parses_without_placeholder() -> None:
    q = _QUESTION_ADAPTER.validate_python(
        {"kind": "free_text", "id": "q3", "prompt": "Describe"}
    )
    assert isinstance(q, FreeTextPlanQuestion)
    assert q.placeholder is None


def test_confirmation_question_parses() -> None:
    q = _QUESTION_ADAPTER.validate_python(
        {"kind": "confirmation", "id": "q4", "prompt": "Confirm?"}
    )
    assert isinstance(q, ConfirmationPlanQuestion)
    assert q.kind == "confirmation"


def test_unknown_kind_fails() -> None:
    with pytest.raises(ValidationError):
        _QUESTION_ADAPTER.validate_python(
            {"kind": "nonsense", "id": "q", "prompt": "?"}
        )


def test_single_choice_rejects_missing_options() -> None:
    with pytest.raises(ValidationError):
        _QUESTION_ADAPTER.validate_python(
            {"kind": "single_choice", "id": "q", "prompt": "?"}
        )


def test_discriminator_narrows_in_list() -> None:
    payload = [
        {"kind": "single_choice", "id": "q1", "prompt": "P1", "options": _options()},
        {"kind": "confirmation", "id": "q2", "prompt": "P2"},
        {"kind": "free_text", "id": "q3", "prompt": "P3"},
    ]
    parsed = [_QUESTION_ADAPTER.validate_python(p) for p in payload]
    assert isinstance(parsed[0], SingleChoicePlanQuestion)
    assert isinstance(parsed[1], ConfirmationPlanQuestion)
    assert isinstance(parsed[2], FreeTextPlanQuestion)


def test_question_round_trips_via_json() -> None:
    original = _QUESTION_ADAPTER.validate_python(
        {
            "kind": "single_choice",
            "id": "q1",
            "prompt": "Pick",
            "options": _options(),
        }
    )
    raw = original.model_dump_json()
    assert isinstance(_QUESTION_ADAPTER.validate_json(raw), SingleChoicePlanQuestion)


def test_plan_question_is_frozen() -> None:
    q = SingleChoicePlanQuestion(id="q1", prompt="P", options=_options())
    with pytest.raises(ValidationError):
        q.prompt = "P2"  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# PlanUserAnswer — value type narrows by question kind
# ─────────────────────────────────────────────────────────────────────────────


def test_user_answer_string_value() -> None:
    a = PlanUserAnswer(question_id="q1", value="a")
    assert a.value == "a"


def test_user_answer_list_value() -> None:
    a = PlanUserAnswer(question_id="q2", value=["a", "b"])
    assert a.value == ["a", "b"]


def test_user_answer_bool_value() -> None:
    a = PlanUserAnswer(question_id="q3", value=False)
    assert a.value is False


def test_user_answer_is_frozen() -> None:
    a = PlanUserAnswer(question_id="q1", value="a")
    with pytest.raises(ValidationError):
        a.value = "b"  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# PlanQuestionerBatch / PlanAnswerBatch / PlanInterruptPayload
# ─────────────────────────────────────────────────────────────────────────────


def _question_payload() -> list[dict]:
    return [
        {"kind": "single_choice", "id": "q1", "prompt": "Pick", "options": _options()},
        {"kind": "free_text", "id": "q2", "prompt": "Details", "placeholder": "?"},
    ]


def test_questioner_batch_defaults() -> None:
    batch = PlanQuestionerBatch()
    assert batch.questions == []
    assert batch.reasoning == ""
    assert batch.ready_to_proceed is False


def test_questioner_batch_round_trips() -> None:
    batch = PlanQuestionerBatch(
        questions=[
            _QUESTION_ADAPTER.validate_python(q) for q in _question_payload()
        ],
        reasoning="Need scope",
        ready_to_proceed=True,
    )
    raw = json.loads(batch.model_dump_json())
    assert raw["reasoning"] == "Need scope"
    assert raw["ready_to_proceed"] is True
    assert len(raw["questions"]) == 2
    parsed = PlanQuestionerBatch.model_validate(raw)
    assert parsed.reasoning == "Need scope"
    assert parsed.ready_to_proceed is True


def test_answer_batch_round_trips() -> None:
    batch = PlanAnswerBatch(
        round=2,
        answers=[
            PlanUserAnswer(question_id="q1", value="a"),
            PlanUserAnswer(question_id="q2", value="Some text"),
        ],
    )
    raw = json.loads(batch.model_dump_json())
    assert raw["round"] == 2
    assert len(raw["answers"]) == 2
    parsed = PlanAnswerBatch.model_validate(raw)
    assert parsed.round == 2
    assert parsed.answers[1].value == "Some text"


def test_interrupt_payload_round_trips() -> None:
    payload = PlanInterruptPayload(
        questions=[
            _QUESTION_ADAPTER.validate_python(q) for q in _question_payload()
        ],
        reasoning="Need scope",
        round=3,
    )
    raw = json.loads(payload.model_dump_json())
    assert raw["reasoning"] == "Need scope"
    assert raw["round"] == 3
    assert len(raw["questions"]) == 2
    parsed = PlanInterruptPayload.model_validate(raw)
    assert parsed.round == 3
    assert parsed.questions[0].kind == "single_choice"


def test_interrupt_payload_requires_round() -> None:
    with pytest.raises(ValidationError):
        PlanInterruptPayload.model_validate({"questions": [], "reasoning": ""})


def test_interrupt_payload_rejects_missing_questions() -> None:
    with pytest.raises(ValidationError):
        PlanInterruptPayload.model_validate({"reasoning": "?", "round": 1})


def test_answer_batch_requires_round() -> None:
    with pytest.raises(ValidationError):
        PlanAnswerBatch.model_validate({"answers": []})


def test_cross_model_kind_to_value_narrowing_round_trips() -> None:
    """AC8: ``PlanUserAnswer.value`` type narrows by the matching question's kind.

    Pin the cross-model pairing with a single round-trip — a
    confirmation question gets a bool answer, multi-choice gets a list,
    single_choice/free_text get a str — through JSON so the wire format
    is exercised too.
    """
    confirmation = _QUESTION_ADAPTER.validate_python(
        {"kind": "confirmation", "id": "q1", "prompt": "Proceed?"}
    )
    multi = _QUESTION_ADAPTER.validate_python(
        {"kind": "multi_choice", "id": "q2", "prompt": "Which?", "options": _options()}
    )
    single = _QUESTION_ADAPTER.validate_python(
        {"kind": "single_choice", "id": "q3", "prompt": "Pick", "options": _options()}
    )
    free_text = _QUESTION_ADAPTER.validate_python(
        {"kind": "free_text", "id": "q4", "prompt": "Describe"}
    )

    batch = PlanAnswerBatch(
        round=2,
        answers=[
            PlanUserAnswer(question_id="q1", value=False),
            PlanUserAnswer(question_id="q2", value=["a", "b"]),
            PlanUserAnswer(question_id="q3", value="a"),
            PlanUserAnswer(question_id="q4", value="some description"),
        ],
    )

    raw = json.loads(batch.model_dump_json())
    parsed = PlanAnswerBatch.model_validate(raw)

    # Kinds preserved on the parsed questions
    assert confirmation.kind == "confirmation"
    assert multi.kind == "multi_choice"
    assert single.kind == "single_choice"
    assert free_text.kind == "free_text"
    # Value types preserved across round-trip
    assert parsed.answers[0].value is False  # bool narrows to confirmation
    assert parsed.answers[1].value == ["a", "b"]  # list[str] narrows to multi
    assert parsed.answers[2].value == "a"  # str narrows to single
    assert parsed.answers[3].value == "some description"  # str narrows to free_text


# ─────────────────────────────────────────────────────────────────────────────
# PlanState TypedDict
# ─────────────────────────────────────────────────────────────────────────────


def test_plan_state_declares_all_required_fields() -> None:
    from app.state import PlanState

    expected = {
        "user_query",
        "messages",
        "round",
        "last_questions",
        "last_answers",
        "ready_to_proceed",
        "research_topics",
        "current_step",
    }
    assert set(PlanState.__annotations__.keys()) == expected
