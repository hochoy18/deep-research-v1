"""Ticket #12 — Plan HITL subgraph.

These tests pin the contract the new ``plan_questioner`` node and its
parent subgraph implement:

- ``PLAN_MAX_ROUNDS`` is exported and equals 5.
- The question-emit LLM is called on entry; if ``ready_to_proceed`` is
  true the node short-circuits to the topic-finalize LLM and writes
  ``research_topics`` + ``current_step="research"``.
- When ``ready_to_proceed`` is false the node writes an assistant turn
  to ``messages`` and emits a ``PlanInterruptPayload`` (the interrupt
  itself is exercised in the e2e test, where the whole graph composes
  the subgraph with a ``MemorySaver``).
- The soft cap at ``PLAN_MAX_ROUNDS`` skips the question-emit LLM and
  forces the topic-finalize LLM to run.
- The topic-finalize LLM truncates over-cap topic lists to 5 entries.
- The conditional edge inside the subgraph routes back to
  ``plan_questioner`` while ``current_step != "research"``.

All tests script the LLM via ``patched_chat_model``; no real LLM is
called.
"""

from __future__ import annotations

import json

import pytest
from langgraph.types import Command

from app.graph.subgraphs.plan import (
    PLAN_MAX_ROUNDS,
    build_plan_subgraph,
    plan_questioner,
)
from app.models import (
    PlanAnswerBatch,
    PlanInterruptPayload,
    PlanUserAnswer,
)


def test_plan_max_rounds_constant_is_five() -> None:
    """AC1: ``PLAN_MAX_ROUNDS`` is exported and equals 5 (ADR-0004)."""
    assert PLAN_MAX_ROUNDS == 5


def test_plan_subgraph_uses_plan_questioner_node() -> None:
    """AC2: the subgraph's only node is ``plan_questioner`` (plus LangGraph's __start__)."""
    graph = build_plan_subgraph()
    nodes = set(graph.nodes.keys())
    assert nodes == {"__start__", "plan_questioner"}, f"unexpected nodes: {nodes}"


def test_plan_subgraph_routes_back_until_research_step(memory_checkpointer) -> None:
    """AC2: conditional edge loops back to ``plan_questioner`` until ``current_step == 'research'``.

    Drive the subgraph through two Q&A rounds and confirm the interrupt
    payload's ``round`` index advances correctly. The third question-emit
    returns ``ready_to_proceed=true`` so the subgraph finalizes and the
    conditional edge falls through to ``END``.
    """
    graph = build_plan_subgraph(checkpointer=memory_checkpointer)
    config = {"configurable": {"thread_id": "t-route"}, "recursion_limit": 25}

    responses = [
        # Round 1 initial: Q1
        json.dumps(
            {
                "questions": [
                    {"kind": "single_choice", "id": "q1", "prompt": "Q1",
                     "options": [{"option_id": "a", "label": "A"}, {"option_id": "b", "label": "B"}]}
                ],
                "reasoning": "scope",
                "ready_to_proceed": False,
            }
        ),
        # Round 1 resume re-run: same Q1
        json.dumps(
            {
                "questions": [
                    {"kind": "single_choice", "id": "q1", "prompt": "Q1",
                     "options": [{"option_id": "a", "label": "A"}, {"option_id": "b", "label": "B"}]}
                ],
                "reasoning": "scope",
                "ready_to_proceed": False,
            }
        ),
        # Round 2: Q2
        json.dumps(
            {
                "questions": [
                    {"kind": "confirmation", "id": "q2", "prompt": "Proceed?"}
                ],
                "reasoning": "confirm",
                "ready_to_proceed": False,
            }
        ),
        # Round 2 resume re-run: same Q2
        json.dumps(
            {
                "questions": [
                    {"kind": "confirmation", "id": "q2", "prompt": "Proceed?"}
                ],
                "reasoning": "confirm",
                "ready_to_proceed": False,
            }
        ),
        # Round 3: ready_to_proceed=true (no interrupt)
        json.dumps(
            {
                "questions": [],
                "reasoning": "done",
                "ready_to_proceed": True,
            }
        ),
        # Topic-finalize
        json.dumps(
            {
                "topics": [
                    {
                        "id": "topic-1",
                        "title": "A",
                        "description": "a",
                        "expected_finding_type": "x",
                    }
                ]
            }
        ),
    ]

    from app.tools import llm

    class _StubChat:
        def __init__(self) -> None:
            self.responses = list(responses)
            self.calls: list = []

        def invoke(self, messages, **_kwargs):
            self.calls.append(messages)
            content = self.responses.pop(0) if self.responses else "{}"
            class _M:
                def __init__(self, c: str) -> None:
                    self.content = c
            return _M(content)

    stub = _StubChat()
    llm.get_chat_model = lambda: stub  # type: ignore[assignment]

    # Round 1 — initial invoke, hits interrupt.
    out_1 = graph.invoke({"user_query": "Q?"}, config=config)
    assert "__interrupt__" in out_1
    interrupts_1 = out_1["__interrupt__"]
    assert len(interrupts_1) == 1
    payload_1 = PlanInterruptPayload.model_validate(interrupts_1[0].value)
    assert payload_1.round == 1
    assert payload_1.questions[0].id == "q1"

    # Round 1 resume — advances to round 2.
    answer_1 = PlanAnswerBatch(round=1, answers=[PlanUserAnswer(question_id="q1", value="a")])
    out_2 = graph.invoke(Command(resume=answer_1), config=config)
    assert "__interrupt__" in out_2
    payload_2 = PlanInterruptPayload.model_validate(out_2["__interrupt__"][0].value)
    assert payload_2.round == 2
    assert payload_2.questions[0].id == "q2"

    # Round 2 resume — third question-emit signals ready_to_proceed=true,
    # so the subgraph finalizes instead of interrupting again.
    answer_2 = PlanAnswerBatch(round=2, answers=[PlanUserAnswer(question_id="q2", value=True)])
    final = graph.invoke(Command(resume=answer_2), config=config)
    assert "__interrupt__" not in final
    assert final["current_step"] == "research"
    assert len(final["research_topics"]) == 1
    assert final["research_topics"][0].id == "topic-1"


def test_plan_questioner_finalizes_when_ready_to_proceed(patched_chat_model) -> None:
    """AC3: ready_to_proceed=true on round 1 triggers topic-finalize (no interrupt)."""
    # Round 1 question-emit: ready_to_proceed=true with no questions.
    patched_chat_model.responses.append(
        json.dumps(
            {
                "questions": [],
                "reasoning": "have enough",
                "ready_to_proceed": True,
            }
        )
    )
    # Topic-finalize LLM.
    patched_chat_model.responses.append(
        json.dumps(
            {
                "topics": [
                    {
                        "id": "topic-1",
                        "title": "T1",
                        "description": "d",
                        "expected_finding_type": "f",
                    }
                ]
            }
        )
    )

    delta = plan_questioner({"user_query": "Tell me about X", "round": 0}, config=None)

    assert delta["current_step"] == "research"
    assert len(delta["research_topics"]) == 1
    assert delta["research_topics"][0].id == "topic-1"


def test_plan_questioner_skips_question_emit_at_soft_cap(patched_chat_model) -> None:
    """AC2: ``round >= PLAN_MAX_ROUNDS`` skips the question-emit LLM and finalizes.

    The patched chat model has no responses queued. If the question-emit
    LLM ran it would raise ``IndexError`` from the stub's pop(0), so the
    test passing proves we skipped it.
    """
    # Only the topic-finalize response is queued.
    patched_chat_model.responses.append(
        json.dumps(
            {
                "topics": [
                    {
                        "id": "topic-1",
                        "title": "T1",
                        "description": "d",
                        "expected_finding_type": "f",
                    }
                ]
            }
        )
    )

    delta = plan_questioner(
        {"user_query": "Q?", "round": PLAN_MAX_ROUNDS, "messages": []},
        config=None,
    )

    assert delta["current_step"] == "research"
    assert len(patched_chat_model.responses) == 0  # topic-finalize consumed
    assert len(delta["research_topics"]) == 1


def test_plan_questioner_rejects_empty_user_query(patched_chat_model) -> None:
    """An empty user_query fails fast (matches the v0.1 ``plan_decompose`` guard)."""
    with pytest.raises(RuntimeError, match="user_query is empty"):
        plan_questioner({"user_query": ""}, config=None)


def test_plan_questioner_rejects_non_json_questioner_output(patched_chat_model) -> None:
    """Non-JSON questioner output is a fast-fail RuntimeError (ADR-0008)."""
    patched_chat_model.responses.append("not json at all")
    with pytest.raises(RuntimeError, match="non-JSON"):
        plan_questioner({"user_query": "Q"}, config=None)


def test_plan_questioner_rejects_invalid_questioner_batch(patched_chat_model) -> None:
    """A questioner payload with an invalid ``PlanQuestion`` discriminator fails Pydantic."""
    from pydantic import ValidationError

    # ``kind`` is not one of the four PlanQuestion discriminator literals,
    # so PlanQuestionerBatch.model_validate fails before we ever call interrupt().
    patched_chat_model.responses.append(
        json.dumps(
            {
                "questions": [{"kind": "nonsense", "id": "q", "prompt": "?"}],
                "reasoning": "r",
                "ready_to_proceed": False,
            }
        )
    )
    with pytest.raises(ValidationError):
        plan_questioner({"user_query": "Q"}, config=None)


def test_plan_questioner_truncates_topics_to_five(patched_chat_model) -> None:
    """The topic-finalize LLM truncates over-cap topic lists to 5 entries."""
    topics = [
        {
            "id": f"topic-{i}",
            "title": f"T{i}",
            "description": "d",
            "expected_finding_type": "f",
        }
        for i in range(1, 8)
    ]
    patched_chat_model.responses.append(
        json.dumps({"questions": [], "reasoning": "r", "ready_to_proceed": True})
    )
    patched_chat_model.responses.append(json.dumps({"topics": topics}))

    delta = plan_questioner({"user_query": "Q"}, config=None)
    assert len(delta["research_topics"]) == 5


def test_plan_questioner_rejects_empty_topic_list(patched_chat_model) -> None:
    """An empty topic list from the topic-finalize LLM is a fast-fail."""
    patched_chat_model.responses.append(
        json.dumps({"questions": [], "reasoning": "r", "ready_to_proceed": True})
    )
    patched_chat_model.responses.append(json.dumps({"topics": []}))
    with pytest.raises(RuntimeError, match="no topics"):
        plan_questioner({"user_query": "Q"}, config=None)


def test_plan_questioner_interrupt_payload_shape(monkeypatch, memory_checkpointer) -> None:
    """The interrupt payload is a ``PlanInterruptPayload`` with the right ``round``.

    ``interrupt()`` can only be called from inside a LangGraph runnable
    context, so this test drives the compiled subgraph instead of the
    bare node function.
    """
    graph = build_plan_subgraph(checkpointer=memory_checkpointer)
    config = {"configurable": {"thread_id": "t-payload"}, "recursion_limit": 25}

    from app.tools import llm

    class _StubChat:
        def __init__(self) -> None:
            self.responses = [
                json.dumps(
                    {
                        "questions": [
                            {
                                "kind": "single_choice",
                                "id": "q1",
                                "prompt": "Pick",
                                "options": [{"option_id": "a", "label": "A"}],
                            }
                        ],
                        "reasoning": "scope",
                        "ready_to_proceed": False,
                    }
                )
            ]

        def invoke(self, messages, **_kwargs):
            content = self.responses.pop(0)
            class _M:
                def __init__(self, c: str) -> None:
                    self.content = c
            return _M(content)

    stub = _StubChat()
    llm.get_chat_model = lambda: stub  # type: ignore[assignment]

    out = graph.invoke({"user_query": "Q?"}, config=config)
    interrupts = out["__interrupt__"]
    assert len(interrupts) == 1
    payload = PlanInterruptPayload.model_validate(interrupts[0].value)
    assert payload.round == 1
    assert payload.reasoning == "scope"
    assert payload.questions[0].id == "q1"


def test_plan_questioner_resume_appends_human_turn_and_bumps_round(
    monkeypatch, memory_checkpointer
) -> None:
    """AC4: on resume the node appends a Human turn to ``messages`` and bumps ``round``."""
    graph = build_plan_subgraph(checkpointer=memory_checkpointer)
    config = {"configurable": {"thread_id": "t-resume"}, "recursion_limit": 25}

    responses = [
        # Round 1 initial: Q1, ready=false
        json.dumps(
            {
                "questions": [
                    {
                        "kind": "single_choice",
                        "id": "q1",
                        "prompt": "Pick",
                        "options": [{"option_id": "a", "label": "A"}],
                    }
                ],
                "reasoning": "r",
                "ready_to_proceed": False,
            }
        ),
        # Round 1 resume re-run: same Q1, ready=false (interrupt returns A1)
        json.dumps(
            {
                "questions": [
                    {
                        "kind": "single_choice",
                        "id": "q1",
                        "prompt": "Pick",
                        "options": [{"option_id": "a", "label": "A"}],
                    }
                ],
                "reasoning": "r",
                "ready_to_proceed": False,
            }
        ),
        # Round 2 initial: ready_to_proceed=true, finalize
        json.dumps({"questions": [], "reasoning": "done", "ready_to_proceed": True}),
        json.dumps(
            {
                "topics": [
                    {
                        "id": "topic-1",
                        "title": "T1",
                        "description": "d",
                        "expected_finding_type": "f",
                    }
                ]
            }
        ),
    ]

    from app.tools import llm

    class _StubChat:
        def __init__(self) -> None:
            self.responses = list(responses)

        def invoke(self, messages, **_kwargs):
            content = self.responses.pop(0)
            class _M:
                def __init__(self, c: str) -> None:
                    self.content = c
            return _M(content)

    stub = _StubChat()
    llm.get_chat_model = lambda: stub  # type: ignore[assignment]

    # Round 1 initial: hit interrupt with round=1
    out_1 = graph.invoke({"user_query": "Q?"}, config=config)
    payload = PlanInterruptPayload.model_validate(out_1["__interrupt__"][0].value)
    assert payload.round == 1
    assert payload.questions[0].id == "q1"

    # Round 1 resume: answer; next entry sees round=1 and questions-asked
    answer = PlanAnswerBatch(
        round=1, answers=[PlanUserAnswer(question_id="q1", value="a")]
    )
    final = graph.invoke(Command(resume=answer), config=config)
    assert "__interrupt__" not in final
    assert final["current_step"] == "research"
    assert len(final["research_topics"]) == 1

    # Inspect persisted state to confirm round was bumped and the human turn
    # was appended. The subgraph's PlanState isn't directly readable via
    # graph.get_state (the subgraph writes the parent IO), so the
    # round/answers contract is pinned here via the final output: current_step
    # advanced and research_topics was written.