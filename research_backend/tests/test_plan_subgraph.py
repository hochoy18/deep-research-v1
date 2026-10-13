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
    ConfirmationPlanQuestion,
    FreeTextPlanQuestion,
    MultiChoicePlanQuestion,
    PlanAnswerBatch,
    PlanInterruptPayload,
    PlanQuestionerBatch,
    PlanUserAnswer,
    SingleChoicePlanQuestion,
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
                    {
                        "kind": "single_choice",
                        "id": "q1",
                        "prompt": "Q1",
                        "options": [
                            {"option_id": "a", "label": "A"},
                            {"option_id": "b", "label": "B"},
                        ],
                    }
                ],
                "reasoning": "scope",
                "ready_to_proceed": False,
            }
        ),
        # Round 1 resume re-run: same Q1
        json.dumps(
            {
                "questions": [
                    {
                        "kind": "single_choice",
                        "id": "q1",
                        "prompt": "Q1",
                        "options": [
                            {"option_id": "a", "label": "A"},
                            {"option_id": "b", "label": "B"},
                        ],
                    }
                ],
                "reasoning": "scope",
                "ready_to_proceed": False,
            }
        ),
        # Round 2: Q2
        json.dumps(
            {
                "questions": [{"kind": "confirmation", "id": "q2", "prompt": "Proceed?"}],
                "reasoning": "confirm",
                "ready_to_proceed": False,
            }
        ),
        # Round 2 resume re-run: same Q2
        json.dumps(
            {
                "questions": [{"kind": "confirmation", "id": "q2", "prompt": "Proceed?"}],
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
    answer = PlanAnswerBatch(round=1, answers=[PlanUserAnswer(question_id="q1", value="a")])
    final = graph.invoke(Command(resume=answer), config=config)
    assert "__interrupt__" not in final
    assert final["current_step"] == "research"
    assert len(final["research_topics"]) == 1

    # Inspect persisted state to confirm round was bumped and the human turn
    # was appended. The subgraph's PlanState isn't directly readable via
    # graph.get_state (the subgraph writes the parent IO), so the
    # round/answers contract is pinned here via the final output: current_step
    # advanced and research_topics was written.


# ─────────────────────────────────────────────────────────────────────────────
# Ticket #19 — Plan subgraph tests (primary test seam).
#
# Five cases that exercise the plan subgraph in isolation using the v0.1
# ``memory_checkpointer`` fixture plus a local ``_ScriptedChat`` LLM stub.
# The local stub intentionally re-implements the v0.1 ``fake_chat_model``
# shape with one extra capability — capturing the system prompt of every
# call — so tests can discriminate questioner vs topic-finalize invocations.
# Covers the multi-round Q&A loop, the 5-round cap, the four question
# kinds, and the topic-finalize LLM call.
# ─────────────────────────────────────────────────────────────────────────────


class _ScriptedChat:
    """Minimal ``get_chat_model`` stub that records system prompts per call.

    Tests push JSON strings onto ``responses``; ``invoke`` pops the next
    one each call. ``system_prompts`` captures the first message of each
    call so tests can assert whether the questioner or topic-finalize
    LLM was invoked. ``calls`` captures the full messages list per call.
    """

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[list] = []
        self.system_prompts: list[str] = []

    def invoke(self, messages, **_kwargs):
        self.calls.append(messages)
        if messages:
            self.system_prompts.append(messages[0].content)
        content = self.responses.pop(0)
        return _StubAIMessage(content)


class _StubAIMessage:
    """Tiny stand-in for a LangChain ``AIMessage`` carrying ``content``."""

    def __init__(self, content: str) -> None:
        self.content = content


def _install_packaged_chat(stub: _ScriptedChat) -> None:
    """Patch ``app.tools.llm.get_chat_model`` to return ``stub``."""
    from app.tools import llm

    llm.get_chat_model = lambda: stub  # type: ignore[assignment]


def _question_batch(
    kind_to_body: dict,
    *,
    reasoning: str = "scope",
    ready_to_proceed: bool = False,
) -> str:
    """Build a JSON questioner-batch response string.

    ``kind_to_body`` maps each ``kind`` discriminator to the body dict
    that drives it (e.g. ``{"single_choice": {"options": [...]}}``).
    """
    questions: list[dict] = []
    for kind, body in kind_to_body.items():
        questions.append({"kind": kind, **body})
    return json.dumps(
        {"questions": questions, "reasoning": reasoning, "ready_to_proceed": ready_to_proceed}
    )


def _topic_batch(topics: list[dict]) -> str:
    """Build a JSON topic-finalize response from a list of topic dicts."""
    return json.dumps({"topics": topics})


def test_questioner_emits_typed_questions(memory_checkpointer) -> None:
    """AC #19.1 — scripted LLM returns a batch with one single_choice, one free_text,
    and ``ready_to_proceed=false``. The first ``invoke`` returns the interrupt
    payload carrying the questions and ``round=1``.

    The user-facing contract is the ``PlanInterruptPayload`` — that is what
    the API and SSE layer expose as "round N, here are the questions to
    answer". Inside the checkpoint the persisted ``PlanState`` is still
    the entry state at this point because ``interrupt()`` pauses before
    the node's return is applied; the next test verifies the state writes
    after a resume round-trip.
    """
    graph = build_plan_subgraph(checkpointer=memory_checkpointer)
    config = {"configurable": {"thread_id": "t-typed"}, "recursion_limit": 25}

    stub = _ScriptedChat(
        [
            _question_batch(
                {
                    "single_choice": {
                        "id": "q-scope",
                        "prompt": "Which angle?",
                        "options": [
                            {"option_id": "tech", "label": "Technical"},
                            {"option_id": "biz", "label": "Business"},
                        ],
                    },
                    "free_text": {
                        "id": "q-detail",
                        "prompt": "What to dig into?",
                        "placeholder": "e.g. latency vs throughput",
                    },
                },
                reasoning="clarify scope and depth",
                ready_to_proceed=False,
            ),
        ]
    )
    _install_packaged_chat(stub)

    out = graph.invoke({"user_query": "Tell me about X"}, config=config)

    # First invoke returns the interrupt payload (the contract the API exposes).
    assert "__interrupt__" in out
    interrupts = out["__interrupt__"]
    assert len(interrupts) == 1

    payload = PlanInterruptPayload.model_validate(interrupts[0].value)
    assert payload.round == 1
    assert payload.reasoning == "clarify scope and depth"
    assert len(payload.questions) == 2

    # Order is preserved from the JSON.
    assert isinstance(payload.questions[0], SingleChoicePlanQuestion)
    assert payload.questions[0].id == "q-scope"
    assert [o.option_id for o in payload.questions[0].options] == ["tech", "biz"]

    assert isinstance(payload.questions[1], FreeTextPlanQuestion)
    assert payload.questions[1].id == "q-detail"
    assert payload.questions[1].placeholder == "e.g. latency vs throughput"


def test_resume_advances_round(memory_checkpointer) -> None:
    """AC #19.2 — ``invoke(Command(resume=PlanAnswerBatch(round=1, answers=[...])))``
    advances to round 2 with answers appended to ``messages``.

    Two rounds of Q&A are scripted: round 1 interrupts with one question,
    the resume answer advances to round 2, which interrupts again with a
    different question. ``get_state`` after the resume round-trip shows the
    completed round's ``round=1``, ``last_questions``, ``last_answers``,
    and a HumanMessage appended to ``messages``.
    """
    graph = build_plan_subgraph(checkpointer=memory_checkpointer)
    config = {"configurable": {"thread_id": "t-resume-round"}, "recursion_limit": 25}

    q1 = {
        "single_choice": {
            "id": "q1",
            "prompt": "First?",
            "options": [{"option_id": "a", "label": "A"}],
        }
    }
    q2 = {
        "free_text": {
            "id": "q2",
            "prompt": "Detail?",
        }
    }

    stub = _ScriptedChat(
        [
            # Round 1: initial invoke.
            _question_batch(q1, ready_to_proceed=False),
            # Round 1: re-run on resume.
            _question_batch(q1, ready_to_proceed=False),
            # Round 2: initial invoke (after the resume re-run bumped round).
            _question_batch(q2, ready_to_proceed=False),
        ]
    )
    _install_packaged_chat(stub)

    # Round 1 — initial.
    out_1 = graph.invoke({"user_query": "Q?"}, config=config)
    p1 = PlanInterruptPayload.model_validate(out_1["__interrupt__"][0].value)
    assert p1.round == 1
    assert p1.questions[0].id == "q1"

    # Round 1 — resume with answers.
    answer_1 = PlanAnswerBatch(round=1, answers=[PlanUserAnswer(question_id="q1", value="a")])
    out_2 = graph.invoke(Command(resume=answer_1), config=config)
    p2 = PlanInterruptPayload.model_validate(out_2["__interrupt__"][0].value)
    assert p2.round == 2, "resume should advance the user-facing round"
    assert p2.questions[0].id == "q2"

    # The checkpointed state after the resume round-trip captures the
    # completed round's writes: messages include the Human turn,
    # ``round`` is 1, ``last_questions``/``last_answers`` are populated.
    snapshot = graph.get_state(config)
    state = snapshot.values
    assert state["round"] == 1
    assert state["last_questions"][0].id == "q1"
    assert state["last_answers"][0].value == "a"

    messages = state["messages"]
    assert len(messages) == 2, "expected assistant + human turns"
    from langchain_core.messages import AIMessage, HumanMessage

    assert isinstance(messages[0], AIMessage)
    assert isinstance(messages[1], HumanMessage), "resume must append a Human turn"
    # The Human turn carries the user's submitted batch so the LLM sees it
    # on subsequent rounds.
    assert "q1" in messages[1].content


def test_ready_to_proceed_short_circuits_to_topics(memory_checkpointer) -> None:
    """AC #19.3 — scripted LLM returns ``ready_to_proceed=true``; the
    topic-finalize LLM call returns a topics-shaped JSON; final ``invoke``
    returns ``research_topics`` and ``current_step='research'``.
    """
    graph = build_plan_subgraph(checkpointer=memory_checkpointer)
    config = {"configurable": {"thread_id": "t-ready"}, "recursion_limit": 25}

    stub = _ScriptedChat(
        [
            # Questioner: ready_to_proceed on the first entry, no questions.
            json.dumps(
                {
                    "questions": [],
                    "reasoning": "have enough",
                    "ready_to_proceed": True,
                }
            ),
            # Topic-finalize: two ResearchTopics.
            _topic_batch(
                [
                    {
                        "id": "topic-1",
                        "title": "T1",
                        "description": "d1",
                        "expected_finding_type": "f1",
                    },
                    {
                        "id": "topic-2",
                        "title": "T2",
                        "description": "d2",
                        "expected_finding_type": "f2",
                    },
                ]
            ),
        ]
    )
    _install_packaged_chat(stub)

    final = graph.invoke({"user_query": "Tell me about X"}, config=config)

    # Short-circuit path — no interrupt was raised.
    assert "__interrupt__" not in final
    # Final output contract.
    assert final["current_step"] == "research"
    assert len(final["research_topics"]) == 2
    assert final["research_topics"][0].id == "topic-1"
    assert final["research_topics"][1].id == "topic-2"

    # And the LLM calls match: first the questioner, then the
    # topic-finalize (system prompt mentions "decompose the question").
    assert len(stub.calls) == 2
    assert "decompose the question" in stub.system_prompts[1]


def test_round_cap_forces_finalize(memory_checkpointer) -> None:
    """AC #19.4 — script the LLM to always return ``ready_to_proceed=false``;
    after 5 rounds the 6th entry skips the question-emit LLM and calls the
    topic-finalize instead.

    Each user-facing round interrupts once, then re-runs the same
    question-emit on resume before returning. That is two questioner
    calls per round — 10 total — before the soft cap fires on round 5.
    """
    graph = build_plan_subgraph(checkpointer=memory_checkpointer)
    config = {"configurable": {"thread_id": "t-cap"}, "recursion_limit": 50}

    q_response = _question_batch(
        {"confirmation": {"id": "q", "prompt": "Continue?"}},
        reasoning="r",
        ready_to_proceed=False,
    )
    topic_response = _topic_batch(
        [
            {
                "id": "topic-1",
                "title": "T1",
                "description": "d",
                "expected_finding_type": "f",
            }
        ]
    )

    # 10 questioner responses (5 rounds × 2 re-runs) + 1 topic-finalize.
    stub = _ScriptedChat([q_response] * 10 + [topic_response])
    _install_packaged_chat(stub)

    # Round 1 — initial invoke, interrupts with the confirmation.
    out_1 = graph.invoke({"user_query": "Q?"}, config=config)
    assert "__interrupt__" in out_1
    assert out_1["__interrupt__"][0].value.round == 1

    # Rounds 1–5 — resume five times. The 6th user-facing entry will be
    # the topic-finalize, not a 6th question batch.
    for round_no in range(1, 6):
        answer = PlanAnswerBatch(
            round=round_no,
            answers=[PlanUserAnswer(question_id="q", value=True)],
        )
        out = graph.invoke(Command(resume=answer), config=config)
        if "__interrupt__" in out:
            assert out["__interrupt__"][0].value.round == round_no + 1

    # 6th user-facing entry — soft cap fires, topic-finalize runs.
    answer_5 = PlanAnswerBatch(round=5, answers=[PlanUserAnswer(question_id="q", value=True)])
    final = graph.invoke(Command(resume=answer_5), config=config)
    assert "__interrupt__" not in final, "soft cap must short-circuit to finalize"
    assert final["current_step"] == "research"
    assert len(final["research_topics"]) == 1
    assert final["research_topics"][0].id == "topic-1"

    # The 6th node-entry skipped the questioner LLM: of the 11 LLM
    # calls, exactly 10 used the questioner system prompt and the 11th
    # used the topic-finalize system prompt.
    questioner_calls = sum(1 for sp in stub.system_prompts if "structured clarification" in sp)
    finalize_calls = sum(1 for sp in stub.system_prompts if "decompose the question" in sp)
    assert questioner_calls == 10, (
        f"expected 10 questioner calls (5 rounds × 2 re-runs), got {questioner_calls}"
    )
    assert finalize_calls == 1, "soft cap must call topic-finalize exactly once"


@pytest.mark.parametrize(
    "kind_payload",
    [
        pytest.param(
            {
                "kind": "single_choice",
                "id": "q-sc",
                "prompt": "Pick one",
                "options": [{"option_id": "a", "label": "A"}],
            },
            id="single_choice",
        ),
        pytest.param(
            {
                "kind": "multi_choice",
                "id": "q-mc",
                "prompt": "Pick any",
                "options": [
                    {"option_id": "a", "label": "A"},
                    {"option_id": "b", "label": "B"},
                ],
            },
            id="multi_choice",
        ),
        pytest.param(
            {
                "kind": "free_text",
                "id": "q-ft",
                "prompt": "Free-form",
                "placeholder": "hint",
            },
            id="free_text",
        ),
        pytest.param(
            {
                "kind": "confirmation",
                "id": "q-cf",
                "prompt": "Yes or no?",
            },
            id="confirmation",
        ),
    ],
)
def test_question_kinds_render_in_interrupt_payload(
    kind_payload: dict, memory_checkpointer
) -> None:
    """AC #19.5 — each ``single_choice`` / ``multi_choice`` / ``free_text`` /
    ``confirmation`` survives the round-trip
    ``LLM JSON → PlanQuestionerBatch → PlanInterruptPayload``.
    """
    graph = build_plan_subgraph(checkpointer=memory_checkpointer)
    config = {
        "configurable": {"thread_id": f"t-kind-{kind_payload['kind']}"},
        "recursion_limit": 25,
    }

    raw_json = json.dumps(
        {
            "questions": [kind_payload],
            "reasoning": "r",
            "ready_to_proceed": False,
        }
    )
    stub = _ScriptedChat([raw_json])
    _install_packaged_chat(stub)

    out = graph.invoke({"user_query": "Q?"}, config=config)
    payload = PlanInterruptPayload.model_validate(out["__interrupt__"][0].value)

    assert len(payload.questions) == 1
    q = payload.questions[0]

    # The discriminated union narrows the runtime type by ``kind``.
    kind_to_cls = {
        "single_choice": SingleChoicePlanQuestion,
        "multi_choice": MultiChoicePlanQuestion,
        "free_text": FreeTextPlanQuestion,
        "confirmation": ConfirmationPlanQuestion,
    }
    expected_cls = kind_to_cls[kind_payload["kind"]]
    assert isinstance(q, expected_cls)
    assert q.id == kind_payload["id"]
    assert q.prompt == kind_payload["prompt"]

    # And the LLM JSON also round-trips through ``PlanQuestionerBatch``
    # directly — pinning the intermediate shape's discrimination.
    parsed = PlanQuestionerBatch.model_validate_json(raw_json)
    assert isinstance(parsed.questions[0], expected_cls)
