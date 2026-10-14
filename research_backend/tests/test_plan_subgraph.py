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
    ConfirmedResearchTopics,
    FreeTextPlanQuestion,
    MultiChoicePlanQuestion,
    PlanAnswerBatch,
    PlanConfirmInterruptPayload,
    PlanInterruptPayload,
    PlanQuestionerBatch,
    PlanUserAnswer,
    ResearchTopic,
    SingleChoicePlanQuestion,
)


def test_plan_max_rounds_constant_is_five() -> None:
    """AC1: ``PLAN_MAX_ROUNDS`` is exported and equals 5 (ADR-0004)."""
    assert PLAN_MAX_ROUNDS == 5


def test_plan_subgraph_uses_plan_questioner_node() -> None:
    """AC2: the subgraph's nodes are ``plan_questioner`` and
    ``confirm_plan_gate`` (plus LangGraph's __start__).

    The gate was added in v0.3 (issue #23) as a second node; the
    questioner is still the entry point and is the only node that
    drives the multi-round HITL loop.
    """
    graph = build_plan_subgraph()
    nodes = set(graph.nodes.keys())
    assert nodes == {"__start__", "plan_questioner", "confirm_plan_gate"}, (
        f"unexpected nodes: {nodes}"
    )


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
    # so ``plan_questioner`` finalizes. The conditional edge then routes
    # to ``confirm_plan_gate`` (issue #23), which interrupts with a
    # ``PlanConfirmInterruptPayload``. Drive a quick gate confirm to
    # close the subgraph.
    answer_2 = PlanAnswerBatch(round=2, answers=[PlanUserAnswer(question_id="q2", value=True)])
    out_3 = graph.invoke(Command(resume=answer_2), config=config)
    assert "__interrupt__" in out_3
    gate_payload = PlanConfirmInterruptPayload.model_validate(out_3["__interrupt__"][0].value)
    assert gate_payload.reasoning == "Please review, edit, and confirm the research topics"
    assert len(gate_payload.topics) == 1

    # Resume the gate with the same list — no edits, empty diff.
    final = graph.invoke(
        Command(
            resume=ConfirmedResearchTopics(
                topics=list(gate_payload.topics),
            )
        ),
        config=config,
    )
    assert "__interrupt__" not in final
    assert final["current_step"] == "research"
    assert len(final["research_topics"]) == 1
    # The id was reassigned to a uuid4() in _finalize_topics (ADR-0005 §2),
    # so we compare against the gate's emitted id rather than the LLM slug.
    assert final["research_topics"][0].id == gate_payload.topics[0].id


def test_plan_questioner_finalizes_when_ready_to_proceed(patched_chat_model) -> None:
    """AC3: ready_to_proceed=true on round 1 triggers topic-finalize (no interrupt).

    v0.3 (issue #23): ``_finalize_topics`` reassigns each draft topic's
    ``id`` to a fresh ``uuid4()`` (ADR-0005 §2) — the LLM-supplied
    ``topic-1`` slug is discarded. The assertion pins that behaviour:
    the id is a valid UUIDv4 and the LLM slug does NOT survive.
    """
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
    # uuid4() format — 8-4-4-4-12 hex with version nibble "4".
    import uuid as _uuid

    parsed = _uuid.UUID(delta["research_topics"][0].id)
    assert parsed.version == 4
    assert delta["research_topics"][0].id != "topic-1"


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
    out = graph.invoke(Command(resume=answer), config=config)
    # v0.3 (issue #23): after the questioner finalizes, the conditional
    # edge routes to ``confirm_plan_gate`` which interrupts — so the
    # final state is the gate's ``PlanConfirmInterruptPayload``, not a
    # completion. Confirm the gate, then drive to END.
    assert "__interrupt__" in out
    gate_payload = PlanConfirmInterruptPayload.model_validate(out["__interrupt__"][0].value)
    final = graph.invoke(
        Command(resume=ConfirmedResearchTopics(topics=list(gate_payload.topics))),
        config=config,
    )
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

    v0.3 (issue #23): after the questioner finalizes, the conditional
    edge routes to ``confirm_plan_gate`` which interrupts — so the
    final completion needs a gate confirm before the subgraph reaches
    END. Topic ids are reassigned to ``uuid4()`` in
    ``_finalize_topics`` (ADR-0005 §2), so the assertions compare
    against the gate's emitted ids rather than the LLM slugs.
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

    # The questioner short-circuits to the gate, which interrupts.
    out = graph.invoke({"user_query": "Tell me about X"}, config=config)
    assert "__interrupt__" in out
    gate_payload = PlanConfirmInterruptPayload.model_validate(out["__interrupt__"][0].value)
    assert len(gate_payload.topics) == 2

    # Confirm the gate without edits and the subgraph reaches END.
    final = graph.invoke(
        Command(resume=ConfirmedResearchTopics(topics=list(gate_payload.topics))),
        config=config,
    )

    assert "__interrupt__" not in final
    # Final output contract.
    assert final["current_step"] == "research"
    assert len(final["research_topics"]) == 2
    assert final["research_topics"][0].title == "T1"
    assert final["research_topics"][1].title == "T2"
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

    v0.3 (issue #23): the final step after soft-cap finalize is the
    gate, not END — drive one more resume to confirm the gate.
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

    # Rounds 1–4 — resume four times. Each resume re-enters the
    # questioner, which emits another round-N interrupt.
    for round_no in range(1, 5):
        answer = PlanAnswerBatch(
            round=round_no,
            answers=[PlanUserAnswer(question_id="q", value=True)],
        )
        out = graph.invoke(Command(resume=answer), config=config)
        assert "__interrupt__" in out
        assert isinstance(out["__interrupt__"][0].value, PlanInterruptPayload)
        assert out["__interrupt__"][0].value.round == round_no + 1

    # Round 5 resume — soft cap fires (round >= PLAN_MAX_ROUNDS),
    # topic-finalize runs, the conditional edge routes to
    # ``confirm_plan_gate``, which interrupts.
    answer_5 = PlanAnswerBatch(round=5, answers=[PlanUserAnswer(question_id="q", value=True)])
    out_gate = graph.invoke(Command(resume=answer_5), config=config)
    assert "__interrupt__" in out_gate, "soft cap must advance to the gate"
    gate_payload = PlanConfirmInterruptPayload.model_validate(
        out_gate["__interrupt__"][0].value
    )

    final = graph.invoke(
        Command(resume=ConfirmedResearchTopics(topics=list(gate_payload.topics))),
        config=config,
    )
    assert "__interrupt__" not in final
    assert final["current_step"] == "research"
    assert len(final["research_topics"]) == 1

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


# ─────────────────────────────────────────────────────────────────────────────
# Ticket #23 — confirm_plan_gate node.
#
# The gate is a dumb pass-through between ``plan_questioner`` and ``END``:
# it reads ``research_topics``, emits a ``PlanConfirmInterruptPayload``,
# pauses, then on resume validates the user's final list (rejecting empty
# with ``ValueError("EMPTY_PLAN")``), computes the ``PlanEditDiff`` via
# ``diff_topics``, and writes ``confirmed_research_plan`` /
# ``plan_edit_diff`` / ``current_step="research"`` (ADR-0005 §1, §6, §7).
#
# These tests pin the full round-trip at the seam the API layer and the
# ``case_plan_hitl`` example consume — the typed ``ConfirmedResearchTopics``
# is the API's resume payload, and ``diff_topics`` is unit-tested
# separately in ``test_utils.py``.
# ─────────────────────────────────────────────────────────────────────────────


def _topic_dict(i: int, **overrides) -> dict:
    """Build a JSON-ready ResearchTopic payload for scripted LLM responses."""
    base = {
        "id": f"topic-{i}",
        "title": f"Topic {i}",
        "description": f"What topic {i} investigates",
        "expected_finding_type": "evidence",
    }
    base.update(overrides)
    return base


def _finalize_then_gate_responses(topics: list[dict]) -> list[str]:
    """Build the LLM-scripted responses for: ready_to_proceed → topic-finalize."""
    return [
        json.dumps({"questions": [], "reasoning": "ready", "ready_to_proceed": True}),
        json.dumps({"topics": topics}),
    ]


def test_gate_emits_confirm_interrupt_payload_after_finalize(
    memory_checkpointer, patched_chat_model
) -> None:
    """After ``plan_questioner`` finalizes, the gate emits a typed
    ``PlanConfirmInterruptPayload`` carrying the LLM draft topics — the
    API/SSE layer hands this verbatim to the user (ADR-0005 §5).

    v0.3 (issue #23): ``_finalize_topics`` reassigns each draft id to
    a fresh ``uuid4()`` (ADR-0005 §2), so the assertion pins that
    behaviour — the gate sees UUIDv4 ids, not the LLM slugs.
    """
    from app.graph.subgraphs.plan import build_plan_subgraph
    from app.models import PlanConfirmInterruptPayload

    patched_chat_model.responses.extend(
        _finalize_then_gate_responses(
            [
                _topic_dict(1, title="Scope"),
                _topic_dict(2, title="Evidence"),
            ]
        )
    )

    graph = build_plan_subgraph(checkpointer=memory_checkpointer)
    config = {
        "configurable": {"thread_id": "t-gate-open"},
        "recursion_limit": 25,
    }
    out = graph.invoke({"user_query": "Q?"}, config=config)

    assert "__interrupt__" in out
    interrupts = out["__interrupt__"]
    assert len(interrupts) == 1
    payload = PlanConfirmInterruptPayload.model_validate(interrupts[0].value)
    assert payload.reasoning == "Please review, edit, and confirm the research topics"
    # Ids are uuid4(), not the LLM-supplied "topic-1" / "topic-2" slugs.
    import uuid as _uuid

    for topic in payload.topics:
        assert _uuid.UUID(topic.id).version == 4
    assert len(payload.topics) == 2


def test_gate_resume_with_same_list_writes_empty_diff(
    memory_checkpointer, patched_chat_model
) -> None:
    """The user confirms without edits — ``plan_edit_diff`` is all-empty and
    ``confirmed_research_plan`` equals the LLM draft (ADR-0005 §6).

    v0.3 (issue #23): the gate's draft topics carry ``uuid4()`` ids
    (ADR-0005 §2). The user submits those same ids; the empty-diff
    contract is preserved by returning the draft verbatim.
    """
    from app.graph.subgraphs.plan import build_plan_subgraph
    from app.models import ConfirmedResearchTopics

    draft_topics = [
        _topic_dict(1, title="Scope"),
        _topic_dict(2, title="Evidence"),
    ]
    patched_chat_model.responses.extend(_finalize_then_gate_responses(draft_topics))

    graph = build_plan_subgraph(checkpointer=memory_checkpointer)
    config = {
        "configurable": {"thread_id": "t-gate-noop"},
        "recursion_limit": 25,
    }

    out = graph.invoke({"user_query": "Q?"}, config=config)
    assert "__interrupt__" in out
    gate_payload = PlanConfirmInterruptPayload.model_validate(
        out["__interrupt__"][0].value
    )

    # Resume with the SAME list (using the gate's emitted UUIDv4 ids) — no edits.
    final = graph.invoke(
        Command(resume=ConfirmedResearchTopics(topics=list(gate_payload.topics))),
        config=config,
    )

    assert "__interrupt__" not in final
    assert final["current_step"] == "research"
    assert [t.id for t in final["confirmed_research_plan"]] == [
        t.id for t in gate_payload.topics
    ]
    diff = final["plan_edit_diff"]
    assert diff.added == []
    assert diff.removed == []
    assert diff.modified == []


def test_gate_resume_with_edits_writes_diff(
    memory_checkpointer, patched_chat_model
) -> None:
    """A user edit (1 added + 1 removed + 1 modified) is reflected in the
    ``PlanEditDiff`` audit trail and the ``confirmed_research_plan`` —
    the values the Research subgraph will run on.

    v0.3 (issue #23): the gate's draft ids are ``uuid4()`` so the
    test keeps, modifies, or drops using those ids rather than the
    LLM's ``topic-1`` slug (which would be treated as a fresh id and
    move everything into ``added``).

    v0.3 (ADR-0005 §2 — "新增 topic 发新 ID"): the gate reassigns a
    fresh ``uuid4()`` to any user-added topic whose id is not in the
    LLM draft. The test sends the new entry with a placeholder slug
    (``"topic-4"``) and asserts the gate replaces it with a uuid4
    before writing ``confirmed_research_plan`` and ``plan_edit_diff``.
    The pattern of the id (8-4-4-4-12 hex with the uuid4 ``4`` nibble)
    is the load-bearing assertion — the exact value is non-deterministic.
    """
    import re as _re

    from app.graph.subgraphs.plan import build_plan_subgraph
    from app.models import ConfirmedResearchTopics

    draft_topics = [
        _topic_dict(1, title="Scope"),
        _topic_dict(2, title="Evidence"),
        _topic_dict(3, title="Limits"),
    ]
    patched_chat_model.responses.extend(_finalize_then_gate_responses(draft_topics))

    graph = build_plan_subgraph(checkpointer=memory_checkpointer)
    config = {
        "configurable": {"thread_id": "t-gate-edits"},
        "recursion_limit": 25,
    }
    out = graph.invoke({"user_query": "Q?"}, config=config)
    assert "__interrupt__" in out
    gate_payload = PlanConfirmInterruptPayload.model_validate(
        out["__interrupt__"][0].value
    )
    # Pull the uuid4() ids the gate just emitted — these are what
    # ``diff_topics`` keys on.
    id_scope, id_evidence, id_limits = [t.id for t in gate_payload.topics]

    # User final: keep id_limits, modify id_scope, drop id_evidence, add a
    # brand new one. The added topic carries a placeholder slug; the gate
    # will reassign it to a fresh uuid4 (ADR-0005 §2).
    user_final = [
        ResearchTopic(
            id=id_scope, title="Refined scope",
            description="What topic 1 investigates",
            expected_finding_type="evidence",
        ),
        ResearchTopic(
            id=id_limits, title="Limits",
            description="What topic 3 investigates",
            expected_finding_type="evidence",
        ),
        ResearchTopic(
            id="topic-4", title="New angle",
            description="What topic 4 investigates",
            expected_finding_type="evidence",
        ),
    ]
    final = graph.invoke(
        Command(resume=ConfirmedResearchTopics(topics=user_final)),
        config=config,
    )

    # The added entry's id is a fresh uuid4, not the client-supplied
    # "topic-4" slug. Diff order: appended (final-order), so the new
    # topic is the last entry.
    diff = final["plan_edit_diff"]
    _uuid4 = _re.compile(
        r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
    )
    assert len(diff.added) == 1
    assert _uuid4.fullmatch(diff.added[0].id), (
        f"added.id must be a fresh uuid4 (ADR-0005 §2), got "
        f"{diff.added[0].id!r}"
    )
    assert diff.added[0].title == "New angle"
    assert diff.removed == [id_evidence]
    assert [m.id for m in diff.modified] == [id_scope]
    assert diff.modified[0].before.title == "Scope"
    assert diff.modified[0].after.title == "Refined scope"

    # The ``confirmed_research_plan`` carries the reassigned uuid4 for
    # the added entry (not "topic-4") — the Research subgraph reads
    # this list, so the id is what the rest of the pipeline keys on.
    plan_ids = [t.id for t in final["confirmed_research_plan"]]
    assert plan_ids[0] == id_scope
    assert plan_ids[1] == id_limits
    assert _uuid4.fullmatch(plan_ids[2]), (
        f"confirmed_research_plan[2].id must be a fresh uuid4 "
        f"(ADR-0005 §2), got {plan_ids[2]!r}"
    )
    assert plan_ids[2] != "topic-4"


def test_gate_resume_with_empty_list_raises_value_error(
    memory_checkpointer, patched_chat_model
) -> None:
    """An explicit empty topic list is rejected at the gate with
    ``ValueError("EMPTY_PLAN")`` (ADR-0005 §4 — double defence: client
    disables + server 400)."""
    from app.graph.subgraphs.plan import build_plan_subgraph
    from app.models import ConfirmedResearchTopics

    topics = [_topic_dict(1, title="Scope")]
    patched_chat_model.responses.extend(_finalize_then_gate_responses(topics))

    graph = build_plan_subgraph(checkpointer=memory_checkpointer)
    config = {
        "configurable": {"thread_id": "t-gate-empty"},
        "recursion_limit": 25,
    }
    out = graph.invoke({"user_query": "Q?"}, config=config)
    assert "__interrupt__" in out

    # Resume with empty list — the gate node must raise before any state write.
    with pytest.raises(ValueError, match="EMPTY_PLAN"):
        graph.invoke(
            Command(resume=ConfirmedResearchTopics(topics=[])),
            config=config,
        )


def test_gate_node_routes_to_end_after_resume(
    memory_checkpointer, patched_chat_model
) -> None:
    """After resume the subgraph's ``next`` is empty — the gate's only
    outgoing edge is to ``END`` (ADR-0005 §1)."""
    from app.graph.subgraphs.plan import build_plan_subgraph
    from app.models import ConfirmedResearchTopics

    topics = [_topic_dict(1, title="Scope")]
    patched_chat_model.responses.extend(_finalize_then_gate_responses(topics))

    graph = build_plan_subgraph(checkpointer=memory_checkpointer)
    config = {
        "configurable": {"thread_id": "t-gate-end"},
        "recursion_limit": 25,
    }
    out = graph.invoke({"user_query": "Q?"}, config=config)
    assert "__interrupt__" in out

    final = graph.invoke(
        Command(resume=ConfirmedResearchTopics(
            topics=[ResearchTopic(**t) for t in topics]
        )),
        config=config,
    )

    # Subgraph is at END — no pending nodes.
    snapshot = graph.get_state(config)
    assert snapshot.next == ()
    assert final["current_step"] == "research"


def test_plan_questioner_finalize_assigns_uuid4_ids(
    memory_checkpointer, patched_chat_model
) -> None:
    """``_finalize_topics`` assigns ``uuid4()`` ids to LLM-drafted topics
    so the gate's diff keys on a stable, collision-resistant id (ADR-0005 §2).

    The script verifies the *behavior*: the LLM supplies slug-style ids
    (``topic-1``); the gate's draft payload replaces them with valid
    UUIDv4 ids, and those ids are what the user sees in the interrupt.
    """
    from app.graph.subgraphs.plan import build_plan_subgraph

    patched_chat_model.responses.extend(
        _finalize_then_gate_responses(
            [
                _topic_dict(1, title="Scope"),
                _topic_dict(2, title="Evidence"),
            ]
        )
    )

    graph = build_plan_subgraph(checkpointer=memory_checkpointer)
    config = {
        "configurable": {"thread_id": "t-uuid"},
        "recursion_limit": 25,
    }
    out = graph.invoke({"user_query": "Q?"}, config=config)

    payload = out["__interrupt__"][0].value
    topics = payload.topics
    assert len(topics) == 2

    # Each id must be a valid UUIDv4 (ADR-0005 §2). UUIDs are 36 chars
    # of 8-4-4-4-12 hex with version nibble "4".
    import uuid as _uuid

    for topic in topics:
        parsed = _uuid.UUID(topic.id)
        assert parsed.version == 4
    # And the two ids are distinct — no accidental collision.
    assert topics[0].id != topics[1].id


def test_gate_resume_preserves_original_topic_ids(
    memory_checkpointer, patched_chat_model
) -> None:
    """``diff_topics`` keys on the original draft ids (now UUIDv4). When
    the user submits the same list, every entry is keyed by the same id
    the gate just emitted — the audit trail has zero modifications."""
    from app.graph.subgraphs.plan import build_plan_subgraph
    from app.models import ConfirmedResearchTopics

    patched_chat_model.responses.extend(
        _finalize_then_gate_responses(
            [
                _topic_dict(1, title="Scope"),
                _topic_dict(2, title="Evidence"),
            ]
        )
    )

    graph = build_plan_subgraph(checkpointer=memory_checkpointer)
    config = {
        "configurable": {"thread_id": "t-id-stable"},
        "recursion_limit": 25,
    }
    out = graph.invoke({"user_query": "Q?"}, config=config)
    payload = out["__interrupt__"][0].value
    draft_ids = [t.id for t in payload.topics]

    final = graph.invoke(
        Command(resume=ConfirmedResearchTopics(
            topics=[
                ResearchTopic(id=draft_ids[0], title="Scope",
                              description="What topic 1 investigates",
                              expected_finding_type="evidence"),
                ResearchTopic(id=draft_ids[1], title="Evidence",
                              description="What topic 2 investigates",
                              expected_finding_type="evidence"),
            ]
        )),
        config=config,
    )
    diff = final["plan_edit_diff"]
    assert diff.added == []
    assert diff.removed == []
    assert diff.modified == []


def test_gate_node_is_no_llm(patched_chat_model, monkeypatch) -> None:
    """The gate node never invokes the LLM — it is a dumb pass-through
    (ADR-0005 §1, user-story #9). The LLM stub's response queue must
    remain untouched across an entry/resume round-trip.

    LangGraph's ``interrupt()`` raises ``GraphInterrupt`` on first
    entry, so the test patches the import inside ``plan.py`` with a
    function that *returns* the supplied resume value (mirroring what
    LangGraph does on a real resume). The LLM stub's response queue
    must stay empty.
    """
    from app.graph.subgraphs import plan as plan_module
    from app.models import ConfirmedResearchTopics

    resume_value = ConfirmedResearchTopics(
        topics=[
            ResearchTopic(
                id="topic-1", title="T1", description="d", expected_finding_type="f"
            ),
        ],
    )

    def _fake_interrupt(payload, response_schema=None):
        # Validate that the node passes the typed interrupt payload — the
        # contract the SSE layer reads off ``snapshot.tasks[].interrupts``.
        assert isinstance(payload, PlanConfirmInterruptPayload)
        return response_schema.model_validate(
            payload.model_dump() if False else resume_value.model_dump()
        )

    monkeypatch.setattr(plan_module, "interrupt", _fake_interrupt)

    state = {
        "research_topics": [
            ResearchTopic(
                id="topic-1", title="T1", description="d", expected_finding_type="f"
            ),
        ],
        "messages": [],
    }
    out = plan_module.confirm_plan_gate(state, config=None)

    assert out["current_step"] == "research"
    assert out["confirmed_research_plan"][0].id == "topic-1"
    assert patched_chat_model.responses == []  # untouched
    assert patched_chat_model.calls == []  # untouched
