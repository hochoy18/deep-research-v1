"""Plan subgraph — multi-round structured Q&A with interrupt() (v0.2 — issue #12)
plus the Plan→Research transition gate (v0.3 — issue #23).

The v0.1 single-shot ``plan_decompose`` node is gone. The plan subgraph is
now a two-node ``StateGraph`` whose state schema is ``PlanState`` and
whose IO is constrained by ``PlanInput`` / ``PlanOutput`` so the main
graph can compose it without leaking the plan stage's internals.

The first node is ``plan_questioner``. Each entry does one of:

- if ``round >= PLAN_MAX_ROUNDS``: skip question-emit and call the
  topic-finalize LLM (soft cap forces the plan to materialize even when
  the questioner never says ``ready_to_proceed=true``);
- otherwise call the question-emit LLM (returns ``PlanQuestionerBatch``).
  If ``ready_to_proceed=true``, call topic-finalize. Otherwise append
  the assistant turn to ``messages``, write ``last_questions``, and call
  ``interrupt(PlanInterruptPayload(...))`` — the graph pauses here.

On resume (``Command(resume=PlanAnswerBatch)``) the node re-executes
from the start; the same question-emit LLM call runs again and yields
the same batch, but ``interrupt()`` now returns the answer instead of
raising. The node appends a Human turn to ``messages``, bumps ``round``,
and writes ``last_answers``. The conditional edge after
``plan_questioner`` then routes back to itself while the gate has not
opened (no ``"confirm"`` / ``"research"`` signal) — that loop body is
the full HITL round. Once ``ready_to_proceed=true`` (or the soft cap
is hit) the node finalizes (assigning each new topic a stable ``uuid4()`` id, per
ADR-0005 §2), writes ``current_step="confirm"`` — the gate-open
marker that sits between ``"plan"`` and ``"research"`` (issue #22 /
parent #20 user-story #13) — and the edge falls through to
``confirm_plan_gate``.

The second node is ``confirm_plan_gate`` — the dumb pass-through gate
between Plan and Research (ADR-0005 §1, issue #23). It is not an LLM
call: it reads ``research_topics`` from state, emits a typed
``PlanConfirmInterruptPayload`` via ``interrupt()``, then on resume
validates the user's final list (``EMPTY_PLAN`` for an empty list),
computes the ``PlanEditDiff`` audit trail via ``diff_topics``, and
writes ``confirmed_research_plan`` + ``plan_edit_diff`` +
``current_step="research"`` + ``messages`` so the main graph can
advance to the Research subgraph. The only outgoing edge is to
``END``.

The checkpointer is optional; when wired into the main graph the
parent's checkpointer is shared, but the standalone case in
``examples/case_plan_hitl.py`` passes its own ``MemorySaver``.
"""

from __future__ import annotations

import uuid
from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from app.models import (
    ConfirmedResearchTopics,
    PlanAnswerBatch,
    PlanConfirmInterruptPayload,
    PlanInterruptPayload,
    PlanQuestionerBatch,
    ResearchTopic,
)
from app.state import PlanInput, PlanOutput, PlanState
from app.tools import llm as llm_module
from app.utils import diff_topics, parse_llm_json

# ─────────────────────────────────────────────────────────────────────────────
# Constants — the soft cap on Q&A rounds (ADR-0004).
# ─────────────────────────────────────────────────────────────────────────────

PLAN_MAX_ROUNDS = 5

# ─────────────────────────────────────────────────────────────────────────────
# Prompts — the questioner asks 1–4 typed questions per round or signals
# ready_to_proceed; the topic-finalizer turns the dialogue into 1–5 topics.
# ─────────────────────────────────────────────────────────────────────────────

_QUESTIONER_SYSTEM_PROMPT = (
    "You are a research planner conducting a structured clarification "
    "interview with the user. Given the user's research question and the "
    "prior dialogue, decide whether to ask more questions or proceed.\n"
    "When you need more from the user, ask 1–4 typed questions. Every "
    "question object MUST include a 'kind' field set to exactly one of "
    "the four string literals below — this is the discriminator that "
    "drives the schema, so omitting it is a hard error:\n"
    "  - kind='single_choice' (one option is chosen):\n"
    "      {kind:'single_choice', id:'<stable-id>', prompt:'<text>', "
    "options:[{option_id:'a', label:'A'}, ...]}\n"
    "  - kind='multi_choice' (any subset is chosen): same shape as "
    "single_choice, but the user may select zero or more options.\n"
    "  - kind='free_text' (free-form string):\n"
    "      {kind:'free_text', id:'<stable-id>', prompt:'<text>', "
    "placeholder:'<optional hint>'}\n"
    "  - kind='confirmation' (yes/no):\n"
    "      {kind:'confirmation', id:'<stable-id>', prompt:'<text>'}\n"
    "Every question also needs a stable 'id' (used to match the user's "
    "reply) and a 'prompt'. When you have enough information to plan "
    "the research, set ready_to_proceed=true and leave 'questions' "
    "empty. Do NOT invent kinds other than the four above.\n"
    "Respond ONLY with a JSON object matching the schema below — no "
    "prose, no Markdown fences.\n\n"
    "Schema:\n"
    '{"questions": [<question objects, each with kind + id + prompt + '
    "kind-specific fields>], "
    '"reasoning": "<why you chose to ask or proceed>", '
    '"ready_to_proceed": <bool>}'
)

_FINALIZE_SYSTEM_PROMPT = (
    "You are a research planner. Given the user's research question and "
    "the prior clarification dialogue, decompose the question into 1–5 "
    "complementary ResearchTopics that, taken together, would let a "
    "writer produce a thorough Markdown Report. Each topic must be "
    "narrowly scoped enough that one or two web searches could answer "
    "it. Respond ONLY with a JSON object matching the schema below — no "
    "prose, no Markdown fences.\n\n"
    "Schema:\n"
    '{"topics":[{"id":"<slug like topic-1>","title":"...","description":"...","expected_finding_type":"..."}]}'
)

_QUESTIONER_USER_TEMPLATE = (
    "User research question:\n{user_query}\n\n"
    "Prior clarification dialogue (currently on round {round_label}):\n"
    "{history}\n\n"
    "Produce the JSON."
)

_FINALIZE_USER_TEMPLATE = (
    "User research question:\n{user_query}\n\n"
    "Prior clarification dialogue:\n{history}\n\n"
    "Produce the JSON of topics."
)


class PlanTopicsResponse(BaseModel):
    """Typed schema the topic-finalize LLM must satisfy."""

    topics: list[ResearchTopic] = Field(
        ...,
        description="1–5 complementary ResearchTopics covering the user's question",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Node — plan_questioner covers the full HITL loop body.
# ─────────────────────────────────────────────────────────────────────────────


def plan_questioner(state: PlanState, config) -> dict:
    """Plan HITL loop body — emits, interrupts, resumes, and finalizes.

    See module docstring for the per-entry behaviour. The function is
    stateless across entries: every entry reads ``state`` afresh, calls
    the questioner LLM if there's room for another round, and either
    interrupts or finalizes. On resume the only thing that changes is
    that ``interrupt()`` returns the answer instead of raising.
    """
    user_query = state.get("user_query", "")
    if not user_query or not user_query.strip():
        raise RuntimeError("Plan questioner: user_query is empty")

    round_idx = state.get("round", 0)
    messages = list(state.get("messages") or [])

    if round_idx < PLAN_MAX_ROUNDS:
        batch = _call_questioner_llm(user_query, messages, round_idx)
    else:
        # Soft cap: skip the question-emit LLM, go straight to topic-finalize.
        batch = None

    if batch is None or batch.ready_to_proceed:
        return _finalize_topics(user_query, messages)

    # Append an assistant turn describing the questions, then interrupt.
    messages.append(AIMessage(content=_serialize_question_batch(batch)))

    interrupt_payload = PlanInterruptPayload(
        questions=batch.questions,
        reasoning=batch.reasoning,
        round=round_idx + 1,  # 1-based for the user-facing payload
    )
    answer_batch = interrupt(interrupt_payload, response_schema=PlanAnswerBatch)

    # --- resume path only --- the code below this line is unreachable on
    # the initial entry because ``interrupt()`` raised ``GraphInterrupt``.
    messages.append(HumanMessage(content=_serialize_answer_batch(answer_batch)))

    return {
        "messages": messages,
        "round": round_idx + 1,
        "last_questions": batch.questions,
        "last_answers": answer_batch.answers,
    }


def _call_questioner_llm(
    user_query: str,
    messages: list[BaseMessage],
    round_idx: int,
) -> PlanQuestionerBatch:
    """Run the question-emit LLM and parse the typed ``PlanQuestionerBatch``."""
    chat = llm_module.get_chat_model()
    user_text = _QUESTIONER_USER_TEMPLATE.format(
        user_query=user_query,
        round_label=round_idx + 1,  # 1-based for the LLM
        history=_format_history(messages) or "(none yet)",
    )
    response = chat.invoke(
        [
            SystemMessage(content=_QUESTIONER_SYSTEM_PROMPT),
            HumanMessage(content=user_text),
        ]
    )
    payload = parse_llm_json(response.content, "Plan questioner")
    return PlanQuestionerBatch.model_validate(payload)


def _finalize_topics(user_query: str, messages: list[BaseMessage]) -> dict:
    """Run the topic-finalize LLM and return the ``research_topics`` delta.

    Per ADR-0005 §2 each freshly proposed topic receives a stable
    ``uuid4()`` id so the gate's ``diff_topics`` (and any future audit
    trail) joins on a collision-resistant key. The LLM's own
    ``topic-N`` slug pattern is discarded — the ``id`` field on the
    schema is required, but its *value* is the gate's contract, not
    the LLM's. Existing topics (carried over from a previous plan
    invocation against the same thread) keep their id; here all topics
    are freshly drafted so every id is a new ``uuid4()``.

    ``current_step`` is written as ``"confirm"`` — the gate-open
    marker (issue #22). The finalize itself has NOT completed the Plan
    stage; the user still has to review / edit / confirm the list at
    the gate. ``"research"`` is written by ``confirm_plan_gate`` on
    resume only, so the SSE detector (issue #25) can distinguish a
    gate-open pause from a genuinely completed Plan stage.
    """
    chat = llm_module.get_chat_model()
    user_text = _FINALIZE_USER_TEMPLATE.format(
        user_query=user_query,
        history=_format_history(messages) or "(none)",
    )
    response = chat.invoke(
        [
            SystemMessage(content=_FINALIZE_SYSTEM_PROMPT),
            HumanMessage(content=user_text),
        ]
    )
    payload = parse_llm_json(response.content, "Plan topic-finalize")
    parsed = PlanTopicsResponse.model_validate(payload)
    topics = parsed.topics
    if not topics:
        raise RuntimeError("Plan topic-finalize: LLM returned no topics")
    if len(topics) > 5:
        topics = topics[:5]

    # ADR-0005 §2 — reassign each id to a fresh uuid4(). ResearchTopic is
    # frozen, so we rebuild the list rather than mutating in place.
    topics = [
        ResearchTopic(
            id=str(uuid.uuid4()),
            title=topic.title,
            description=topic.description,
            expected_finding_type=topic.expected_finding_type,
        )
        for topic in topics
    ]

    return {
        "research_topics": topics,
        "current_step": "confirm",
        "messages": messages,
    }


def _format_history(messages: list[BaseMessage]) -> str:
    """Render the ``messages`` list as a human-readable transcript."""
    parts: list[str] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            parts.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            parts.append(f"Assistant: {msg.content}")
        else:
            parts.append(f"{type(msg).__name__}: {msg.content}")
    return "\n\n".join(parts)


def _serialize_question_batch(batch: PlanQuestionerBatch) -> str:
    """Serialize a ``PlanQuestionerBatch`` for the assistant turn's content.

    The LLM sees this on subsequent rounds as the prior assistant turn,
    so we keep the wire shape intact rather than prose-fying it.
    """
    return batch.model_dump_json()


def _serialize_answer_batch(batch: PlanAnswerBatch) -> str:
    """Serialize a ``PlanAnswerBatch`` for the human turn's content."""
    return batch.model_dump_json()


# ─────────────────────────────────────────────────────────────────────────────
# Gate node — dumb pass-through between Plan and Research (issue #23).
# ─────────────────────────────────────────────────────────────────────────────


# Why the user sees this exact copy. The API and SSE layer forward this
# verbatim — the front end renders it above the editable table — so a
# future tweak should be coordinated with the front end (see ADR-0005 §5).
_GATE_REASONING = "Please review, edit, and confirm the research topics"


def confirm_plan_gate(state: PlanState, config) -> dict:
    """Dumb pass-through gate between ``plan_questioner`` and ``END`` (issue #23).

    On entry the node reads ``state["research_topics"]`` (the LLM draft
    ``plan_questioner._finalize_topics`` just wrote), wraps it in a
    ``PlanConfirmInterruptPayload``, and calls ``interrupt()`` —
    LangGraph raises ``GraphInterrupt`` here on the initial run and the
    graph pauses. ``State`` at the pause point carries
    ``current_step="confirm"`` from ``plan_questioner``'s finalize —
    the gate-open marker (issue #22). The gate does not touch
    ``current_step`` until the resume path runs.

    On resume the user submits a typed ``ConfirmedResearchTopics`` via
    ``Command(resume=...)``. The node:

    1. rejects an empty list with ``ValueError("EMPTY_PLAN")`` — the
       double-defence seam per ADR-0005 §4 (the API also 400s on this,
       but the gate must fail closed too);
    2. computes the ``PlanEditDiff`` via ``diff_topics`` against the
       LLM draft that was paused (ADR-0005 §7);
    3. writes ``confirmed_research_plan``, ``plan_edit_diff``,
       ``current_step="research"`` (the gate-open marker's successor —
       the step advance that moves the pipeline into the Research
       subgraph), and ``messages`` (no-op — preserved from the
       questioner's last write).

    The node never invokes an LLM (ADR-0005 §1, user-story #9) and has
    no outgoing edge other than to ``END`` — there is no re-loop back
    to ``plan_questioner`` (ADR-0005 §4).
    """
    topics = state.get("research_topics") or []
    if not topics:
        # No draft topics means plan_questioner finalized with an empty
        # list — that path already raises RuntimeError, so reaching
        # here is a programmer error rather than a user error.
        raise RuntimeError(
            "confirm_plan_gate: state has no research_topics; "
            "plan_questioner should have raised on an empty finalize"
        )

    payload = PlanConfirmInterruptPayload(
        topics=topics,
        reasoning=_GATE_REASONING,
    )
    # First call raises GraphInterrupt; on resume it returns the user's
    # ``ConfirmedResearchTopics`` payload (issue #23 AC).
    confirmed = interrupt(payload, response_schema=ConfirmedResearchTopics)

    # --- resume path only -------------------------------------------------
    if len(confirmed.topics) < 1:
        # Empty list is the API-layer's defence-in-depth partner — both
        # the client UI (button disabled) and this server-side guard
        # reject. Surfacing as ``ValueError("EMPTY_PLAN")`` so the API
        # layer can map it to a 400 with the agreed code.
        raise ValueError("EMPTY_PLAN")

    # ADR-0005 §2 — "原 topic 的 ID 保留, 新增 topic 发新 ID" (existing
    # topic ids are preserved; newly-added topics get a fresh uuid4()).
    # ``_finalize_topics`` already reassigns uuid4() to every freshly
    # drafted topic; the gate does the same for the user's edits, so
    # ``confirmed_research_plan`` carries stable ids across the gate
    # boundary. We run the diff against the LLM draft vs the
    # post-reassignment final list so the audit trail's ``added`` ids
    # match the ids the Research subgraph actually sees.
    reassigned = _reassign_added_topic_ids(list(topics), list(confirmed.topics))
    diff = diff_topics(list(topics), reassigned)

    return {
        "confirmed_research_plan": reassigned,
        "plan_edit_diff": diff,
        "current_step": "research",
        "messages": list(state.get("messages") or []),
    }


def _reassign_added_topic_ids(
    draft: list[ResearchTopic], final: list[ResearchTopic]
) -> list[ResearchTopic]:
    """Return ``final`` with a fresh ``uuid4()`` for any topic whose id is
    not in ``draft`` (ADR-0005 §2 — "新增 topic 发新 ID").

    Topics whose id is already in ``draft`` are kept verbatim (frozen
    ResearchTopic, so the rebuild preserves the rest of the fields).
    The audit trail (``diff_topics``) is then computed against the
    LLM draft vs the post-reassignment final list so the ``added``
    entries surface the same uuid4 ids the Research subgraph
    eventually runs on.
    """
    draft_ids = {t.id for t in draft}
    return [
        ResearchTopic(
            id=str(uuid.uuid4()) if topic.id not in draft_ids else topic.id,
            title=topic.title,
            description=topic.description,
            expected_finding_type=topic.expected_finding_type,
        )
        for topic in final
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Subgraph wiring — two nodes: ``plan_questioner`` loops on itself; once it
# finalizes the conditional edge routes to ``confirm_plan_gate``, whose
# only outgoing edge is to ``END``.
# ─────────────────────────────────────────────────────────────────────────────


def _route_after_questioner(state: PlanState) -> Literal["plan_questioner", "confirm_plan_gate"]:
    """Loop back into ``plan_questioner`` until the node signals the gate,
    then forward to ``confirm_plan_gate`` (issue #23 AC).

    v0.3 (issue #22 / #26): ``plan_questioner._finalize_topics`` writes
    ``current_step="confirm"`` — the gate-open marker — so the forward
    signal is ``"confirm"``. ``"research"`` is the gate's own post-resume
    write and never appears while this router runs, so it is not (and
    must not be) an accepted forward signal here.
    """
    if state.get("current_step") == "confirm":
        return "confirm_plan_gate"
    return "plan_questioner"


def build_plan_subgraph(checkpointer=None) -> CompiledStateGraph:
    """Return a compiled plan subgraph that runs the multi-round HITL loop
    and the Plan→Research transition gate (issue #23).

    The subgraph declares its IO via ``PlanInput`` / ``PlanOutput`` so
    the main graph can constrain which fields flow in and out at this
    stage. ``PlanState`` is the internal state schema — ``messages``,
    ``round``, ``last_questions`` and ``last_answers`` are local to the
    subgraph and never escape to the parent; ``confirmed_research_plan``
    and ``plan_edit_diff`` DO escape via ``PlanOutput`` (issue #22) so
    the Research subgraph reads from the user's final list, not the
    LLM's draft.

    Pass ``checkpointer`` to compile the subgraph standalone (e.g. the
    ``case_plan_hitl.py`` example). When this subgraph is added as a
    node to a parent graph that already provides a checkpointer, omit
    it and the parent's checkpointer is shared.
    """
    builder = StateGraph(PlanState, input_schema=PlanInput, output_schema=PlanOutput)
    builder.add_node("plan_questioner", plan_questioner)
    builder.add_node("confirm_plan_gate", confirm_plan_gate)
    builder.set_entry_point("plan_questioner")
    builder.add_conditional_edges(
        "plan_questioner",
        _route_after_questioner,
        {
            "plan_questioner": "plan_questioner",
            "confirm_plan_gate": "confirm_plan_gate",
        },
    )
    builder.add_edge("confirm_plan_gate", END)
    return builder.compile(checkpointer=checkpointer)


__all__ = [
    "PLAN_MAX_ROUNDS",
    "build_plan_subgraph",
    "confirm_plan_gate",
    "plan_questioner",
]