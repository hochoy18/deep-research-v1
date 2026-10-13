"""Plan subgraph — multi-round structured Q&A with interrupt() (v0.2 — issue #12).

The v0.1 single-shot ``plan_decompose`` node is gone. The plan subgraph is
now a one-node ``StateGraph`` whose state schema is ``PlanState`` and
whose IO is constrained by ``PlanInput`` / ``PlanOutput`` so the main
graph can compose it without leaking the plan stage's internals.

The single node is ``plan_questioner``. Each entry does one of:

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
``plan_questioner`` then routes back to itself while
``current_step != "research"`` — that loop body is the full HITL
round. Once ``ready_to_proceed=true`` (or the soft cap is hit) the
node sets ``current_step: "research"`` and the edge falls through to
``END``.

The checkpointer is optional; when wired into the main graph the
parent's checkpointer is shared, but the standalone case in
``examples/case_plan_hitl.py`` passes its own ``MemorySaver``.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from app.models import (
    PlanAnswerBatch,
    PlanInterruptPayload,
    PlanQuestionerBatch,
    ResearchTopic,
)
from app.state import PlanInput, PlanOutput, PlanState
from app.tools import llm as llm_module
from app.utils import parse_llm_json

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
    """Run the topic-finalize LLM and return the ``research_topics`` delta."""
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

    return {
        "research_topics": topics,
        "current_step": "research",
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
# Subgraph wiring — one node, conditional edge that loops back to itself.
# ─────────────────────────────────────────────────────────────────────────────


def _route_after_questioner(state: PlanState) -> Literal["plan_questioner", "__end__"]:
    """Loop back into ``plan_questioner`` until the node signals 'research'."""
    if state.get("current_step") == "research":
        return END
    return "plan_questioner"


def build_plan_subgraph(checkpointer=None) -> CompiledStateGraph:
    """Return a compiled plan subgraph that runs the multi-round HITL loop.

    The subgraph declares its IO via ``PlanInput`` / ``PlanOutput`` so
    the main graph can constrain which fields flow in and out at this
    stage. ``PlanState`` is the internal state schema — ``messages``,
    ``round``, ``last_questions`` and ``last_answers`` are local to the
    subgraph and never escape to the parent.

    Pass ``checkpointer`` to compile the subgraph standalone (e.g. the
    ``case_plan_hitl.py`` example). When this subgraph is added as a
    node to a parent graph that already provides a checkpointer, omit
    it and the parent's checkpointer is shared.
    """
    builder = StateGraph(PlanState, input_schema=PlanInput, output_schema=PlanOutput)
    builder.add_node("plan_questioner", plan_questioner)
    builder.set_entry_point("plan_questioner")
    builder.add_conditional_edges(
        "plan_questioner",
        _route_after_questioner,
        {"plan_questioner": "plan_questioner", END: END},
    )
    return builder.compile(checkpointer=checkpointer)


__all__ = ["PLAN_MAX_ROUNDS", "build_plan_subgraph", "plan_questioner"]