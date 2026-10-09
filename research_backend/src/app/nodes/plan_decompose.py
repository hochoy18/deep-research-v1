"""Plan node — decompose the user query into 1–5 ResearchTopics (ticket #3).

The LLM is given a typed JSON schema (``PlanTopicsResponse``) in the prompt and
its response is parsed back into the same Pydantic model, so the round-trip is
type-safe. Any LLM error propagates up — per ADR-0008 we fast-fail rather than
retry.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.models import ResearchTopic
from app.state import State
from app.tools import llm as llm_module
from app.utils import parse_llm_json

_PLAN_SYSTEM_PROMPT = (
    "You are a research planner. Given a user's research question, decompose it "
    "into 1–5 complementary ResearchTopics that, taken together, would let a "
    "writer produce a thorough Markdown Report. Each topic must be narrowly "
    "scoped enough that one or two web searches could answer it. "
    "Respond ONLY with a JSON object matching the schema below — no prose, no "
    "Markdown fences.\n\n"
    "Schema:\n"
    '{"topics":[{"id":"<slug like topic-1>","title":"...","description":"...","expected_finding_type":"..."}]}'
)


class PlanTopicsResponse(BaseModel):
    """Typed schema the LLM must satisfy when returning the topic plan."""

    topics: list[ResearchTopic] = Field(
        ...,
        description="1–5 complementary ResearchTopics covering the user's question",
    )


_PLAN_USER_TEMPLATE = "User research question:\n\n{query}\n\nProduce the JSON."


def plan_decompose(state: State) -> dict:
    """Decompose ``state["user_query"]`` into ``state["research_topics"]``."""
    user_query = state["user_query"]
    if not user_query or not user_query.strip():
        raise RuntimeError("Plan node: user_query is empty")

    chat = llm_module.get_chat_model()
    messages = [
        SystemMessage(content=_PLAN_SYSTEM_PROMPT),
        HumanMessage(content=_PLAN_USER_TEMPLATE.format(query=user_query)),
    ]
    response = chat.invoke(messages)

    payload = parse_llm_json(response.content, "Plan node")
    parsed = PlanTopicsResponse(**payload)
    topics = parsed.topics
    if not topics:
        raise RuntimeError("Plan node: LLM returned no topics")
    if len(topics) > 5:
        topics = topics[:5]

    return {
        "research_topics": topics,
        "current_step": "research",
    }