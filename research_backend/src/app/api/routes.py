"""HTTP routes — health, run creation, run inspection, plan answer resume (issue #13)."""

from __future__ import annotations

import uuid
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.models import PlanAnswerBatch, PlanUserAnswer
from app.nodes.research_run import RECURSION_LIMIT

router = APIRouter()


class CreateRunRequest(BaseModel):
    user_query: str = Field(..., min_length=1)


class RunCreatedResponse(BaseModel):
    thread_id: str


class PlanAnswerSubmission(BaseModel):
    """Body of ``POST /runs/{thread_id}/answer`` (issue #13).

    ``round`` is intentionally absent: the route fills in the
    user-facing 1-based round from the checkpointer so the client never
    has to track it. ``answers`` is required — a missing ``answers``
    field is a Pydantic 422 (matches issue #13's acceptance criteria).
    The list may be empty (no questions asked this round); each item
    carries ``question_id`` + a typed ``value``
    (``str | list[str] | bool``). The Plan subgraph re-emits questions
    for unknown ``question_id``s (v0.2 does not enforce set membership),
    so no semantic validation is performed here.
    """

    answers: list[PlanUserAnswer] = Field(
        ...,
        description="User's answers to the round's questions (typed ``PlanUserAnswer``s)",
    )


class AnswerAcceptedResponse(BaseModel):
    thread_id: str
    accepted: Literal[True] = True


def _build_config(thread_id: str) -> dict[str, Any]:
    """RunnableConfig used for both ``invoke`` and ``get_state`` calls."""
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
    }


def _to_jsonable(value: Any) -> Any:
    """Coerce a single value (Pydantic model / container / primitive) to JSON-friendly form."""
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    return value


def _get_graph_or_503(request: Request):
    """Return the compiled graph bound to ``app.state``, or raise 503.

    Three routes (``create_run``, ``get_run``, ``answer_run``) all need
    the same lifespan guard — a missing graph means the FastAPI lifespan
    handler has not run (e.g. in tests that bypass ``app.state.graph``
    assignment). Centralising here keeps the error message in one place.
    """
    graph = request.app.state.graph
    if graph is None:  # pragma: no cover — only when lifespan hasn't run
        raise HTTPException(status_code=503, detail="Graph not initialized")
    return graph


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/runs", response_model=RunCreatedResponse)
async def create_run(payload: CreateRunRequest, request: Request) -> RunCreatedResponse:
    graph = _get_graph_or_503(request)
    thread_id = str(uuid.uuid4())
    graph.invoke({"user_query": payload.user_query}, config=_build_config(thread_id))
    return RunCreatedResponse(thread_id=thread_id)


@router.get("/runs/{thread_id}")
async def get_run(thread_id: str, request: Request) -> dict[str, Any]:
    graph = _get_graph_or_503(request)
    snapshot = graph.get_state(_build_config(thread_id))
    if snapshot is None or snapshot.values is None:
        raise HTTPException(status_code=404, detail="Thread not found")

    return _to_jsonable(snapshot.values)


@router.post("/runs/{thread_id}/answer", response_model=AnswerAcceptedResponse)
async def answer_run(
    thread_id: str,
    payload: PlanAnswerSubmission,
    request: Request,
) -> AnswerAcceptedResponse:
    """Submit a round of answers and resume the graph from its ``interrupt()`` pause.

    Issue #13 — the only mechanism that drives the Plan HITL loop forward
    once the questioner has emitted a batch. Reads the persisted round
    from the checkpointer, wraps the body in a typed ``PlanAnswerBatch``,
    and invokes ``graph.invoke(Command(resume=...))`` against the
    thread's config. Returns the same envelope ``/runs`` returns so
    clients can correlate the resume with the run.

    FastAPI raises 422 for any Pydantic schema failure (missing
    ``answers``, wrong ``question_id`` type, non-``str | list[str] | bool``
    ``value``) before this handler runs, so a malformed body never reaches
    the graph.
    """
    graph = _get_graph_or_503(request)
    config = _build_config(thread_id)
    snapshot = graph.get_state(config)
    if snapshot is None or snapshot.values is None:
        raise HTTPException(status_code=404, detail="Thread not found")

    # ``state['round']`` is 0-indexed (incremented by ``plan_questioner``
    # after each resume). The user-facing ``PlanAnswerBatch.round`` and
    # ``PlanInterruptPayload.round`` are 1-indexed; offset here.
    persisted_round = int(snapshot.values.get("round", 0) or 0)
    answer_batch = PlanAnswerBatch(
        round=persisted_round + 1,
        answers=payload.answers,
    )
    graph.invoke(Command(resume=answer_batch), config=config)
    return AnswerAcceptedResponse(thread_id=thread_id)
