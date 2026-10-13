"""HTTP routes — health, run creation, run inspection, plan answer resume (issue #13),
and the Plan-stage SSE event stream (issue #14).
"""

from __future__ import annotations

import contextlib
import uuid
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.api.events import stream_plan_events
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

    All four graph-backed routes (``create_run``, ``get_run``,
    ``answer_run``, ``stream_run_events``) need the same lifespan guard —
    a missing graph means the FastAPI lifespan handler has not run (e.g.
    in tests that bypass ``app.state.graph`` assignment). Centralising
    here keeps the error message in one place.
    """
    graph = request.app.state.graph
    if graph is None:  # pragma: no cover — only when lifespan hasn't run
        raise HTTPException(status_code=503, detail="Graph not initialized")
    return graph


def _get_state_or_404(graph, config: dict[str, Any]):
    """Return the checkpoint snapshot for ``config``, or raise 404.

    ``get_run`` and ``answer_run`` need the same thread-existence guard.
    A missing thread shows up in two shapes: ``get_state`` returns
    ``None``, or — what ``MemorySaver`` does — a ``StateSnapshot`` whose
    ``values`` is an empty dict. Both mean "nothing to act on", so the
    falsy check must cover them; otherwise a resume against an unknown
    thread falls through to ``graph.invoke`` and crashes in
    ``plan_questioner`` with ``user_query is empty``. Centralising here
    keeps the error message in one place (see ``_get_graph_or_503``).
    """
    snapshot = graph.get_state(config)
    if snapshot is None or not snapshot.values:
        raise HTTPException(status_code=404, detail="Thread not found")
    return snapshot


def _mark_run_failed(graph, config: dict[str, Any]) -> None:
    """Persist ``current_step="failed"`` after a graph invocation raises.

    ADR-0008 fast-fail is preserved — the exception still propagates to
    the caller (HTTP 500) — but an open ``GET /runs/{id}/events`` stream
    polls *checkpoint state* only: without this write it could never
    observe the failure and would poll until the client disconnected
    (issue #14 AC "the stream closes when the graph fails"; parent #9
    fixes the final frame as ``plan_complete`` with empty topics).
    Best-effort: if the checkpointer itself is down there is no stream
    to notify, and the original raise still surfaces.

    Schema caveat: ``update_state`` filters writes through the graph's
    input schema — the main graph's ``State`` accepts ``current_step``
    (the production shape this helper is designed for), but a
    standalone plan subgraph (``PlanInput`` = ``user_query`` only) would
    silently drop the marker.
    """
    with contextlib.suppress(Exception):  # checkpointer itself down → nothing to notify
        graph.update_state(config, {"current_step": "failed"})


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/runs", response_model=RunCreatedResponse)
async def create_run(payload: CreateRunRequest, request: Request) -> RunCreatedResponse:
    graph = _get_graph_or_503(request)
    thread_id = str(uuid.uuid4())
    config = _build_config(thread_id)
    try:
        graph.invoke({"user_query": payload.user_query}, config=config)
    except Exception:
        _mark_run_failed(graph, config)
        raise
    return RunCreatedResponse(thread_id=thread_id)


@router.get("/runs/{thread_id}")
async def get_run(thread_id: str, request: Request) -> dict[str, Any]:
    graph = _get_graph_or_503(request)
    snapshot = _get_state_or_404(graph, _build_config(thread_id))
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
    snapshot = _get_state_or_404(graph, config)

    # ``state['round']`` is 0-indexed (incremented by ``plan_questioner``
    # after each resume). The user-facing ``PlanAnswerBatch.round`` and
    # ``PlanInterruptPayload.round`` are 1-indexed; offset here.
    persisted_round = int(snapshot.values.get("round", 0) or 0)
    answer_batch = PlanAnswerBatch(
        round=persisted_round + 1,
        answers=payload.answers,
    )
    try:
        graph.invoke(Command(resume=answer_batch), config=config)
    except Exception:
        _mark_run_failed(graph, config)
        raise
    return AnswerAcceptedResponse(thread_id=thread_id)


@router.get("/runs/{thread_id}/events")
async def stream_run_events(thread_id: str, request: Request) -> StreamingResponse:
    """Stream Plan-stage lifecycle events as Server-Sent Events (issue #14).

    v0.2 ships two event types; the other six land in v0.5:

    * ``plan_question`` — fired when the Plan subgraph enters
      ``interrupt()`` on a new round; payload is
      ``PlanInterruptPayload.model_dump()``.
    * ``plan_complete`` — fired when the Plan subgraph writes
      ``current_step="research"`` (or hits the ``"failed"`` terminal
      state); payload is ``{research_topics: list[ResearchTopic]}``.

    The wire contract — envelope, frame layout, and stream-close
    conditions — is specified in ``app.api.events`` (module docstring
    plus ``stream_plan_events``); read it there rather than duplicating
    it here. Failure note: ``create_run`` / ``answer_run`` persist
    ``current_step="failed"`` via ``_mark_run_failed`` before
    propagating the 500, which is what lets an open stream observe the
    fast-fail (ADR-0008) and close.

    The response headers disable any proxy buffering
    (``X-Accel-Buffering: no``) so events arrive live even when the
    service is fronted by nginx; ``Cache-Control: no-store`` keeps
    SSE responses out of caches that would otherwise buffer them.
    """
    graph = _get_graph_or_503(request)
    config = _build_config(thread_id)
    return StreamingResponse(
        stream_plan_events(request, graph, config, thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
