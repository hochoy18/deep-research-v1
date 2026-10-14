"""HTTP routes — health, run creation, run inspection, plan answer resume (issue #13),
the Plan→Research transition gate resume (issue #24), and the Plan-stage SSE event
stream (issue #14).
"""

from __future__ import annotations

import contextlib
import uuid
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.api.events import stream_plan_events
from app.models import ConfirmedResearchTopics, PlanAnswerBatch, PlanUserAnswer
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

    All five graph-backed routes (``create_run``, ``get_run``,
    ``answer_run``, ``confirm_plan``, ``stream_run_events``) need the
    same lifespan guard — a missing graph means the FastAPI lifespan
    handler has not run (e.g. in tests that bypass ``app.state.graph``
    assignment). Centralising here keeps the error message in one place.
    """
    graph = request.app.state.graph
    if graph is None:  # pragma: no cover — only when lifespan hasn't run
        raise HTTPException(status_code=503, detail="Graph not initialized")
    return graph


def _get_state_or_404(graph, config: dict[str, Any]):
    """Return the checkpoint snapshot for ``config``, or raise 404.

    ``get_run``, ``answer_run`` and ``confirm_plan`` need the same
    thread-existence guard. A missing thread shows up in two shapes:
    ``get_state`` returns ``None``, or — what ``MemorySaver`` does — a
    ``StateSnapshot`` whose ``values`` is an empty dict. Both mean
    "nothing to act on", so the falsy check must cover them; otherwise a
    resume against an unknown thread falls through to ``graph.invoke``
    and crashes in ``plan_questioner`` with ``user_query is empty``
    (or in ``confirm_plan_gate`` — issue #23 — which raises
    ``RuntimeError("state has no research_topics")``). Centralising
    here keeps the error message in one place (see ``_get_graph_or_503``).
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

    Not called for user-side rejections (e.g. the ``EMPTY_PLAN`` path in
    ``confirm_plan`` — issue #24): those are not pipeline failures and
    must not surface as ``current_step="failed"`` to an open event
    stream.

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


@router.post("/runs/{thread_id}/confirm-plan", response_model=AnswerAcceptedResponse)
async def confirm_plan(
    thread_id: str,
    payload: ConfirmedResearchTopics,
    request: Request,
) -> AnswerAcceptedResponse:
    """Resume the Plan→Research transition gate from its ``interrupt()`` pause (issue #24).

    The gate node ``confirm_plan_gate`` (issue #23) sits between
    ``plan_questioner`` and ``END`` inside the plan subgraph. Once the
    questioner has finalized ``research_topics``, the gate pauses on a
    typed ``PlanConfirmInterruptPayload`` and waits for the user's
    final topic list. This endpoint is the only mechanism that drives
    that pause forward.

    The body is validated as ``ConfirmedResearchTopics`` (issue #21) —
    a missing ``topics`` field is a Pydantic 422 (FastAPI rejects it
    before the handler runs). An explicit empty list is rejected at
    the API layer with 400 + code ``EMPTY_PLAN`` (ADR-0005 §4) so the
    gate is never reached on a payload that would fail anyway. The
    gate's own ``ValueError("EMPTY_PLAN")`` raise (issue #23) is also
    caught and mapped to the same 400 envelope as the server-side
    defence-in-depth — the client sees one error contract regardless
    of which side caught it.

    On success the route returns the same ``AnswerAcceptedResponse``
    envelope as ``POST /runs/{thread_id}/answer`` so the client can
    correlate the gate resume with the run. The actual
    ``confirmed_research_plan`` and ``plan_edit_diff`` are written by
    the gate node itself (issue #23) and surfaced via
    ``GET /runs/{thread_id}`` (ADR-0005 §6).

    The ``answer`` and ``confirm-plan`` endpoints are intentionally
    orthogonal (ADR-0005): Plan Q&A and the Plan→Research gate are two
    different user actions, not a single flow.
    """
    graph = _get_graph_or_503(request)
    config = _build_config(thread_id)
    _get_state_or_404(graph, config)  # 404 before any resume attempt

    # Client-side defence-in-depth (ADR-0005 §4): the front end already
    # disables the "confirm" button when ``topics.length < 1``; the API
    # mirrors that guard so a misbehaving client cannot advance an empty
    # plan. Short-circuit before invoking the graph so no state is
    # written.
    if not payload.topics:
        return _empty_plan_response()

    try:
        graph.invoke(Command(resume=payload), config=config)
    except ValueError as exc:
        # Server-side defence-in-depth (ADR-0005 §4): the gate node
        # (``confirm_plan_gate`` — issue #23) raises
        # ``ValueError("EMPTY_PLAN")`` for an empty list as its second
        # guard. Both paths converge on the same 400 body so the client
        # has one error contract. ``_mark_run_failed`` is intentionally
        # NOT called — an empty plan is a user error, not a pipeline
        # failure (ADR-0008 fast-fail does not apply), and writing
        # ``current_step="failed"`` would surface a spurious terminal
        # frame on any open ``GET /runs/{id}/events`` stream.
        if str(exc) == "EMPTY_PLAN":
            return _empty_plan_response()
        raise
    except Exception:
        _mark_run_failed(graph, config)
        raise
    return AnswerAcceptedResponse(thread_id=thread_id)


def _empty_plan_response() -> JSONResponse:
    """Build the 400 body for an empty ``topics`` list (ADR-0005 §4).

    Returns ``{"detail": "Empty topic list rejected", "code": "EMPTY_PLAN"}``
    with ``status_code=400`` — top-level ``detail`` and ``code`` keys,
    not the default FastAPI ``{"detail": ...}`` wrapper. Used by both
    the client-side short-circuit and the server-side
    ``ValueError("EMPTY_PLAN")`` catch so the body shape is identical
    regardless of which path triggered.
    """
    return JSONResponse(
        status_code=400,
        content={
            "detail": "Empty topic list rejected",
            "code": "EMPTY_PLAN",
        },
    )


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
