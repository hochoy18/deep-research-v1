"""SSE event generation for the Plan stage (issue #14) + the Plan→Research
transition gate (issue #23 / #25).

Owns the polling loop that drives ``GET /runs/{thread_id}/events``.
The route is the only consumer; the generator is a self-contained
async iterable that yields ``bytes`` in the SSE wire format.

The forward-compatible envelope (issue #14 / spec user-story #16)
ships with three event types in v0.3; the remaining five land in v0.5:

* ``plan_question`` — fired when the Plan subgraph enters
  ``interrupt()`` on a new round. Payload is
  ``PlanInterruptPayload.model_dump()`` (``{questions, reasoning,
  round}``).
* ``plan_confirm_required`` — fired when the Plan→Research transition
  gate (ADR-0005) enters ``interrupt()`` with the LLM draft. Payload
  is ``PlanConfirmInterruptPayload.model_dump()``
  (``{topics, reasoning}``).
* ``plan_complete`` — fired when the Plan subgraph writes
  ``current_step="research"`` (or hits the ``"failed"`` terminal
  state). Payload is ``{research_topics: list[ResearchTopic]}``;
  empty list on failure. The gate-open marker ``"confirm"`` does NOT
  fire it — the Plan stage is only complete once the user has
  confirmed their final topic list (issue #26).

The wire shape is locked for v0.5 forward-compat::

    event: <name>
    data: {"thread_id": "<uuid>", "ts": <unix-ms>, "payload": {...}}
    <blank line>

The ``data`` field is a single JSON object, so embedded newlines in
``payload`` strings must already be JSON-escaped by ``json.dumps`` —
``_format_event`` does NOT insert extra whitespace per SSE lines.

The stream closes when any of these is observed (issue #14 AC):

* the main graph reaches ``END`` (``current_step == "final"``);
* the graph fails (``current_step == "failed"``);
* the client disconnects (``await request.is_disconnected()``);
* the thread is not in the checkpointer — real checkpointers
  (``MemorySaver`` / ``RedisSaver``) report an unknown thread as a
  snapshot with empty ``values`` and no ``tasks``, never ``None``
  (the ``None`` shape is what test doubles use). Deliberately
  slightly narrower than the 404 guard ``_get_state_or_404`` in
  ``routes.py``, which fires on empty ``values`` alone: a thread
  with pending tasks but no state keeps its stream open (the
  interrupt on the task is still worth emitting) instead of
  closing.

Failure semantics ("the graph fails"): the graph code itself never
writes ``current_step="failed"`` — ADR-0008 fast-fail raises. The API
routes persist it (``_mark_run_failed``) *before* re-raising, so an
open stream sees the terminal state, emits the empty ``plan_complete``,
and closes. Topic finalization guarantees >= 1 topic on success
(``plan_questioner`` raises otherwise), so an empty payload list is
unambiguously the failure signal in v0.2.

In v0.2 the connection stays open between ``plan_complete`` and the
final terminal event (no research / write events ship yet); operators
can rely on the connection kept-alive via TCP rather than explicit
pings, since a long-idle SSE stream is allowed by the spec.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import Request
from pydantic import BaseModel

from app.models import PlanConfirmInterruptPayload, PlanInterruptPayload, ResearchTopic

# Default polling cadence — 5 Hz is fast enough for live UX without
# burning CPU on a long-poll. Tests monkeypatch this constant so they
# don't have to wait 200 ms per loop iteration.
SSE_POLL_INTERVAL_SECONDS = 0.2


def _now_ms() -> int:
    """Return the current Unix timestamp in milliseconds (issue #14 envelope)."""
    return int(time.time() * 1000)


def _unwrap_lc_envelope(value: dict) -> dict:
    """Return the payload dict out of a LangChain constructor envelope, if wrapped.

    A real ``RedisSaver`` (msgpack) round-trip of a not-registered class
    comes back as e.g. ``{'lc': 2, 'type': 'constructor', 'id': [...],
    'kwargs': {...}}`` — the payload fields live under ``kwargs`` (lc v2)
    or ``default`` (lc v1). Plain dicts pass through untouched.
    """
    if value.get("lc") in (1, 2) and value.get("type") == "constructor":
        return value.get("kwargs") or value.get("default") or {}
    return value


def _interrupt_payload_dict(value: Any) -> dict | None:
    """Coerce an ``Interrupt.value`` to a ``PlanInterruptPayload.model_dump()`` dict.

    The LangGraph checkpointer deserializes the interrupt value back
    into a typed ``PlanInterruptPayload`` (Pydantic). When
    deserialization fails the value comes back as a plain dict — guard
    both paths so the wire format is the same either way. Returns
    ``None`` if the value can't be coerced (the caller skips the emit
    rather than failing the stream).
    """
    if isinstance(value, PlanInterruptPayload):
        return value.model_dump()
    if isinstance(value, BaseModel):
        # Some other Pydantic model — round-trip through the typed
        # payload to narrow the discriminator and discard unknown keys.
        return PlanInterruptPayload.model_validate(value.model_dump()).model_dump()
    if isinstance(value, dict):
        # Covers both the plain dict and the LangChain constructor
        # envelope the Redis checkpointer really returns — unwrap first
        # so both shapes validate identically.
        return PlanInterruptPayload.model_validate(_unwrap_lc_envelope(value)).model_dump()
    return None


def _gate_payload_dict(value: Any) -> dict | None:
    """Coerce an ``Interrupt.value`` to a ``PlanConfirmInterruptPayload.model_dump()`` dict.

    Mirror of ``_interrupt_payload_dict`` for the Plan→Research
    transition gate (ADR-0005). The discriminator vs the questioner
    payload is the presence of the ``topics`` field (the gate
    payload) vs ``questions`` (the questioner payload) — see issue
    #25. The same Redis-envelope / Pydantic fallback ladder applies
    because the gate and the questioner share the ``Interrupt``
    channel — a deserializer failure on either path must not crash
    the stream.
    """
    if isinstance(value, PlanConfirmInterruptPayload):
        return value.model_dump()
    if isinstance(value, BaseModel):
        return PlanConfirmInterruptPayload.model_validate(value.model_dump()).model_dump()
    if isinstance(value, dict):
        return PlanConfirmInterruptPayload.model_validate(_unwrap_lc_envelope(value)).model_dump()
    return None


def _is_gate_interrupt_value(value: Any) -> bool:
    """True iff ``value`` looks like a ``PlanConfirmInterruptPayload`` (gate), not a
    questioner ``PlanInterruptPayload``.

    The discriminator is the ``topics`` field — only the gate payload
    carries it; the questioner payload carries ``questions`` instead.
    A ``BaseModel`` is identified by its concrete class, falling back
    to the field-shape check for the dict / Redis-envelope case.
    """
    if isinstance(value, PlanConfirmInterruptPayload):
        return True
    if isinstance(value, PlanInterruptPayload):
        return False
    if isinstance(value, BaseModel):
        return hasattr(value, "topics")
    if isinstance(value, dict):
        return "topics" in value and "questions" not in value
    return False


def _research_topics_dict(topics: Any) -> list[dict]:
    """Coerce ``research_topics`` to a JSON-friendly ``list[dict]``.

    Real LangGraph returns ``list[ResearchTopic]``; mock graphs in
    tests may return either typed models or pre-dumped dicts. Either
    path must yield the same SSE payload so clients don't see a
    schema drift between unit and integration tests.
    """
    out: list[dict] = []
    for t in topics or []:
        if isinstance(t, ResearchTopic):
            out.append(t.model_dump())
        elif isinstance(t, dict):
            # Same Redis envelope caveat as ``_interrupt_payload_dict``.
            out.append(ResearchTopic.model_validate(_unwrap_lc_envelope(t)).model_dump())
        elif isinstance(t, BaseModel):
            out.append(t.model_dump())
    return out


def _format_event(event_name: str, thread_id: str, payload: dict) -> bytes:
    """Render one SSE event envelope per the issue #14 contract.

    The output is exactly::

        event: <name>\\n
        data: {"thread_id": "...", "ts": <ms>, "payload": {...}}\\n
        \\n

    Compact separators (``,`` ``:``) keep the wire payload minimal —
    SSE clients parse the whole ``data:`` line as a single JSON
    blob, and ``json.dumps`` handles any embedded newlines that
    happen to land inside the payload.
    """
    envelope = {
        "thread_id": thread_id,
        "ts": _now_ms(),
        "payload": payload,
    }
    data = json.dumps(envelope, separators=(",", ":"))
    return f"event: {event_name}\ndata: {data}\n\n".encode()


async def stream_plan_events(
    request: Request,
    graph: Any,
    config: dict[str, Any],
    thread_id: str,
) -> AsyncIterator[bytes]:
    """Yield SSE bytes for the lifetime of a client connection.

    The generator polls ``graph.get_state(config)`` (in the default
    thread executor, so a synchronous checkpointer can't stall the
    event loop) and yields events when state changes. State changes
    are tracked by:

    * Interrupt id set — one ``plan_question`` per new ``Interrupt.id``.
      Re-emitting the same interrupt would dup the event on the wire.
    * A boolean — at most one ``plan_complete`` *per connection*. A
      reconnecting client gets a fresh generator instance, which may
      replay it from the checkpoint once the stage has advanced
      (late-join).

    The ``graph`` parameter is typed as ``Any`` because the real
    graph (``CompiledStateGraph``) and the mock graphs in tests do
    not share an interface beyond ``get_state(config)``. The duck
    typing keeps the test seam lightweight.
    """
    seen_interrupt_ids: set[str] = set()
    plan_complete_emitted = False

    while True:
        if await request.is_disconnected():
            return

        snapshot = await asyncio.to_thread(graph.get_state, config)
        if snapshot is None or (not snapshot.values and not snapshot.tasks):
            # Unknown thread — real checkpointers return an empty
            # StateSnapshot (never None) for thread ids they have
            # never seen; close instead of polling forever. (The 404
            # guard in ``routes.py`` is stricter: empty ``values``
            # alone trips it, with or without tasks.)
            return

        # Step 1 — emit plan_question / plan_confirm_required for every
        # newly-seen interrupt. Two interrupt types live on the same
        # channel: the questioner emits ``PlanInterruptPayload`` (per
        # round) and the gate emits ``PlanConfirmInterruptPayload``
        # (Plan→Research transition, ADR-0005). The discriminator is
        # the payload's field shape — see ``_is_gate_interrupt_value``.
        for task in snapshot.tasks or ():
            for interrupt in task.interrupts or ():
                if interrupt.id in seen_interrupt_ids:
                    continue
                seen_interrupt_ids.add(interrupt.id)
                if _is_gate_interrupt_value(interrupt.value):
                    payload = _gate_payload_dict(interrupt.value)
                    event_name = "plan_confirm_required"
                else:
                    payload = _interrupt_payload_dict(interrupt.value)
                    event_name = "plan_question"
                if payload is None:
                    # Defensive: skip if the interrupt's value isn't a
                    # recognised shape rather than crashing the stream.
                    continue
                yield _format_event(event_name, thread_id, payload)

        # Step 2 — emit plan_complete exactly once when the Plan stage
        # has advanced past ``"plan"`` (``"research"`` / ``"write"`` /
        # ``"final"`` / ``"failed"``). ``"confirm"`` is deliberately
        # excluded: it is the gate-open marker (issue #22), not a
        # completion — firing here would tell the client the Plan
        # stage is done while the user is still reviewing the topic
        # list. The event fires only after the gate resumes and the
        # gate node writes ``current_step="research"`` (issue #26).
        # A late-joining client (one that connects after the stage
        # already advanced) sees it as its first event rather than a
        # silent close — the event is the stage's handshake and the
        # checkpointer makes it replayable. The ``"failed"`` path
        # carries an empty topic list per the issue #9 API section
        # (the dedicated ``error`` event lands in v0.5).
        values = snapshot.values or {}
        current_step = values.get("current_step")
        research_topics = values.get("research_topics")

        if not plan_complete_emitted and current_step not in (None, "plan", "confirm"):
            plan_complete_emitted = True
            topics = (
                [] if current_step == "failed" else _research_topics_dict(research_topics)
            )
            yield _format_event(
                "plan_complete",
                thread_id,
                {"research_topics": topics},
            )

        # Step 3 — close on the terminal states (issue #14 AC).
        # Two close signals combine to cover all reachable terminals:
        #   * `current_step in ("final", "failed")` — the main graph's
        #     natural terminal ("final") and the route's failure-marker
        #     write ("failed"; ADR-0008 fast-fail). The graph's `next`
        #     may still be non-empty in the failed case (the failing
        #     node never completed, so LangGraph keeps it pending) —
        #     this signal is load-bearing there. Regression test:
        #     ``test_graph_failure_writes_failed_state_and_finalizes_the_stream``.
        #   * `not snapshot.next` — the canonical "graph at END"
        #     signal. Covers the standalone plan subgraph, which
        #     terminates with ``current_step='research'`` (not in the
        #     failure/final set) and ``next=()``. Regression test:
        #     ``test_real_plan_subgraph_completion_closes_stream``.
        if not snapshot.next or current_step in ("final", "failed"):
            return

        await asyncio.sleep(SSE_POLL_INTERVAL_SECONDS)


__all__ = [
    "SSE_POLL_INTERVAL_SECONDS",
    "stream_plan_events",
]
