"""Ticket #14 — ``GET /runs/{thread_id}/events`` (SSE Plan-stage stream).

The endpoint streams two events in v0.2 (the other six land in v0.5):

* ``plan_question`` — fired when the Plan subgraph enters
  ``interrupt()`` on a new round; payload is
  ``PlanInterruptPayload.model_dump()``.
* ``plan_complete`` — fired when the Plan subgraph writes
  ``current_step="research"`` (or the graph hits the ``"failed"``
  terminal state).

Each event carries the v0.5 forward-compatible envelope
``{"thread_id": ..., "ts": <unix-ms>, "payload": {...}}`` (issue #14
AC, user-story #16).

Tests pin each acceptance criterion via a controllable mock graph
whose ``get_state`` returns scripted snapshots. The real ``StreamingResponse``
+ ``TestClient.stream(...)`` plumbing is exercised end-to-end; only the
graph itself is faked. The SSE poll interval is monkeypatched to a tiny
value so tests don't pay the 200 ms cost per loop.

The real-graph tests at the bottom drive actual LangGraph primitives
(``build_plan_subgraph`` paused at ``interrupt()``, and a fully-run
``build_graph`` pipeline) so the wire contract is pinned against the
checkpointer's real round-trip of ``snapshot.tasks[].interrupts[]``.

Note on ``TestClient.stream``: Starlette's TestClient runs the ASGI
app to completion before the context manager returns (its ``receive``
only yields ``http.disconnect`` after the response body finishes). An
SSE generator that never terminates therefore *deadlocks* the client —
so the paused-subgraph case consumes the generator directly with a
disconnect-stub Request, while the completed-pipeline case is finite
(``plan_complete`` replayed once, then close at END) and exercises the
full route + ``StreamingResponse`` stack.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver

from app.api import events as events_module
from app.api.events import stream_plan_events
from app.api.routes import _build_config, router
from app.models import (
    PlanInterruptPayload,
    ResearchTopic,
    SingleChoicePlanQuestion,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures + helpers — small poll interval so tests don't pay 200ms per loop.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _short_sse_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop the SSE poll to a few ms so the polling loop terminates quickly.

    The generator is configurable through a module-level constant; the
    route reads it indirectly via the ``stream_plan_events`` closure.
    We rebind the constant here so both the route and the standalone
    helper used in unit tests stay fast.
    """
    monkeypatch.setattr(events_module, "SSE_POLL_INTERVAL_SECONDS", 0.005)


class _FakeInterrupt:
    """Tiny stand-in for LangGraph's ``Interrupt`` class.

    LangGraph's interrupt object exposes ``id`` and ``value``. Tests
    build this dict-shaped variant so we don't need to import the
    internal class, which is not part of the public API surface.
    """

    def __init__(self, id_: str, value: Any) -> None:
        self.id = id_
        self.value = value


class _FakeTask:
    """Stand-in for the ``PregelTask`` in ``snapshot.tasks``."""

    def __init__(self, interrupts: list[_FakeInterrupt] | None = None) -> None:
        self.interrupts = list(interrupts or [])
        self.state = None
        self.name = "plan_subgraph"


class _FakeSnapshot:
    """Stand-in for the ``StateSnapshot`` from ``graph.get_state(config)``.

    The SSE generator touches ``.values`` (dict), ``.tasks`` (iterable
    of objects with ``.interrupts``), and ``.next`` (tuple of pending
    node names — empty means the graph is at END; the close-condition
    fix for issue #14 reads this).

    ``next`` defaults to a non-empty tuple so existing tests that
    weren't exercising the ``next=()`` close path continue to close
    only via ``current_step in ("final", "failed")`` (or the
    empty-values-and-tasks guard). Tests that want to script the
    standalone-plan-subgraph "research + next=()" shape pass
    ``next=()`` explicitly.
    """

    def __init__(
        self,
        values: dict[str, Any] | None = None,
        tasks: list[_FakeTask] | None = None,
        next: tuple[str, ...] = ("plan_subgraph",),
    ) -> None:
        self.values = values or {}
        self.tasks = tasks or []
        self.next = next


class _MockGraph:
    """Test double for the compiled LangGraph used in the route tests.

    ``scripts`` is a FIFO of ``_FakeSnapshot`` (or ``None``) returned by
    successive ``get_state`` calls — letting one test drive a full
    plan→research→final transition with deterministic timing. An empty
    queue returns ``None`` which the SSE generator treats as "close the
    stream".
    """

    def __init__(self, scripts: list[_FakeSnapshot | None] | None = None) -> None:
        self.scripts: list[_FakeSnapshot | None] = list(scripts or [])
        self.get_state_calls = 0

    def get_state(self, config: dict[str, Any]) -> _FakeSnapshot | None:
        self.get_state_calls += 1
        if not self.scripts:
            return None
        return self.scripts.pop(0)


@pytest.fixture
def app_with_mock_graph() -> tuple[FastAPI, _MockGraph]:
    """Return ``(app, graph)`` with the routes mounted and a default mock."""
    app = FastAPI()
    app.include_router(router)
    graph = _MockGraph()
    app.state.graph = graph
    return app, graph


def _collect_sse_events(response) -> list[dict[str, Any]]:
    """Parse a streaming ``text/event-stream`` response into structured events.

    Each test wants to read ``event: <name>\\ndata: <json>\\n\\n`` chunks.
    ``response.iter_lines()`` yields one SSE line at a time (including the
    blank separator), so we accumulate until we see the blank line and
    parse the pair. The returned list preserves emission order.
    """
    events: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    for raw in response.iter_lines():
        line = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
        if line == "":
            if pending is not None:
                events.append(pending)
                pending = None
            continue
        if line.startswith("event:"):
            pending = pending or {}
            pending["event"] = line[len("event:") :].strip()
        elif line.startswith("data:"):
            pending = pending or {}
            pending["data"] = json.loads(line[len("data:") :].strip())
    return events


class _StubChat:
    """Scripted chat-model stub for the real-graph tests.

    ``get_chat_model`` must return ONE shared instance: the questioner,
    the topic-finalizer, and the write stage each call it separately,
    and a fresh stub per call would restart the scripted response queue.
    """

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses

    def invoke(self, messages, **_kwargs):
        content = self.responses.pop(0) if self.responses else "{}"

        class _M:
            def __init__(self, c: str) -> None:
                self.content = c

        return _M(content)


# ─────────────────────────────────────────────────────────────────────────────
# Acceptance criterion 1 — endpoint returns ``text/event-stream``.
# ─────────────────────────────────────────────────────────────────────────────


def test_events_endpoint_returns_event_stream_content_type(app_with_mock_graph) -> None:
    """Acceptance: ``GET /runs/{thread_id}/events`` returns ``text/event-stream``."""
    app, graph = app_with_mock_graph
    # Script a final-state snapshot so the stream closes immediately.
    graph.scripts = [
        _FakeSnapshot(values={"current_step": "final"}),
        None,
    ]

    client = TestClient(app)
    with client.stream("GET", "/runs/some-thread/events") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        # Disable the rest of the body by closing early; the headers are
        # what this assertion pins.
        response.close()


# ─────────────────────────────────────────────────────────────────────────────
# Acceptance criterion 2 — ``plan_question`` payload is the typed
# ``PlanInterruptPayload.model_dump()`` shape.
# ─────────────────────────────────────────────────────────────────────────────


def test_plan_question_emitted_with_interrupt_payload(app_with_mock_graph) -> None:
    """When the snapshot has a new interrupt, emit ``plan_question`` with the dumped payload.

    The mocked interrupt's ``value`` is a typed ``PlanInterruptPayload``
    (mirroring what the LangGraph checkpointer hands back); the SSE
    payload must serialise that to ``{questions, reasoning, round}``.
    The envelope's ``thread_id`` matches the URL path component and
    ``ts`` is a unix-ms integer.
    """
    app, graph = app_with_mock_graph
    payload = PlanInterruptPayload(
        questions=[
            SingleChoicePlanQuestion(
                id="q-scope",
                prompt="Pick scope",
                options=[
                    {"option_id": "tech", "label": "Tech"},
                    {"option_id": "biz", "label": "Business"},
                ],
            )
        ],
        reasoning="clarify scope",
        round=1,
    )
    interrupt = _FakeInterrupt(id_="intr-1", value=payload)
    graph.scripts = [
        _FakeSnapshot(
            values={"user_query": "X"},
            tasks=[_FakeTask(interrupts=[interrupt])],
        ),
        None,  # close the stream
    ]

    client = TestClient(app)
    with client.stream("GET", "/runs/thread-abc/events") as response:
        events = _collect_sse_events(response)

    assert len(events) == 1
    ev = events[0]
    assert ev["event"] == "plan_question"
    assert ev["data"]["thread_id"] == "thread-abc"
    assert isinstance(ev["data"]["ts"], int)
    # ts must be in the unix-ms ballpark (within a minute of now).
    assert abs(ev["data"]["ts"] - int(time.time() * 1000)) < 60_000

    body = ev["data"]["payload"]
    assert body["round"] == 1
    assert body["reasoning"] == "clarify scope"
    assert len(body["questions"]) == 1
    assert body["questions"][0]["kind"] == "single_choice"
    assert body["questions"][0]["id"] == "q-scope"
    assert [o["option_id"] for o in body["questions"][0]["options"]] == ["tech", "biz"]


def test_plan_question_accepts_dict_interrupt_value(app_with_mock_graph) -> None:
    """If the interrupt value is a plain dict (deserializer failed), the
    generator coerces through ``PlanInterruptPayload.model_validate``."""
    app, graph = app_with_mock_graph
    payload_dict = {
        "questions": [
            {
                "kind": "confirmation",
                "id": "q-confirm",
                "prompt": "Continue?",
            }
        ],
        "reasoning": "ok",
        "round": 2,
    }
    interrupt = _FakeInterrupt(id_="intr-2", value=payload_dict)
    graph.scripts = [
        _FakeSnapshot(values={}, tasks=[_FakeTask(interrupts=[interrupt])]),
        None,
    ]

    client = TestClient(app)
    with client.stream("GET", "/runs/t-dict/events") as response:
        events = _collect_sse_events(response)

    assert len(events) == 1
    body = events[0]["data"]["payload"]
    assert body["round"] == 2
    assert body["questions"][0]["kind"] == "confirmation"
    assert body["questions"][0]["id"] == "q-confirm"


def test_plan_question_accepts_lc_envelope_interrupt_value(app_with_mock_graph) -> None:
    """A real ``RedisSaver`` msgpack round-trip of an unregistered class
    returns the LangChain constructor envelope (fields nested under
    ``kwargs``) — the generator must unwrap it before validating."""
    app, graph = app_with_mock_graph
    envelope = {
        "lc": 2,
        "type": "constructor",
        "id": ["app", "models", "PlanInterruptPayload"],
        "kwargs": {
            "questions": [
                {
                    "kind": "free_text",
                    "id": "q-region",
                    "prompt": "Which regions?",
                }
            ],
            "reasoning": "narrow scope",
            "round": 1,
        },
    }
    interrupt = _FakeInterrupt(id_="intr-lc", value=envelope)
    graph.scripts = [
        _FakeSnapshot(values={}, tasks=[_FakeTask(interrupts=[interrupt])]),
        None,
    ]

    client = TestClient(app)
    with client.stream("GET", "/runs/t-lc/events") as response:
        events = _collect_sse_events(response)

    assert len(events) == 1
    body = events[0]["data"]["payload"]
    assert body["round"] == 1
    assert body["reasoning"] == "narrow scope"
    assert body["questions"][0]["id"] == "q-region"


# ─────────────────────────────────────────────────────────────────────────────
# Acceptance criterion 3 — ``plan_complete`` payload is
# ``{research_topics: list[ResearchTopic]}``.
# ─────────────────────────────────────────────────────────────────────────────


def test_plan_complete_emitted_when_state_advances_past_plan(app_with_mock_graph) -> None:
    """When the snapshot's ``current_step`` leaves ``"plan"`` and carries topics,
    emit ``plan_complete`` with the topics list."""
    app, graph = app_with_mock_graph
    topics = [
        ResearchTopic(
            id="topic-1", title="T1", description="d1", expected_finding_type="f1"
        ),
        ResearchTopic(
            id="topic-2", title="T2", description="d2", expected_finding_type="f2"
        ),
    ]
    graph.scripts = [
        _FakeSnapshot(values={"current_step": "research", "research_topics": topics}),
        # The "final" snapshot closes the stream per AC.
        _FakeSnapshot(values={"current_step": "final"}),
        None,
    ]

    client = TestClient(app)
    with client.stream("GET", "/runs/t-fin/events") as response:
        events = _collect_sse_events(response)

    assert len(events) == 1
    ev = events[0]
    assert ev["event"] == "plan_complete"
    assert ev["data"]["thread_id"] == "t-fin"
    assert isinstance(ev["data"]["ts"], int)
    assert ev["data"]["payload"]["research_topics"] == [
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


def test_plan_complete_emitted_exactly_once(app_with_mock_graph) -> None:
    """Multiple consecutive snapshots that all clear the plan stage emit the
    event only the first time — dedupe by boolean flag prevents wire-dup
    when the connection lingers between ``plan_complete`` and the
    final-terminal close.
    """
    app, graph = app_with_mock_graph
    topics = [
        ResearchTopic(
            id="topic-1", title="T1", description="d", expected_finding_type="f"
        ),
    ]
    graph.scripts = [
        # First poll: research-step snapshot — emit plan_complete.
        _FakeSnapshot(
            values={"current_step": "research", "research_topics": topics}
        ),
        # Second poll: still research-step, no new events.
        _FakeSnapshot(
            values={"current_step": "research", "research_topics": topics}
        ),
        # Third poll: final — closes.
        _FakeSnapshot(values={"current_step": "final"}),
        None,
    ]

    client = TestClient(app)
    with client.stream("GET", "/runs/t-once/events") as response:
        events = _collect_sse_events(response)

    assert [e["event"] for e in events] == ["plan_complete"]


def test_plan_complete_emitted_with_empty_topics_on_failure(app_with_mock_graph) -> None:
    """Per spec: when the graph hits the ``failed`` terminal state, the
    final ``plan_complete`` carries ``research_topics=[]`` (the
    dedicated ``error`` event lands in v0.5)."""
    app, graph = app_with_mock_graph
    graph.scripts = [
        _FakeSnapshot(values={"current_step": "failed", "research_topics": None}),
    ]

    client = TestClient(app)
    with client.stream("GET", "/runs/t-fail/events") as response:
        events = _collect_sse_events(response)

    assert len(events) == 1
    assert events[0]["event"] == "plan_complete"
    assert events[0]["data"]["payload"] == {"research_topics": []}


# ─────────────────────────────────────────────────────────────────────────────
# Acceptance criterion 4 — same interrupt id is not re-emitted.
# ─────────────────────────────────────────────────────────────────────────────


def test_plan_question_dedupes_by_interrupt_id(app_with_mock_graph) -> None:
    """A snapshot whose interrupt id was already seen does not re-emit
    ``plan_question`` — one event per interrupt id is the wire
    contract.
    """
    app, graph = app_with_mock_graph
    interrupt = _FakeInterrupt(
        id_="same-id",
        value=PlanInterruptPayload(questions=[], reasoning="r", round=1),
    )
    graph.scripts = [
        _FakeSnapshot(values={}, tasks=[_FakeTask(interrupts=[interrupt])]),
        # Same snapshot again — must NOT re-emit.
        _FakeSnapshot(values={}, tasks=[_FakeTask(interrupts=[interrupt])]),
        _FakeSnapshot(values={"current_step": "final"}),
        None,
    ]

    client = TestClient(app)
    with client.stream("GET", "/runs/t-dedupe/events") as response:
        events = _collect_sse_events(response)

    # One plan_question across two identical paused snapshots; the
    # trailing "final" snapshot then replays the plan_complete
    # handshake (step-2 rule) before closing.
    assert [e["event"] for e in events] == ["plan_question", "plan_complete"]


def test_two_distinct_interrupts_emit_two_plan_question_events(
    app_with_mock_graph,
) -> None:
    """A second ``interrupt()`` round id surfaces as a second ``plan_question``.
    The dedupe key is the LangGraph-assigned ``Interrupt.id``, so a
    new round on the same thread produces a fresh id and a fresh
    event.
    """
    app, graph = app_with_mock_graph
    interrupt_r1 = _FakeInterrupt(
        id_="intr-r1",
        value=PlanInterruptPayload(
            questions=[
                SingleChoicePlanQuestion(
                    id="q1", prompt="First?", options=[{"option_id": "a", "label": "A"}]
                )
            ],
            reasoning="r1",
            round=1,
        ),
    )
    interrupt_r2 = _FakeInterrupt(
        id_="intr-r2",
        value=PlanInterruptPayload(
            questions=[
                SingleChoicePlanQuestion(
                    id="q2", prompt="Second?", options=[{"option_id": "a", "label": "A"}]
                )
            ],
            reasoning="r2",
            round=2,
        ),
    )
    graph.scripts = [
        _FakeSnapshot(values={}, tasks=[_FakeTask(interrupts=[interrupt_r1])]),
        _FakeSnapshot(values={}, tasks=[_FakeTask(interrupts=[interrupt_r2])]),
        _FakeSnapshot(values={"current_step": "final"}),
        None,
    ]

    client = TestClient(app)
    with client.stream("GET", "/runs/t-multi/events") as response:
        events = _collect_sse_events(response)

    # The third snapshot ("final") replays the plan_complete handshake
    # before closing (step-2 rule); the two rounds above are the
    # distinct-interrupt assertion.
    assert [e["event"] for e in events] == [
        "plan_question",
        "plan_question",
        "plan_complete",
    ]
    assert events[0]["data"]["payload"]["round"] == 1
    assert events[1]["data"]["payload"]["round"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# Acceptance criterion 5 — stream closes when the graph reaches END or the
# thread is unknown.
# ─────────────────────────────────────────────────────────────────────────────


def test_stream_closes_when_current_step_is_final(app_with_mock_graph) -> None:
    """``current_step == "final"`` closes the stream. A late-joining
    client that never saw the stage advance still gets the
    ``plan_complete`` handshake before the close (the checkpointer
    makes it replayable), so the single snapshot produces exactly one
    event — with whatever topics the final state carries, none here.
    """
    app, graph = app_with_mock_graph
    graph.scripts = [
        _FakeSnapshot(values={"current_step": "final"}),
        # The generator must NOT consult any further snapshots after
        # the close — list intentionally ends here.
    ]

    client = TestClient(app)
    with client.stream("GET", "/runs/t-end/events") as response:
        events = _collect_sse_events(response)

    assert [e["event"] for e in events] == ["plan_complete"]
    assert events[0]["data"]["payload"] == {"research_topics": []}
    # And the generator stopped polling — only the single scripted snapshot was consulted.
    assert graph.get_state_calls == 1


def test_stream_closes_when_thread_unknown(app_with_mock_graph) -> None:
    """``get_state`` returning ``None`` closes the stream immediately
    (issue #14 AC: "the graph fails" — the closest analogue when the
    thread is gone)."""
    app, graph = app_with_mock_graph
    graph.scripts = [None]

    client = TestClient(app)
    with client.stream("GET", "/runs/t-unknown/events") as response:
        events = _collect_sse_events(response)

    assert events == []
    assert graph.get_state_calls == 1


def test_stream_closes_when_snapshot_is_empty(app_with_mock_graph) -> None:
    """``MemorySaver`` reports an unknown thread as a snapshot with empty
    ``values`` AND no ``tasks`` — never ``None``. That shape must close
    the stream too (mirrors the ``_get_state_or_404`` guard in
    ``routes.py``); a real-checkpointer version of this case lives at the
    bottom of the file."""
    app, graph = app_with_mock_graph
    graph.scripts = [_FakeSnapshot(values={}, tasks=[])]

    client = TestClient(app)
    with client.stream("GET", "/runs/t-empty/events") as response:
        events = _collect_sse_events(response)

    assert events == []
    assert graph.get_state_calls == 1


def test_stream_closes_when_current_step_is_failed(app_with_mock_graph) -> None:
    """``current_step == "failed"`` emits a final ``plan_complete`` and closes."""
    app, graph = app_with_mock_graph
    graph.scripts = [
        _FakeSnapshot(values={"current_step": "failed"}),
    ]

    client = TestClient(app)
    with client.stream("GET", "/runs/t-fail-end/events") as response:
        events = _collect_sse_events(response)

    assert len(events) == 1
    assert events[0]["event"] == "plan_complete"
    assert events[0]["data"]["payload"]["research_topics"] == []


# ─────────────────────────────────────────────────────────────────────────────
# Acceptance criterion 6 — full happy-path walk in one stream.
# ─────────────────────────────────────────────────────────────────────────────


def test_full_plan_lifecycle_stream_emits_question_then_complete(
    app_with_mock_graph,
) -> None:
    """A scripted plan→question→complete→final walk emits ``plan_question``
    then ``plan_complete`` and closes on the final snapshot — pinning
    the two-event sequence in one pass.
    """
    app, graph = app_with_mock_graph
    interrupt = _FakeInterrupt(
        id_="intr-walk",
        value=PlanInterruptPayload(
            questions=[
                SingleChoicePlanQuestion(
                    id="q1", prompt="Pick", options=[{"option_id": "a", "label": "A"}]
                )
            ],
            reasoning="r",
            round=1,
        ),
    )
    topics = [
        ResearchTopic(
            id="topic-1", title="T1", description="d", expected_finding_type="f"
        )
    ]
    graph.scripts = [
        # Poll 1: plan paused.
        _FakeSnapshot(
            values={"current_step": "plan"},
            tasks=[_FakeTask(interrupts=[interrupt])],
        ),
        # Poll 2: research-stage reached.
        _FakeSnapshot(
            values={"current_step": "research", "research_topics": topics}
        ),
        # Poll 3: pipeline complete — close.
        _FakeSnapshot(values={"current_step": "final"}),
        None,
    ]

    client = TestClient(app)
    with client.stream("GET", "/runs/t-walk/events") as response:
        events = _collect_sse_events(response)

    assert [e["event"] for e in events] == ["plan_question", "plan_complete"]
    # ts is monotonically non-decreasing between events.
    assert events[1]["data"]["ts"] >= events[0]["data"]["ts"]
    # thread_id is the URL path component.
    for e in events:
        assert e["data"]["thread_id"] == "t-walk"


# ─────────────────────────────────────────────────────────────────────────────
# Acceptance criterion 7 — wire format is exactly ``event: \\ndata: \\n\\n``.
# ─────────────────────────────────────────────────────────────────────────────


def test_event_wire_format_matches_sse_spec(app_with_mock_graph) -> None:
    """Pin the exact byte layout — two-line ``event`` and ``data`` followed
    by a blank line. A frontend that parses SSE by line splitting must
    see a trailing blank line per event.
    """
    app, graph = app_with_mock_graph
    interrupt = _FakeInterrupt(
        id_="i",
        value=PlanInterruptPayload(questions=[], reasoning="r", round=1),
    )
    graph.scripts = [
        _FakeSnapshot(values={}, tasks=[_FakeTask(interrupts=[interrupt])]),
        None,
    ]

    client = TestClient(app)
    with client.stream("GET", "/runs/t-format/events") as response:
        # Reconstruct the raw wire bytes — assert the exact shape.
        raw = b""
        for chunk in response.iter_bytes():
            raw += chunk

    # One event = ``event: <name>\\ndata: <json>\\n\\n``.
    assert raw.count(b"\n\n") == 1, f"expected one trailing blank line, got {raw!r}"
    # The data line is exactly ``data: {json}\\n``.
    assert b"\ndata: {" in raw
    # And the event line leads.
    assert raw.startswith(b"event: plan_question\n")


# ─────────────────────────────────────────────────────────────────────────────
# 503 path — graph not initialised.
# ─────────────────────────────────────────────────────────────────────────────


def test_events_returns_503_when_graph_not_initialised() -> None:
    """If ``app.state.graph is None`` (lifespan didn't run), the route
    returns 503 just like the other plan routes. A failed lifespan must
    not silently start an empty SSE stream."""
    app = FastAPI()
    app.include_router(router)
    app.state.graph = None
    client = TestClient(app)
    response = client.get("/runs/anything/events")
    assert response.status_code == 503
    assert "Graph not initialized" in response.json()["detail"]


# ─────────────────────────────────────────────────────────────────────────────
# Real LangGraph primitives — pin the wire contract against the live
# ``get_state().tasks[].interrupts`` shape, not just our mocks.
# ─────────────────────────────────────────────────────────────────────────────


_QUESTION_BATCH_JSON = json.dumps(
    {
        "questions": [
            {
                "kind": "single_choice",
                "id": "q-scope",
                "prompt": "Which angle?",
                "options": [
                    {"option_id": "tech", "label": "Technical"},
                    {"option_id": "biz", "label": "Business"},
                ],
            }
        ],
        "reasoning": "clarify scope",
        "ready_to_proceed": False,
    }
)


class _DisconnectAfterFirstPoll:
    """Fake ``Request`` whose ``is_disconnected`` is checked at the top of
    each generator pass: the first pass proceeds, the second ends the
    loop — the generator's client-close path."""

    def __init__(self) -> None:
        self.checks = 0

    async def is_disconnected(self) -> bool:
        self.checks += 1
        return self.checks > 1


@pytest.mark.asyncio
async def test_real_plan_subgraph_pause_emits_plan_question(monkeypatch) -> None:
    """A thread paused at a real ``interrupt()`` surfaces exactly one
    ``plan_question`` frame carrying the typed payload round-tripped
    through the ``MemorySaver``.

    Driven through the generator directly rather than
    ``TestClient.stream``: Starlette's TestClient runs the ASGI app to
    completion before returning, so a still-paused (never-terminating)
    stream deadlocks it — see the module docstring. The route wiring
    itself is covered by the mock-graph tests above; this pins the wire
    *content* against the real LangGraph primitives.
    """
    from app.graph.subgraphs.plan import build_plan_subgraph
    from app.tools import llm

    stub_chat = _StubChat([_QUESTION_BATCH_JSON])
    monkeypatch.setattr(llm, "get_chat_model", lambda: stub_chat)

    real_graph = build_plan_subgraph(MemorySaver())
    thread_id = str(uuid.uuid4())
    config = _build_config(thread_id)

    first = real_graph.invoke({"user_query": "Tell me about X"}, config=config)
    assert "__interrupt__" in first, "plan subgraph must be paused at interrupt()"

    # The checkpointer round-trips the interrupt value as the TYPED
    # payload — this is exactly what the SSE generator reads off
    # ``snapshot.tasks[].interrupts[].value``.
    paused = real_graph.get_state(config)
    assert isinstance(paused.tasks[0].interrupts[0].value, PlanInterruptPayload)

    chunks = [
        c
        async for c in stream_plan_events(
            _DisconnectAfterFirstPoll(), real_graph, config, thread_id
        )
    ]

    assert len(chunks) == 1, "one poll must emit exactly one plan_question frame"
    frame = chunks[0].decode("utf-8")
    assert frame.startswith("event: plan_question\ndata: ")
    assert frame.endswith("\n\n")
    envelope = json.loads(frame.split("data: ", 1)[1].split("\n\n", 1)[0])
    assert envelope["thread_id"] == thread_id
    assert isinstance(envelope["ts"], int)
    body = envelope["payload"]
    assert body["round"] == 1
    assert body["reasoning"] == "clarify scope"
    assert body["questions"][0]["id"] == "q-scope"
    assert body["questions"][0]["kind"] == "single_choice"
    assert [o["option_id"] for o in body["questions"][0]["options"]] == ["tech", "biz"]


@pytest.mark.asyncio
async def test_real_checkpointer_unknown_thread_closes_immediately() -> None:
    """A real ``MemorySaver`` reports an unknown thread as a snapshot with
    empty ``values`` and no ``tasks`` — never ``None``. The generator
    must recognise that shape and close on the first poll, otherwise
    every mistyped thread id would hold an open, poll-forever stream."""
    from app.graph.subgraphs.plan import build_plan_subgraph

    real_graph = build_plan_subgraph(MemorySaver())
    thread_id = "no-such-thread"

    class _NeverDisconnects:
        async def is_disconnected(self) -> bool:
            return False

    chunks = [
        c
        async for c in stream_plan_events(
            _NeverDisconnects(), real_graph, _build_config(thread_id), thread_id
        )
    ]
    assert chunks == []


@pytest.mark.asyncio
async def test_real_plan_subgraph_completion_closes_stream(monkeypatch) -> None:
    """A completed standalone plan subgraph closes the SSE stream.

    Regression for the bug where ``stream_plan_events`` only closed on
    ``current_step in ("final", "failed")``. The standalone plan
    subgraph terminates with ``current_step='research'`` and
    ``next=()`` — neither half of the old condition matched, so the
    stream polled forever against a thread the graph will never
    advance. The fix adds ``not snapshot.next`` as a close signal
    (canonical "graph at END"), which fires here without breaking the
    failure path (where ``next`` stays non-empty after the failing
    node — see ``test_graph_failure_writes_failed_state_and_finalizes_the_stream``).

    In v0.3 the questioner finalizes into the Plan→Research transition
    gate (ADR-0005, issue #23), so the subgraph pauses at the gate
    instead of reaching END after ``ready_to_proceed=true``. The test
    drives the gate resume via ``Command(resume=ConfirmedResearchTopics(...))``
    so the standalone subgraph lands at END and the close signal
    fires — this test pins the close signal, NOT the gate event
    shape (the gate event is covered by the unit test below).

    ``asyncio.timeout(2.0)`` converts the hang into a real
    ``TimeoutError``; without the timeout the test deadlocks
    ``async for`` instead of failing, and pytest can't report a verdict.
    """
    from app.graph.subgraphs.plan import build_plan_subgraph
    from app.models import ConfirmedResearchTopics
    from app.tools import llm

    topics_payload = {
        "topics": [
            {
                "id": "topic-1",
                "title": "T1",
                "description": "d1",
                "expected_finding_type": "f1",
            }
        ]
    }
    stub_chat = _StubChat(
        [
            json.dumps(
                {
                    "questions": [],
                    "reasoning": "Have enough; ready to plan.",
                    "ready_to_proceed": True,
                }
            ),
            json.dumps(topics_payload),
        ]
    )
    monkeypatch.setattr(llm, "get_chat_model", lambda: stub_chat)

    real_graph = build_plan_subgraph(MemorySaver())
    thread_id = str(uuid.uuid4())
    config = _build_config(thread_id)

    paused = real_graph.invoke({"user_query": "Tell me about X"}, config=config)
    # v0.3 — the questioner finalizes into the gate; the subgraph
    # pauses at the gate instead of reaching END.
    assert paused["current_step"] == "research", (
        "precondition: the standalone subgraph must have reached the "
        "research step before the gate pauses"
    )
    assert real_graph.get_state(config).next, (
        "precondition: the gate is paused (next must be non-empty)"
    )

    # Drive the gate resume — the gate's draft topics are what
    # ``plan_questioner._finalize_topics`` just wrote. A no-op
    # confirm (the user accepts the draft unchanged) keeps the diff
    # empty and the test focused on the close signal.
    from langgraph.types import Command

    gate_payload = real_graph.get_state(config).tasks[0].interrupts[0].value
    real_graph.invoke(
        Command(resume=ConfirmedResearchTopics(topics=list(gate_payload.topics))),
        config=config,
    )
    # Sanity — the subgraph is at END; this is the close signal the
    # fix relies on.
    assert real_graph.get_state(config).next == ()

    class _NeverDisconnects:
        async def is_disconnected(self) -> bool:
            return False

    chunks: list[bytes] = []
    try:
        async with asyncio.timeout(2.0):
            async for c in stream_plan_events(
                _NeverDisconnects(), real_graph, config, thread_id
            ):
                chunks.append(c)
    except TimeoutError:
        pytest.fail(
            "SSE stream did not close for a completed standalone plan "
            "subgraph within 2s — stream_plan_events must close on "
            "`not snapshot.next` (graph at END), not just on "
            "`current_step in ('final', 'failed')`."
        )

    # One plan_complete frame with the topics the finalizer produced.
    # The poll happens AFTER the gate is confirmed and the subgraph
    # is at END, so the gate interrupt is no longer in the snapshot;
    # the only observable event is the plan_complete handshake on
    # the research step. (The gate event itself is covered by
    # ``test_real_plan_subgraph_gate_pause_emits_plan_confirm_required``
    # below, which polls while the gate is paused.)
    assert len(chunks) == 1, f"expected exactly one plan_complete frame, got {chunks!r}"
    frame = chunks[0].decode("utf-8")
    assert frame.startswith("event: plan_complete\ndata: ")
    assert frame.endswith("\n\n")
    envelope = json.loads(frame.split("data: ", 1)[1].split("\n\n", 1)[0])
    assert envelope["thread_id"] == thread_id
    assert envelope["payload"]["research_topics"][0]["title"] == "T1"


@pytest.mark.asyncio
async def test_real_plan_subgraph_gate_pause_emits_plan_confirm_required(
    monkeypatch,
) -> None:
    """A standalone plan subgraph paused at the gate emits ``plan_confirm_required``.

    Companion to ``test_real_plan_subgraph_pause_emits_plan_question``:
    the questioner interrupt is ``plan_question`` (issue #14), the
    gate interrupt is ``plan_confirm_required`` (issue #25). The two
    share the same ``Interrupt`` channel so the SSE generator has to
    discriminate by payload shape (the gate carries ``topics``, the
    questioner carries ``questions``) — that discrimination is what
    this test pins against the real LangGraph primitives + checkpointer
    round-trip.

    The test drives the generator directly rather than via
    ``TestClient.stream`` because a still-paused (never-terminating)
    stream deadlocks Starlette's TestClient (see the module
    docstring). A small ``_NeverDisconnects`` fake keeps the loop
    running until the disconnect stub's check kicks in.
    """
    from app.graph.subgraphs.plan import build_plan_subgraph
    from app.tools import llm

    topics_payload = {
        "topics": [
            {
                "id": "topic-1",
                "title": "T1",
                "description": "d1",
                "expected_finding_type": "f1",
            }
        ]
    }
    stub_chat = _StubChat(
        [
            json.dumps(
                {
                    "questions": [],
                    "reasoning": "Have enough; ready to plan.",
                    "ready_to_proceed": True,
                }
            ),
            json.dumps(topics_payload),
        ]
    )
    monkeypatch.setattr(llm, "get_chat_model", lambda: stub_chat)

    real_graph = build_plan_subgraph(MemorySaver())
    thread_id = str(uuid.uuid4())
    config = _build_config(thread_id)

    # Drive the questioner to ``ready_to_proceed=true`` → finalize →
    # gate pause. The subgraph is paused at the gate (not at the
    # questioner), so the SSE generator must surface the gate's
    # ``plan_confirm_required`` event with the draft topics.
    paused = real_graph.invoke({"user_query": "Tell me about X"}, config=config)
    assert paused["current_step"] == "research"
    assert real_graph.get_state(config).next, "gate must be paused"

    gate_paused = real_graph.get_state(config)
    from app.models import PlanConfirmInterruptPayload

    assert isinstance(
        gate_paused.tasks[0].interrupts[0].value, PlanConfirmInterruptPayload
    )

    class _DisconnectAfterFirstPoll:
        def __init__(self) -> None:
            self.checks = 0

        async def is_disconnected(self) -> bool:
            self.checks += 1
            return self.checks > 1

    chunks = [
        c
        async for c in stream_plan_events(
            _DisconnectAfterFirstPoll(), real_graph, config, thread_id
        )
    ]

    # Two events from the same snapshot: the gate interrupt
    # (plan_confirm_required) and the plan_complete handshake
    # (current_step is "research"). The generator walks tasks before
    # current_step, so the gate event comes first.
    assert len(chunks) == 2, f"expected 2 frames, got {chunks!r}"
    confirm_frame = chunks[0].decode("utf-8")
    assert confirm_frame.startswith("event: plan_confirm_required\ndata: ")
    assert confirm_frame.endswith("\n\n")
    confirm_envelope = json.loads(
        confirm_frame.split("data: ", 1)[1].split("\n\n", 1)[0]
    )
    assert confirm_envelope["thread_id"] == thread_id
    assert isinstance(confirm_envelope["ts"], int)
    assert confirm_envelope["payload"]["reasoning"] == (
        "Please review, edit, and confirm the research topics"
    )
    assert confirm_envelope["payload"]["topics"][0]["title"] == "T1"
    # The id is a uuid4 (ADR-0005 §2) — not the LLM's ``topic-1``
    # slug. Pin the id format rather than the value to keep the test
    # stable across reruns.
    import re as _re

    assert _re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
        confirm_envelope["payload"]["topics"][0]["id"],
    ), f"expected a uuid4 id, got {confirm_envelope['payload']['topics'][0]['id']!r}"


def test_graph_failure_writes_failed_state_and_finalizes_the_stream(
    monkeypatch, memory_checkpointer
) -> None:
    """Acceptance: ``the stream closes when ... the graph fails``.

    The graph code never writes ``current_step="failed"`` itself
    (ADR-0008 fast-fail raises), so ``answer_run`` persists it via
    ``_mark_run_failed`` before propagating the 500. This drives the
    REAL MAIN GRAPH (the production shape — ``app.state.graph``) to a
    paused round-1 state over ``POST /runs``, blows up the resume, and
    asserts the stream finalizes with parent #9's empty-topics handshake
    and closes (finite, so safe under TestClient).

    Note: the standalone plan subgraph is NOT a valid target for this —
    its ``PlanInput`` schema silently drops ``current_step`` writes from
    ``update_state``; only the main graph's schemaless ``State`` accepts
    the failure marker.
    """
    from app.graph import build_graph
    from app.tools import llm

    stub_chat = _StubChat([_QUESTION_BATCH_JSON])
    monkeypatch.setattr(llm, "get_chat_model", lambda: stub_chat)

    real_graph = build_graph(memory_checkpointer)
    app = FastAPI()
    app.include_router(router)
    app.state.graph = real_graph
    client = TestClient(app, raise_server_exceptions=False)

    created = client.post("/runs", json={"user_query": "will fail"})
    assert created.status_code == 200
    thread_id = created.json()["thread_id"]

    # The resume re-enters plan_questioner; building the chat model now
    # explodes — the route must 500 AND persist the terminal state.
    def _boom(*_a, **_k):
        raise RuntimeError("LLM exploded")

    monkeypatch.setattr(llm, "get_chat_model", _boom)
    failed = client.post(
        f"/runs/{thread_id}/answer",
        json={"answers": [{"question_id": "q-scope", "value": "tech"}]},
    )
    assert failed.status_code == 500

    with client.stream("GET", f"/runs/{thread_id}/events") as response:
        events = _collect_sse_events(response)

    assert [e["event"] for e in events] == ["plan_complete"]
    assert events[0]["data"]["payload"] == {"research_topics": []}


def test_real_main_graph_late_join_replays_plan_complete(monkeypatch, memory_checkpointer) -> None:
    """Full route + ``StreamingResponse`` + ``TestClient`` against a real
    main graph run to END with stubbed LLM/search (same stub shapes as
    ``test_graph_e2e.py``).

    The finished thread's ``current_step`` is ``"final"`` with topics
    populated, so the generator replays the ``plan_complete`` handshake
    once and closes — a FINITE stream, which is exactly what
    ``TestClient`` can consume. This pins the manual-smoke shape as a
    test: late joiners see the stage result, not a silent close.

    v0.3 adds the Plan→Research transition gate (ADR-0005, issue
    #23) between ``plan_questioner`` and the research subgraph. The
    test drives the gate resume directly via ``Command(resume=...)``
    (no ``/confirm`` endpoint yet — issue #24) so the main graph can
    advance through the gate into the research / write / finalize
    stages and reach the ``"final"`` terminal.
    """
    from app.graph import build_graph
    from app.models import ConfirmedResearchTopics
    from app.tools import llm, web_search

    topics_payload = {
        "topics": [
            {
                "id": "topic-1",
                "title": "History",
                "description": "history of X",
                "expected_finding_type": "factual",
            },
            {
                "id": "topic-2",
                "title": "Current state",
                "description": "current state of X",
                "expected_finding_type": "data",
            },
        ]
    }
    write_payload = {
        "outline": [
            {"title": "History", "summary": "s", "key_points": ["p"]},
            {"title": "Current state", "summary": "s", "key_points": ["p"]},
        ],
        "draft": (
            "Intro paragraph.\n\n"
            "## History\n\nFirst historical fact.\n\n"
            "## Current state\n\nFirst current fact."
        ),
    }

    # ready_to_proceed=true short-circuits the HITL loop (no questioner
    # interrupt at all — see plan.py): question-emit → topic-finalize
    # → GATE PAUSE. The main graph then needs the gate resume to
    # advance into research / write / finalize.
    stub_chat = _StubChat(
        [
            json.dumps(
                {
                    "questions": [],
                    "reasoning": "Have enough; ready to plan.",
                    "ready_to_proceed": True,
                }
            ),
            json.dumps(topics_payload),
            json.dumps(write_payload),
        ]
    )
    monkeypatch.setattr(llm, "get_chat_model", lambda: stub_chat)

    class _StubSearch:
        def invoke(self, payload, **_kwargs):
            topic_id = payload["topic_id"]
            return [
                {
                    "topic_id": topic_id,
                    "url": f"https://example.com/{topic_id}",
                    "title": f"Mock result {topic_id}",
                    "snippet": "snippet",
                    "score": 0.9,
                }
            ]

    stub_search = _StubSearch()
    monkeypatch.setattr(web_search, "get_web_search", lambda: stub_search)

    real_graph = build_graph(memory_checkpointer)
    thread_id = str(uuid.uuid4())
    config = _build_config(thread_id)

    # First invoke runs questioner (ready=true) → finalize → gate pause.
    # The parent's ``values`` only carries ``user_query`` at this point
    # — the plan subgraph's internal ``current_step`` / ``research_topics``
    # live inside the subgraph, not at the parent. The gate's interrupt
    # payload is reachable via ``tasks[0].interrupts[0].value``.
    real_graph.invoke({"user_query": "Tell me about X"}, config=config)
    paused_state = real_graph.get_state(config)
    assert paused_state.next, "gate must be paused (next non-empty)"

    # Drive the gate resume directly — no /confirm endpoint yet. The
    # test accepts the LLM draft unchanged so the diff is empty and
    # the test stays focused on the late-join SSE replay contract.
    from langgraph.types import Command

    gate_payload = paused_state.tasks[0].interrupts[0].value
    final_state = real_graph.invoke(
        Command(resume=ConfirmedResearchTopics(topics=list(gate_payload.topics))),
        config=config,
    )
    assert final_state["current_step"] == "final"

    app = FastAPI()
    app.include_router(router)
    app.state.graph = real_graph
    client = TestClient(app)

    with client.stream("GET", f"/runs/{thread_id}/events") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = _collect_sse_events(response)

    assert [e["event"] for e in events] == ["plan_complete"]
    assert events[0]["data"]["thread_id"] == thread_id
    topics = events[0]["data"]["payload"]["research_topics"]
    # v0.3 — the ids are uuid4 by ADR-0005 §2 (gate assigns them, not
    # the LLM). The test pins the human-visible shape (titles in
    # draft order) rather than the LLM's ``topic-1``/``topic-2``
    # slugs so the assertion stays stable.
    assert [t["title"] for t in topics] == ["History", "Current state"]


# ─────────────────────────────────────────────────────────────────────────────
# Stream-plan-events helper — direct unit coverage of the generator.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stream_plan_events_helper_emits_only_plan_question() -> None:
    """Calling ``stream_plan_events`` directly (no HTTP) still emits the
    right events. This is the documentation seam for libraries that
    want to embed the SSE generator without the FastAPI route."""
    interrupt = _FakeInterrupt(
        id_="only",
        value=PlanInterruptPayload(questions=[], reasoning="r", round=1),
    )

    class _G:
        def __init__(self) -> None:
            self.calls = 0

        def get_state(self, config):
            snapshot = _FakeSnapshot(values={}, tasks=[_FakeTask(interrupts=[interrupt])])
            # Mark subsequent calls to close the loop quickly.
            self.calls += 1
            if self.calls > 2:
                return None
            return snapshot

    captured: list[bytes] = []

    class _Req:
        async def is_disconnected(self) -> bool:
            return False

    req = _Req()
    gen = stream_plan_events(req, _G(), {"configurable": {"thread_id": "t"}}, "t")
    async for chunk in gen:
        captured.append(chunk)
        if len(captured) >= 1:
            break

    assert len(captured) == 1
    assert b"event: plan_question" in captured[0]
    assert b'"thread_id":"t"' in captured[0]


@pytest.mark.asyncio
async def test_stream_plan_events_helper_stops_on_disconnect() -> None:
    """``request.is_disconnected() == True`` halts the generator before the
    next poll — the client-side close path must NOT raise.
    """

    class _G:
        def __init__(self) -> None:
            self.calls = 0

        def get_state(self, config):
            # Implicit ``None`` return — the "unknown thread" sentinel the
            # generator treats as a close signal.
            self.calls += 1

    class _Req:
        async def is_disconnected(self) -> bool:
            return True

    graph = _G()
    gen = stream_plan_events(_Req(), graph, {"configurable": {"thread_id": "t"}}, "t")
    chunks = [c async for c in gen]
    assert chunks == []
    # The generator must short-circuit BEFORE calling ``get_state``.
    assert graph.calls == 0
