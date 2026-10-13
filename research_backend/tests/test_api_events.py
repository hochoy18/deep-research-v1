"""Ticket #16 — ``GET /runs/{thread_id}/events`` SSE tests (the two-AC file).

These two tests pin the two acceptance criteria called out in issue #16:

- ``test_sse_emits_plan_question_and_plan_complete`` — a scripted 2-round
  Q&A run yields exactly 2 ``plan_question`` events + 1 ``plan_complete``
  event in that order, with each ``data`` parsing as JSON shaped as
  ``{thread_id, ts, payload}``.
- ``test_sse_envelope_format`` — each event is rendered as the literal
  byte sequence ``event: <name>\ndata: <json>\n\n`` (matching what
  ``curl -N`` would print).

Both tests drive the real ``plan_subgraph`` compiled with ``MemorySaver``
+ the v0.1 ``fake_chat_model`` fixture (via ``patched_chat_model``); no
real LLM is called, no real network is touched, the v0.1 test suite
continues to pass.

The SSE consumer runs in a worker thread so it observes the in-flight
``interrupt()`` state transitions (round 1 → round 2 → final), not just
the late-join replay (which only ever replays ``plan_complete`` once
the stage has already advanced). The graph is driven to END on each
test so the SSE generator observes the ``next=()`` close signal and
shuts down — ``TestClient.stream`` deadlocks on infinite-poll streams
because Starlette's TestClient only yields ``http.disconnect`` after
the response body finishes (see the long docstring in
``test_api_sse_events.py`` for the same constraint), so a finite, graph
finalized, stream is a hard requirement under TestClient.
"""

from __future__ import annotations

import json
import re
import threading
import time
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import events as events_module
from app.api.routes import router
from app.graph.subgraphs.plan import build_plan_subgraph

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures + helpers.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _short_sse_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the SSE poll interval so tests don't pay 200ms per loop iteration.

    The generator reads ``SSE_POLL_INTERVAL_SECONDS`` on every poll via a
    closure (the module constant). Patching it once at the top of every
    test gives the in-worker-thread SSE consumer a chance to catch each
    state transition well within the test's hard wall-clock budget.
    """
    monkeypatch.setattr(events_module, "SSE_POLL_INTERVAL_SECONDS", 0.01)


def _questioner_json(questions: list[dict], reasoning: str, ready: bool) -> str:
    """One scripted ``PlanQuestionerBatch`` JSON for the fake questioner LLM."""
    return json.dumps({"questions": questions, "reasoning": reasoning, "ready_to_proceed": ready})


def _topics_json(topics: list[dict]) -> str:
    """One scripted ``PlanTopicsResponse`` JSON for the finalize LLM."""
    return json.dumps({"topics": topics})


def _build_app(checkpointer):
    """Compile the plan subgraph against ``checkpointer`` and wire it onto a FastAPI app.

    Returns the ``FastAPI`` instance ready for both ``POST /runs`` (and
    ``/answer``) from the main thread and ``GET /runs/{id}/events`` from
    a worker thread. ``MemorySaver`` lives on the checkpointer so the
    plan subgraph pauses / resumes across HTTP requests without any
    Redis dependency.
    """
    graph = build_plan_subgraph(checkpointer)
    app = FastAPI()
    app.include_router(router)
    app.state.graph = graph
    return app


def _consume_sse(app: FastAPI, thread_id: str, sink: list[bytes]) -> None:
    """Open the SSE stream for ``thread_id`` and collect raw bytes until the stream closes.

    Driven off a fresh ``TestClient`` (Starlette is per-instance stateful
    for streaming responses — sharing a client between threads races the
    portal). The function returns when ``iter_bytes`` exhausts, which
    happens once the SSE generator returns (graph at END).
    """
    client = TestClient(app)
    with client.stream("GET", f"/runs/{thread_id}/events") as response:
        sink.extend(response.iter_bytes())


def _split_sse_frames(raw: bytes) -> list[dict[str, Any]]:
    """Parse ``event: <name>\\ndata: <json>\\n\\n`` chunks into structured events.

    Splits on the SSE blank-line separator (``\\n\\n``), then iterates
    each frame's lines into an ``event`` and ``data`` pair. The ``data``
    line is JSON-decoded once so the test can assert on the envelope
    shape (``{thread_id, ts, payload}``) without re-doing the parsing
    in every test body.
    """
    events: list[dict[str, Any]] = []
    for frame in raw.split(b"\n\n"):
        if not frame:
            continue
        ev: dict[str, Any] = {}
        for line in frame.split(b"\n"):
            if line.startswith(b"event: "):
                ev["event"] = line[len(b"event: ") :].decode("utf-8")
            elif line.startswith(b"data: "):
                ev["data"] = json.loads(line[len(b"data: ") :].decode("utf-8"))
        if ev:
            events.append(ev)
    return events


# ─────────────────────────────────────────────────────────────────────────────
# Acceptance criterion 1 — exactly 2 plan_question + 1 plan_complete.
# ─────────────────────────────────────────────────────────────────────────────


def test_sse_emits_plan_question_and_plan_complete(patched_chat_model, memory_checkpointer) -> None:
    """Scripted 2-round Q&A run yields exactly 2 plan_question + 1 plan_complete.

    Flows through the standard questioner cadence — on entry AND on
    resume the questioner LLM is re-run, so the run consumes 6 LLM
    calls (1 + 2 + 3 across ``POST /runs`` and the two
    ``POST /runs/{id}/answer`` calls — matches the existing
    ``test_plan_subgraph_routes_back_until_research_step`` shape). The
    SSE consumer runs in a worker thread so it observes both
    ``interrupt()`` pauses (round 1 then round 2) plus the
    ``current_step="research"`` finalize handshake.
    """
    # The questioner LLM re-runs on every entry AND on every resume,
    # so each round consumes two LLM calls with the SAME batch
    # (initial emit + resume re-run). Each ``_questioner_json`` below
    # is therefore appended twice — once for the round's first entry,
    # once for the post-resume entry.
    q1_batch = _questioner_json(
        [
            {
                "kind": "single_choice",
                "id": "q1",
                "prompt": "Scope?",
                "options": [{"option_id": "tech", "label": "Tech"}],
            }
        ],
        reasoning="clarify scope",
        ready=False,
    )
    q2_batch = _questioner_json(
        [{"kind": "confirmation", "id": "q2", "prompt": "Confirm?"}],
        reasoning="confirm",
        ready=False,
    )
    ready_batch = _questioner_json([], reasoning="have enough", ready=True)
    topics_batch = _topics_json(
        [
            {
                "id": "topic-1",
                "title": "T1",
                "description": "d1",
                "expected_finding_type": "f1",
            }
        ]
    )

    # Script: round 1 initial → round 1 resume re-run → round 2 initial
    # → round 2 resume re-run → round 3 (ready=true) → topic-finalize.
    patched_chat_model.responses.extend(
        [q1_batch, q1_batch, q2_batch, q2_batch, ready_batch, topics_batch]
    )

    app = _build_app(memory_checkpointer)

    # 1. POST /runs — graph pauses at interrupt #1 (round 1).
    client = TestClient(app)
    response = client.post("/runs", json={"user_query": "Tell me about X"})
    assert response.status_code == 200, response.text
    thread_id = response.json()["thread_id"]

    # 2. Open SSE AFTER the run is created — opening before
    #    ``POST /runs`` would race the first poll (the generator
    #    closes on an empty snapshot, which it would see if it polled
    #    before the graph had anything to interrupt on).
    collected: list[bytes] = []
    sse_thread = threading.Thread(target=_consume_sse, args=(app, thread_id, collected))
    sse_thread.start()
    # Give the worker a few polls to observe interrupt #1. With the
    # 10ms patch this is comfortably over 5 polls.
    time.sleep(0.1)

    # 3. POST /answer #1 — resume, questioner re-runs with the same
    #    Q1 batch (ready=false), interrupt returns the answer, the
    #    loop continues, the questioner emits Q2 (ready=false), and
    #    the graph pauses at interrupt #2 (round 2).
    response = client.post(
        f"/runs/{thread_id}/answer",
        json={"answers": [{"question_id": "q1", "value": "tech"}]},
    )
    assert response.status_code == 200, response.text
    time.sleep(0.1)

    # 4. POST /answer #2 — resume, questioner re-runs with the same
    #    Q2 batch (ready=false), interrupt returns the answer, the
    #    loop continues, the questioner emits ready=true, finalize
    #    runs, and the subgraph lands at END with
    #    current_step="research".
    response = client.post(
        f"/runs/{thread_id}/answer",
        json={"answers": [{"question_id": "q2", "value": True}]},
    )
    assert response.status_code == 200, response.text

    # 5. Wait for the SSE worker to finish (graph at END → SSE
    #    generator returns on ``not snapshot.next``; ``iter_bytes``
    #    exhausts; ``client.stream`` exits its context manager; the
    #    worker function returns).
    sse_thread.join(timeout=10)
    assert not sse_thread.is_alive(), (
        "SSE stream should have closed once the plan subgraph reached END; "
        "worker thread is still running after 10s (likely polling forever "
        "because the graph never advanced to a terminal state)."
    )

    raw = b"".join(collected)
    events = _split_sse_frames(raw)

    # AC1 — exactly two ``plan_question`` events + one ``plan_complete``
    # event, in that order (the SSE dedupe key is the LangGraph
    # ``Interrupt.id``; each new ``interrupt()`` call gets a fresh id,
    # so the two rounds emit two events).
    assert [e["event"] for e in events] == [
        "plan_question",
        "plan_question",
        "plan_complete",
    ], f"unexpected event sequence (raw stream={raw!r})"

    # Each event's ``data`` parses as JSON shaped exactly as
    # ``{thread_id, ts, payload}`` (the locked-in v0.5 forward-compat
    # envelope called out in issue #14 / spec user-story #16).
    for e in events:
        assert set(e["data"].keys()) == {"thread_id", "ts", "payload"}
        assert e["data"]["thread_id"] == thread_id
        assert isinstance(e["data"]["ts"], int)
        assert e["data"]["ts"] > 0
        assert e["data"]["payload"], "payload must not be empty"

    # Spot-check the typed payload survives end-to-end: round indices
    # and the topic id flow from the LLM-scripted JSON through the
    # checkpointer to the SSE bytes without being normalised away.
    plan_question_1, plan_question_2, plan_complete = events
    assert plan_question_1["data"]["payload"]["round"] == 1
    assert plan_question_1["data"]["payload"]["questions"][0]["id"] == "q1"
    assert plan_question_2["data"]["payload"]["round"] == 2
    assert plan_question_2["data"]["payload"]["questions"][0]["id"] == "q2"
    assert plan_complete["data"]["payload"]["research_topics"][0]["id"] == "topic-1"


# ─────────────────────────────────────────────────────────────────────────────
# Acceptance criterion 2 — wire format is the SSE spec's literal bytes.
# ─────────────────────────────────────────────────────────────────────────────


def test_sse_envelope_format(patched_chat_model, memory_checkpointer) -> None:
    """Each event is rendered as ``event: <name>\\ndata: <json>\\n\\n``.

    Single-round Q&A so the SSE generator observes one ``plan_question``
    + one ``plan_complete`` in a single finite stream. The assertion is
    on the raw bytes — no JSON parser involved in the byte-shape check
    — so any divergence in line layout, whitespace, or framing
    separators surfaces as a regex mismatch.
    """
    # Four LLM responses for the 1-round Q&A flow (1 + 3 across
    # ``POST /runs`` and the single ``POST /runs/{id}/answer`` call —
    # matches the ``1 round → finalize`` cadence in
    # ``test_questioner_emits_typed_questions``). Same initial-vs-resume
    # pairing as test 1: each round's batch appears twice.
    q_format_batch = _questioner_json(
        [
            {
                "kind": "single_choice",
                "id": "q-format",
                "prompt": "Wire format check",
                "options": [{"option_id": "a", "label": "A"}],
            }
        ],
        reasoning="format",
        ready=False,
    )
    ready_batch = _questioner_json([], reasoning="done", ready=True)
    topics_batch = _topics_json(
        [
            {
                "id": "topic-1",
                "title": "T1",
                "description": "d",
                "expected_finding_type": "f",
            }
        ]
    )

    # Script: round 1 initial → round 1 resume re-run → round 2
    # (ready=true) → topic-finalize.
    patched_chat_model.responses.extend([q_format_batch, q_format_batch, ready_batch, topics_batch])

    app = _build_app(memory_checkpointer)

    client = TestClient(app)
    response = client.post("/runs", json={"user_query": "Tell me about Y"})
    assert response.status_code == 200, response.text
    thread_id = response.json()["thread_id"]

    collected: list[bytes] = []
    sse_thread = threading.Thread(target=_consume_sse, args=(app, thread_id, collected))
    sse_thread.start()
    time.sleep(0.1)

    # Resume → next-round ready=true → finalize → END. The SSE
    # generator will then observe ``plan_question`` (round 1),
    # ``plan_complete`` (current_step="research"), and close on
    # ``next=()``.
    response = client.post(
        f"/runs/{thread_id}/answer",
        json={"answers": [{"question_id": "q-format", "value": "a"}]},
    )
    assert response.status_code == 200, response.text

    sse_thread.join(timeout=10)
    assert not sse_thread.is_alive(), (
        "SSE stream should have closed once the plan subgraph reached END; "
        "worker thread is still running after 10s."
    )

    raw = b"".join(collected)

    # AC2 — wire format: every event is the literal bytes
    # ``event: <name>\ndata: <json>\n\n`` (matches what ``curl -N``
    # prints). The trailing ``\n\n`` blank-line separator is part of
    # the contract — a stream that ended with a lone ``\n`` or empty
    # bytes would pass a lenient regex but fail the SSE parser on the
    # client side, so pin the termination first.
    assert raw.endswith(b"\n\n"), (
        f"raw SSE stream must terminate with a blank-line separator "
        f"(\\n\\n); got tail={raw[-8:]!r} (full stream={raw!r})"
    )

    # Splitting on ``\n\n`` should give exactly one frame per event
    # plus a trailing empty entry (which the ``if not frame`` filter
    # drops). Any additional non-empty frames would mean a duplicate
    # emission or a malformed frame.
    frames = [f for f in raw.split(b"\n\n") if f]
    assert len(frames) == 2, (
        f"expected exactly 2 SSE frames (one of each event type), got "
        f"{len(frames)} (raw stream={raw!r})"
    )

    # Each frame is exactly two lines: ``event: <name>\n`` + ``data: <json>\n``,
    # with NO extra whitespace, comments, or per-key line breaks. The
    # first capture pins the event name; the second capture is the
    # JSON envelope (greedy ``.*`` reaches the closing ``}`` —
    # nested objects are fine because ``{.*}`` matches the full
    # payload until the last balanced char in this single-line frame).
    frame_pattern = re.compile(rb"^event: (\w+)\ndata: (\{.*\})$")
    for frame in frames:
        match = frame_pattern.match(frame)
        assert match, (
            f"frame did not match the locked-in envelope shape "
            f"event: <name>\\ndata: <json>\\n — got {frame!r} "
            f"(raw stream={raw!r})"
        )
        event_name = match.group(1).decode("utf-8")
        data_bytes = match.group(2)
        assert event_name in {"plan_question", "plan_complete"}, (
            f"unexpected event name {event_name!r}"
        )

        # The ``data:`` payload parses as JSON shaped as
        # ``{thread_id, ts, payload}`` — same envelope the route AC
        # pins, just verified directly on the raw bytes rather than via
        # the higher-level ``_split_sse_frames`` helper.
        envelope = json.loads(data_bytes.decode("utf-8"))
        assert set(envelope.keys()) == {"thread_id", "ts", "payload"}
        assert envelope["thread_id"] == thread_id
        assert isinstance(envelope["ts"], int)
        assert isinstance(envelope["payload"], dict)

    # The first frame is ``plan_question`` (round 1 interrupt), the
    # second is ``plan_complete`` (round 3 finalize) — the order is
    # part of the SSE contract too.
    assert frames[0].startswith(b"event: plan_question\n")
    assert frames[1].startswith(b"event: plan_complete\n")
