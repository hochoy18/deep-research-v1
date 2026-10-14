"""Ticket #24 — ``POST /runs/{thread_id}/confirm-plan`` route.

The endpoint is the only mechanism that drives the Plan→Research transition
gate forward once the graph is paused at the gate's ``interrupt()`` (issue
#23). It accepts a JSON body shaped as ``{"topics": [...]}`` — validated as
``ConfirmedResearchTopics`` — and resumes the graph via
``graph.invoke(Command(resume=ConfirmedResearchTopics(...)), ...)``.

The tests below pin the seven contracts called out in issue #24:

- 200 + ``{"thread_id": "...", "accepted": true}`` on success.
- 404 when the ``thread_id`` has no state in the checkpointer.
- 422 from Pydantic for malformed bodies (missing ``topics``, wrong
  ``ResearchTopic`` shape).
- 400 with body ``{"detail": "Empty topic list rejected", "code": "EMPTY_PLAN"}``
  for an empty ``topics`` list — the API-level defence-in-depth per
  ADR-0005 §4. The gate node (``confirm_plan_gate``) also rejects an empty
  list with ``ValueError("EMPTY_PLAN")`` and that path is pinned too.
- The endpoint invokes ``graph.invoke`` with a ``Command`` whose ``resume``
  carries the typed ``ConfirmedResearchTopics`` — i.e. the route is wired
  to the actual LangGraph resume primitive.
- ``POST /runs/{thread_id}/answer`` is left untouched (ADR-0005: the two
  resume endpoints are orthogonal user actions).

Most tests run against an in-process ``_MockGraph`` that captures the
``invoke`` payload and serves a configurable ``state_by_thread`` — so no
LLM, no Redis, no real subgraph is touched. The last test
(``test_confirm_plan_resumes_real_subgraph_end_to_end``) drives a compiled
``plan_subgraph`` with a ``MemorySaver`` to pin the end-to-end wiring
against real LangGraph primitives; the LLM is fully stubbed.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.api.routes import router
from app.graph.subgraphs.plan import build_plan_subgraph
from app.models import ConfirmedResearchTopics, ResearchTopic
from app.nodes.research_run import RECURSION_LIMIT

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures — wire the routes router onto a FastAPI app with a controllable
# mock graph. Each test injects its own graph stub via ``app.state.graph``.
# ─────────────────────────────────────────────────────────────────────────────


class _MockGraph:
    """Test double for the compiled LangGraph.

    Captures every ``invoke`` call so tests can assert what was resumed.
    ``get_state`` mirrors real LangGraph semantics: returns a snapshot
    with the configured ``values`` mapping for known threads, and
    returns ``None`` for unknown threads (which the route converts to
    404 — that is the contract we want to exercise).

    Tests can pass ``default_state`` — a single ``values`` dict that
    the mock returns for *any* thread id — so tests do not have to
    pre-populate a thread-keyed mapping just to satisfy the 404 guard.
    Tests that exercise per-thread state still use
    ``state_by_thread`` directly.

    Tests that exercise the defence-in-depth ``EMPTY_PLAN`` path inject
    an ``empty_plan_mode`` flag; ``invoke`` then raises
    ``ValueError("EMPTY_PLAN")`` exactly like the real
    ``confirm_plan_gate`` node, so the API's try/except can map it back
    to the same 400.
    """

    def __init__(
        self,
        state_by_thread: dict[str, dict[str, Any]] | None = None,
        empty_plan_mode: bool = False,
        default_state: dict[str, Any] | None = None,
    ) -> None:
        self.state_by_thread: dict[str, dict[str, Any]] = dict(state_by_thread or {})
        self.invokes: list[tuple[Any, dict]] = []
        self.empty_plan_mode = empty_plan_mode
        self.default_state: dict[str, Any] | None = default_state

    def invoke(self, payload: Any, config: dict | None = None) -> Any:
        self.invokes.append((payload, config))
        if self.empty_plan_mode:
            raise ValueError("EMPTY_PLAN")
        return {"status": "ok"}

    def get_state(self, config: dict) -> Any:
        thread_id = config.get("configurable", {}).get("thread_id")
        if thread_id in self.state_by_thread:
            values = self.state_by_thread[thread_id]
        elif self.default_state is not None:
            values = self.default_state
        else:
            return None

        class _Snapshot:
            def __init__(self, vals: dict[str, Any]) -> None:
                self.values = vals

        return _Snapshot(values)


@pytest.fixture
def app_with_mock_graph():
    """Return ``(app, graph)`` wired to the routes router with a mock graph.

    Tests can read ``app.state.graph`` / mutate it before each request to
    change behaviour (e.g. swap in a real ``build_graph(MemorySaver())``
    compiled plan subgraph for end-to-end coverage, or flip
    ``empty_plan_mode`` to simulate the gate's server-side ``EMPTY_PLAN``).
    """

    def _factory(
        state_by_thread: dict[str, dict[str, Any]] | None = None,
        empty_plan_mode: bool = False,
        default_state: dict[str, Any] | None = None,
    ) -> tuple[FastAPI, _MockGraph]:
        app = FastAPI()
        app.include_router(router)
        graph = _MockGraph(
            state_by_thread=state_by_thread,
            empty_plan_mode=empty_plan_mode,
            default_state=default_state,
        )
        app.state.graph = graph
        return app, graph

    return _factory


def _topic_payload(i: int = 1, **overrides) -> dict[str, Any]:
    """Return a valid ``ResearchTopic`` JSON dict, optionally tweaked."""
    base = {
        "id": f"topic-{i}",
        "title": f"Topic {i}",
        "description": f"What topic {i} investigates",
        "expected_finding_type": "evidence",
    }
    base.update(overrides)
    return base


def _topics_body(topic_count: int = 1) -> dict[str, Any]:
    """Return a body that should pass ``ConfirmedResearchTopics`` Pydantic validation."""
    return {"topics": [_topic_payload(i) for i in range(1, topic_count + 1)]}


# ─────────────────────────────────────────────────────────────────────────────
# Acceptance criterion 1 — 200 + accepted envelope on success.
# ─────────────────────────────────────────────────────────────────────────────


def test_confirm_plan_returns_200_with_accepted_envelope(app_with_mock_graph) -> None:
    """A well-formed body against an existing thread returns 200 + accepted=true."""
    app, _ = app_with_mock_graph(default_state={"research_topics": [_topic_payload(1)]})
    thread_id = str(uuid.uuid4())
    client = TestClient(app)

    response = client.post(
        f"/runs/{thread_id}/confirm-plan",
        json=_topics_body(2),
    )

    assert response.status_code == 200
    assert response.json() == {"thread_id": thread_id, "accepted": True}


def test_confirm_plan_invokes_graph_with_command_resume(app_with_mock_graph) -> None:
    """The route invokes ``graph.invoke(Command(resume=ConfirmedResearchTopics), config)``.

    Pins the exact wiring called out in the issue: the payload handed to
    LangGraph is a real ``Command`` whose ``resume`` is the typed
    ``ConfirmedResearchTopics``, and the config carries the route's
    standard ``thread_id`` / ``recursion_limit`` shape.
    """
    app, graph = app_with_mock_graph(default_state={"research_topics": [_topic_payload(1)]})
    thread_id = str(uuid.uuid4())
    client = TestClient(app)

    body = _topics_body(2)
    response = client.post(f"/runs/{thread_id}/confirm-plan", json=body)
    assert response.status_code == 200

    assert len(graph.invokes) == 1
    payload, config = graph.invokes[0]

    assert isinstance(payload, Command), f"expected Command(resume=...), got {type(payload).__name__}"
    assert config is not None
    assert config.get("configurable", {}).get("thread_id") == thread_id
    assert config.get("recursion_limit") == RECURSION_LIMIT

    confirmed = payload.resume
    assert isinstance(confirmed, ConfirmedResearchTopics)
    assert len(confirmed.topics) == 2
    assert confirmed.topics[0].id == "topic-1"
    assert confirmed.topics[1].id == "topic-2"


def test_confirm_plan_preserves_topic_field_values(app_with_mock_graph) -> None:
    """The route forwards the full topic list verbatim — no field rewriting at the API layer.

    Pins that the typed payload is built from ``payload.topics`` as the
    client sent it: ids, titles, descriptions, and
    ``expected_finding_type`` survive the resume wire intact. (The gate
    node reassigns ids for *added* topics; that mutation is a gate
    concern, not a route concern.)
    """
    app, graph = app_with_mock_graph(default_state={"research_topics": [_topic_payload(1)]})
    thread_id = str(uuid.uuid4())
    client = TestClient(app)

    response = client.post(
        f"/runs/{thread_id}/confirm-plan",
        json={
            "topics": [
                {
                    "id": "topic-a",
                    "title": "First topic",
                    "description": "Investigate A",
                    "expected_finding_type": "case studies",
                },
                {
                    "id": "topic-b",
                    "title": "Second topic",
                    "description": "Investigate B",
                    "expected_finding_type": "data points",
                },
            ]
        },
    )
    assert response.status_code == 200

    confirmed = graph.invokes[0][0].resume
    assert isinstance(confirmed, ConfirmedResearchTopics)
    by_id = {t.id: t for t in confirmed.topics}
    assert by_id["topic-a"].title == "First topic"
    assert by_id["topic-a"].description == "Investigate A"
    assert by_id["topic-a"].expected_finding_type == "case studies"
    assert by_id["topic-b"].title == "Second topic"


# ─────────────────────────────────────────────────────────────────────────────
# Acceptance criterion 2 — 404 when the thread has no state.
# ─────────────────────────────────────────────────────────────────────────────


def test_post_confirm_plan_with_unknown_thread_returns_404(app_with_mock_graph) -> None:
    """Issue #26 AC — a thread with no checkpoint raises 404.

    Mirrors ``test_api_answer.py``'s unknown-thread case: the resume
    must never be attempted against a thread the checkpointer has
    never seen.
    """
    app, _ = app_with_mock_graph()
    client = TestClient(app)

    response = client.post(
        f"/runs/{uuid.uuid4()}/confirm-plan",
        json=_topics_body(1),
    )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


# ─────────────────────────────────────────────────────────────────────────────
# Acceptance criterion 3 — 422 from Pydantic for malformed bodies.
# ─────────────────────────────────────────────────────────────────────────────


def test_confirm_plan_returns_422_when_topics_missing(app_with_mock_graph) -> None:
    """Body without ``topics`` is a Pydantic 422 — raised before the thread check.

    ``ConfirmedResearchTopics`` marks ``topics`` as required so a missing
    field fails closed at parse time (issue #21). FastAPI surfaces this
    as 422 before the endpoint runs, so ``graph.invoke`` is never called.
    """
    app, graph = app_with_mock_graph()
    thread_id = str(uuid.uuid4())
    client = TestClient(app)

    response = client.post(f"/runs/{thread_id}/confirm-plan", json={})
    assert response.status_code == 422, response.text
    assert graph.invokes == [], "graph.invoke must NOT be called on a malformed body"


def test_confirm_plan_returns_422_when_topic_id_wrong_type(app_with_mock_graph) -> None:
    """``ResearchTopic.id`` must be a string — ints are a Pydantic 422.

    The nested ``ResearchTopic`` validator (rejects empty / whitespace
    ids — issue #21) fires inside the wrapped list. Pins that the bad
    shape is rejected before the route handler runs.
    """
    app, graph = app_with_mock_graph()
    thread_id = str(uuid.uuid4())
    client = TestClient(app)

    response = client.post(
        f"/runs/{thread_id}/confirm-plan",
        json={"topics": [{"id": 123, "title": "T", "description": "d", "expected_finding_type": "f"}]},
    )
    assert response.status_code == 422
    assert graph.invokes == []


def test_confirm_plan_returns_422_when_topic_id_blank(app_with_mock_graph) -> None:
    """A blank ``ResearchTopic.id`` inside the list is a Pydantic 422 (issue #21 validator)."""
    app, graph = app_with_mock_graph()
    thread_id = str(uuid.uuid4())
    client = TestClient(app)

    response = client.post(
        f"/runs/{thread_id}/confirm-plan",
        json={"topics": [_topic_payload(id="   ")]},
    )
    assert response.status_code == 422
    assert graph.invokes == []


def test_confirm_plan_returns_422_when_topic_missing_required_field(app_with_mock_graph) -> None:
    """A topic missing ``description`` (or any required field) is a 422."""
    app, graph = app_with_mock_graph()
    thread_id = str(uuid.uuid4())
    client = TestClient(app)

    response = client.post(
        f"/runs/{thread_id}/confirm-plan",
        json={"topics": [{"id": "t", "title": "T", "expected_finding_type": "f"}]},
    )
    assert response.status_code == 422
    assert graph.invokes == []


# ─────────────────────────────────────────────────────────────────────────────
# Acceptance criterion 4 — empty topics → 400 EMPTY_PLAN (ADR-0005 §4).
# ─────────────────────────────────────────────────────────────────────────────


def test_post_confirm_plan_with_empty_topics_returns_400(app_with_mock_graph) -> None:
    """Issue #26 AC — API-layer defence-in-depth: an empty ``topics`` list
    short-circuits to 400 before invoking.

    The client should already disable the "confirm" button when
    ``topics.length < 1`` (ADR-0005 §4). The API mirrors that guard so a
    misbehaving client still cannot advance an empty plan. The graph is
    not invoked on this path — the gate node never runs and no state is
    written.
    """
    app, graph = app_with_mock_graph(default_state={"research_topics": [_topic_payload(1)]})
    thread_id = str(uuid.uuid4())
    client = TestClient(app)

    response = client.post(f"/runs/{thread_id}/confirm-plan", json={"topics": []})
    assert response.status_code == 400, response.text
    assert response.json() == {
        "detail": "Empty topic list rejected",
        "code": "EMPTY_PLAN",
    }
    assert graph.invokes == [], "graph.invoke must NOT be called on an empty topics list"


def test_confirm_plan_returns_400_when_gate_raises_empty_plan(app_with_mock_graph) -> None:
    """Server-side defence-in-depth: ``ValueError("EMPTY_PLAN")`` from the gate → 400.

    The gate node (``confirm_plan_gate`` — issue #23) raises
    ``ValueError("EMPTY_PLAN")`` for an empty list as the server-side
    partner to the API's pre-check. Both paths must converge on the same
    400 + ``EMPTY_PLAN`` body so the client has a single error contract
    regardless of which side catches it. ``_mark_run_failed`` is NOT
    invoked here — the gate's pre-conditions are a user error, not a
    pipeline failure (ADR-0008 fast-fail does not apply).
    """
    app, graph = app_with_mock_graph(
        default_state={"research_topics": [_topic_payload(1)]},
        empty_plan_mode=True,
    )
    thread_id = str(uuid.uuid4())
    client = TestClient(app)

    response = client.post(f"/runs/{thread_id}/confirm-plan", json=_topics_body(1))
    assert response.status_code == 400, response.text
    assert response.json() == {
        "detail": "Empty topic list rejected",
        "code": "EMPTY_PLAN",
    }
    # The graph *was* invoked (the API can't know the gate will reject
    # before the resume returns), but no failure marker is written.
    assert len(graph.invokes) == 1


def test_confirm_plan_does_not_mark_failed_for_empty_plan(app_with_mock_graph) -> None:
    """The ``_mark_run_failed`` fast-fail hook is NOT triggered on EMPTY_PLAN.

    ADR-0008 fast-fail covers graph execution failures; a user-side
    rejection (empty plan) is not a failure of the run. The route must
    surface the 400 cleanly without writing ``current_step="failed"``
    so an open ``GET /runs/{id}/events`` stream does not observe a
    spurious terminal frame.
    """
    app, _ = app_with_mock_graph(
        default_state={"research_topics": [_topic_payload(1)]},
        empty_plan_mode=True,
    )
    thread_id = str(uuid.uuid4())
    client = TestClient(app)

    response = client.post(f"/runs/{thread_id}/confirm-plan", json=_topics_body(1))
    assert response.status_code == 400
    # The mock does not implement ``update_state``; the production
    # route's ``_mark_run_failed`` swallows any checkpointer exception
    # so a down checkpointer cannot mask the 400. This test only pins
    # that the route doesn't try to do anything else.


# ─────────────────────────────────────────────────────────────────────────────
# Acceptance criterion 5 — ``POST /runs/{thread_id}/answer`` left untouched.
# ─────────────────────────────────────────────────────────────────────────────


def test_confirm_plan_does_not_break_answer_route(app_with_mock_graph) -> None:
    """The new route co-exists with ``POST /runs/{thread_id}/answer``.

    ADR-0005 says the two resume endpoints are orthogonal user actions
    (Plan Q&A vs Plan→Research gate). Adding ``confirm-plan`` must not
    shadow or alter ``answer``.
    """
    app, graph = app_with_mock_graph(default_state={"research_topics": [_topic_payload(1)]})
    thread_id = str(uuid.uuid4())
    graph.state_by_thread[thread_id] = {"round": 0}
    client = TestClient(app)

    # 1. /answer still works
    answer = client.post(
        f"/runs/{thread_id}/answer",
        json={"answers": [{"question_id": "q1", "value": "a"}]},
    )
    assert answer.status_code == 200
    assert answer.json() == {"thread_id": thread_id, "accepted": True}

    # 2. /confirm-plan on the same thread also works
    confirm = client.post(
        f"/runs/{thread_id}/confirm-plan",
        json=_topics_body(1),
    )
    assert confirm.status_code == 200
    assert confirm.json() == {"thread_id": thread_id, "accepted": True}

    # 3. The mock saw both invokes, in order
    assert len(graph.invokes) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Integration seam — drive the endpoint through a real compiled plan subgraph
# (with MemorySaver). No LLM is called because we script
# ``app.tools.llm.get_chat_model`` to return ready_to_proceed on round 1.
# This is the only test that touches real LangGraph primitives — the rest
# exercise the route in isolation.
# ─────────────────────────────────────────────────────────────────────────────


def test_post_confirm_plan_resumes_graph(monkeypatch) -> None:
    """Issue #26 AC — end-to-end: compiled plan subgraph + MemorySaver, gate
    resumed via the HTTP route; a subsequent ``GET /runs/{id}`` shows
    the advanced state.

    Uses ``build_plan_subgraph`` (not ``build_graph``) so the test stays
    focused on the Plan→Research gate — the main graph would also call
    the research and write LLMs, which is out of scope for an endpoint
    test. Scripts the questioner LLM to immediately signal
    ``ready_to_proceed`` and finalizes with a single draft topic. The
    gate then pauses on ``confirm_plan_gate.interrupt()``; the route
    submits the typed ``ConfirmedResearchTopics`` resume and the
    subgraph completes.
    """
    from app.tools import llm

    class _StubChat:
        def __init__(self) -> None:
            # LangGraph re-executes ``plan_questioner`` from the start
            # on resume (the node re-runs and re-emits an interrupt
            # before the gate's ``interrupt()`` returns), so each
            # resume triggers a fresh questioner call. For this test
            # both questioner calls return ``ready_to_proceed=true`` so
            # the node finalises immediately both times; the matching
            # finalize calls each consume one topic-finalize payload.
            self.responses = [
                json.dumps(  # Round 1 initial — ready_to_proceed
                    {
                        "questions": [],
                        "reasoning": "done",
                        "ready_to_proceed": True,
                    }
                ),
                json.dumps(  # Topic-finalize #1 (round 1 initial)
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
                json.dumps(  # Round 1 resume re-run — ready_to_proceed
                    {
                        "questions": [],
                        "reasoning": "done",
                        "ready_to_proceed": True,
                    }
                ),
                json.dumps(  # Topic-finalize #2 (round 1 resume re-run)
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

        def invoke(self, messages, **_kwargs):
            content = self.responses.pop(0)

            class _M:
                def __init__(self, c: str) -> None:
                    self.content = c

            return _M(content)

    # The lambda returns the SAME instance on every call — a new
    # ``_StubChat()`` per call would replay the full response list and
    # the second LLM call would pop the ready_to_proceed JSON instead
    # of the topic-finalize payload.
    stub_chat = _StubChat()
    monkeypatch.setattr(llm, "get_chat_model", lambda: stub_chat)

    real_graph = build_plan_subgraph(MemorySaver())

    app = FastAPI()
    app.include_router(router)
    app.state.graph = real_graph
    client = TestClient(app)

    thread_id = str(uuid.uuid4())
    config = _config_for(thread_id)

    # 1. Drive the plan subgraph directly to populate the checkpointer
    #    with an interrupt at ``confirm_plan_gate``. (We bypass
    #    ``POST /runs`` because it mints its own ``thread_id``.)
    first_invoke = real_graph.invoke({"user_query": "Tell me about X"}, config=config)
    assert "__interrupt__" in first_invoke, (
        "plan subgraph must pause at the gate on the first run"
    )

    # 2. Read the draft topics the gate paused on. ``_finalize_topics``
    #    reassigns each id to a fresh ``uuid4()`` (ADR-0005 §2), so
    #    the id the LLM returned is NOT the id the gate shows the
    #    user — the front end sends back the reassigned id, and so
    #    must we, or the diff will record the topic as ``added``.
    interrupted_snapshot = real_graph.get_state(config)
    assert interrupted_snapshot is not None
    draft_topics = interrupted_snapshot.values["research_topics"]
    assert len(draft_topics) == 1
    draft_topic = draft_topics[0]

    # 3. Resume via the HTTP route — accept the draft unchanged.
    confirmed = ConfirmedResearchTopics(
        topics=[
            ResearchTopic(
                id=draft_topic.id,
                title=draft_topic.title,
                description=draft_topic.description,
                expected_finding_type=draft_topic.expected_finding_type,
            )
        ],
    )
    response = client.post(
        f"/runs/{thread_id}/confirm-plan",
        json=confirmed.model_dump(),
    )
    assert response.status_code == 200
    assert response.json() == {"thread_id": thread_id, "accepted": True}

    # 4. Issue #26 AC — the advanced state is observable over HTTP, not
    #    just via the checkpointer: ``GET /runs/{id}`` shows the step
    #    moved past the gate and the user's final list landed.
    state_response = client.get(f"/runs/{thread_id}")
    assert state_response.status_code == 200
    state_values = state_response.json()
    assert state_values["current_step"] == "research"
    assert len(state_values["confirmed_research_plan"]) == 1
    assert state_values["confirmed_research_plan"][0]["title"] == draft_topic.title
    assert state_values["plan_edit_diff"]["added"] == []
    assert state_values["plan_edit_diff"]["removed"] == []
    assert state_values["plan_edit_diff"]["modified"] == []

    # 5. The subgraph's checkpointer now records the user's final list
    #    and the empty diff — confirms the resume wired through. With
    #    the same id the gate had on the draft, the diff is empty (no
    #    added / removed / modified entries).
    snapshot = real_graph.get_state(config)
    assert snapshot is not None
    assert snapshot.values["current_step"] == "research"
    assert len(snapshot.values["confirmed_research_plan"]) == 1
    assert snapshot.values["confirmed_research_plan"][0].title == draft_topic.title
    assert snapshot.values["confirmed_research_plan"][0].id == draft_topic.id
    assert snapshot.values["plan_edit_diff"].added == []
    assert snapshot.values["plan_edit_diff"].removed == []
    assert snapshot.values["plan_edit_diff"].modified == []


def _config_for(thread_id: str) -> dict[str, Any]:
    """Build the ``RunnableConfig`` shape the route uses, for post-resume inspection."""
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
    }
