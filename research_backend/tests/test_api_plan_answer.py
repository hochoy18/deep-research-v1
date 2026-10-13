"""Ticket #13 — ``POST /runs/{thread_id}/answer`` route.

The endpoint is the only way to drive a Plan HITL round forward once the
graph is paused at ``interrupt()``. It accepts a JSON body shaped as
``{"answers": [{"question_id": "...", "value": <json>}, ...]}`` (the
``round`` is filled in server-side from the checkpointer so the client
never has to track it), wraps it in a typed ``PlanAnswerBatch``, and
resumes the graph via ``graph.invoke(Command(resume=...), ...)``.

The tests below pin the four contracts called out in issue #13:

- 200 + ``{"thread_id": "...", "accepted": true}`` on success.
- 404 when the ``thread_id`` has no state in the checkpointer.
- 422 from Pydantic for malformed bodies (missing ``answers``, wrong
  ``question_id`` type, wrong ``value`` shape).
- The endpoint invokes ``graph.invoke`` with a ``Command`` whose
  ``resume`` carries the typed ``PlanAnswerBatch`` — i.e. the route is
  wired to the actual LangGraph resume primitive.

Most tests run against an in-process ``_MockGraph`` that captures the
``invoke`` payload and serves a configurable ``state_by_thread`` — so no
LLM, no Redis, no real subgraph is touched. The last test
(``test_answer_resumes_real_subgraph_end_to_end``) drives a compiled
``plan_subgraph`` with a ``MemorySaver`` to pin the end-to-end wiring
against real LangGraph primitives; the LLM is fully stubbed.

``tests/test_api_health.py`` is left untouched (the acceptance criterion
explicitly requires it to remain green).
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
from app.models import PlanAnswerBatch
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
    """

    def __init__(self, state_by_thread: dict[str, dict[str, Any]] | None = None) -> None:
        self.state_by_thread: dict[str, dict[str, Any]] = dict(state_by_thread or {})
        self.invokes: list[tuple[Any, dict]] = []

    def invoke(self, payload: Any, config: dict | None = None) -> Any:
        self.invokes.append((payload, config))
        # Record the answer round so ``get_state`` would simulate progress
        # if a test wants to chain a follow-up resume against the same
        # mock thread. Mirrors LangGraph's checkpointer behaviour: each
        # resume writes the new state on top of the prior one.
        if isinstance(payload, Command) and config:
            thread_id = config.get("configurable", {}).get("thread_id")
            cmd_resume = getattr(payload, "resume", None)
            if isinstance(cmd_resume, PlanAnswerBatch) and thread_id is not None:
                prior = self.state_by_thread.get(thread_id, {})
                self.state_by_thread[thread_id] = {
                    **prior,
                    "round": cmd_resume.round,
                    "last_answers": list(cmd_resume.answers),
                }
        return {"status": "ok"}

    def get_state(self, config: dict) -> Any:
        thread_id = config.get("configurable", {}).get("thread_id")
        if thread_id not in self.state_by_thread:
            return None
        values = self.state_by_thread[thread_id]

        class _Snapshot:
            def __init__(self, vals: dict[str, Any]) -> None:
                self.values = vals

        return _Snapshot(values)


@pytest.fixture
def app_with_mock_graph() -> tuple[FastAPI, _MockGraph]:
    """Return ``(app, graph)`` wired to the routes router with a mock graph.

    Tests can read ``app.state.graph`` / mutate it before each request to
    change behaviour (e.g. swap in a real ``build_graph(MemorySaver())``
    compiled plan subgraph for end-to-end coverage).
    """
    app = FastAPI()
    app.include_router(router)
    graph = _MockGraph()
    app.state.graph = graph
    return app, graph


@pytest.fixture
def client(app_with_mock_graph) -> TestClient:
    """FastAPI TestClient bound to the mocked app."""
    app, _ = app_with_mock_graph
    return TestClient(app)


# ─────────────────────────────────────────────────────────────────────────────
# Acceptance criterion 1 — 200 + accepted envelope on success.
# ─────────────────────────────────────────────────────────────────────────────


def test_answer_returns_200_with_accepted_envelope(app_with_mock_graph) -> None:
    """A well-formed body against an existing thread returns 200 + accepted=true."""
    app, graph = app_with_mock_graph
    thread_id = str(uuid.uuid4())
    # Pretend a prior round has already paused the graph at round=0 (the
    # internal 0-based counter — user-facing round 1).
    graph.state_by_thread[thread_id] = {"round": 0, "last_questions": [], "last_answers": []}

    client = TestClient(app)
    response = client.post(
        f"/runs/{thread_id}/answer",
        json={"answers": [{"question_id": "q1", "value": "a"}]},
    )

    assert response.status_code == 200
    assert response.json() == {"thread_id": thread_id, "accepted": True}


def test_answer_invokes_graph_with_command_resume(app_with_mock_graph) -> None:
    """The route invokes ``graph.invoke(Command(resume=PlanAnswerBatch), config)``.

    Pins the exact wiring called out in the issue: the payload handed to
    LangGraph is a real ``Command`` whose ``resume`` is the typed
    ``PlanAnswerBatch``, and the config carries the route's standard
    ``thread_id`` / ``recursion_limit`` shape.
    """
    app, graph = app_with_mock_graph
    thread_id = str(uuid.uuid4())
    graph.state_by_thread[thread_id] = {"round": 0}

    client = TestClient(app)
    response = client.post(
        f"/runs/{thread_id}/answer",
        json={"answers": [{"question_id": "q-scope", "value": "tech"}]},
    )
    assert response.status_code == 200

    assert len(graph.invokes) == 1
    payload, config = graph.invokes[0]

    assert isinstance(payload, Command), f"expected Command(resume=...), got {type(payload).__name__}"
    assert config is not None
    assert config.get("configurable", {}).get("thread_id") == thread_id
    assert config.get("recursion_limit") == RECURSION_LIMIT

    answer_batch = payload.resume
    assert isinstance(answer_batch, PlanAnswerBatch)
    # Round is filled in server-side from the persisted state. Internal
    # state["round"] is 0-indexed; the user-facing PlanAnswerBatch.round
    # is 1-indexed (matches ``PlanInterruptPayload.round``).
    assert answer_batch.round == 1
    assert len(answer_batch.answers) == 1
    assert answer_batch.answers[0].question_id == "q-scope"
    assert answer_batch.answers[0].value == "tech"


def test_answer_preserves_value_types(app_with_mock_graph) -> None:
    """``value`` is passed through verbatim: ``str | list[str] | bool`` are all accepted."""
    app, graph = app_with_mock_graph
    thread_id = str(uuid.uuid4())
    graph.state_by_thread[thread_id] = {"round": 1}

    client = TestClient(app)
    response = client.post(
        f"/runs/{thread_id}/answer",
        json={
            "answers": [
                {"question_id": "q-single", "value": "tech"},
                {"question_id": "q-multi", "value": ["a", "b", "c"]},
                {"question_id": "q-confirm", "value": False},
            ]
        },
    )
    assert response.status_code == 200

    _, config = graph.invokes[0]
    assert config["configurable"]["thread_id"] == thread_id
    answer_batch = graph.invokes[0][0].resume
    assert isinstance(answer_batch, PlanAnswerBatch)
    by_id = {a.question_id: a for a in answer_batch.answers}
    assert by_id["q-single"].value == "tech"
    assert by_id["q-multi"].value == ["a", "b", "c"]
    assert by_id["q-confirm"].value is False


def test_answer_round_advances_with_state(app_with_mock_graph) -> None:
    """Round is derived from the persisted state — a thread paused at internal round=2 resumes as user-facing round 3."""
    app, graph = app_with_mock_graph
    thread_id = str(uuid.uuid4())
    # Simulate prior resume round-trips that bumped the internal round counter to 2.
    graph.state_by_thread[thread_id] = {"round": 2}

    client = TestClient(app)
    response = client.post(
        f"/runs/{thread_id}/answer",
        json={"answers": [{"question_id": "q1", "value": "a"}]},
    )
    assert response.status_code == 200

    answer_batch = graph.invokes[0][0].resume
    assert isinstance(answer_batch, PlanAnswerBatch)
    assert answer_batch.round == 3, "user-facing round = persisted round + 1"


def test_answer_writes_resumed_round_back_to_state(app_with_mock_graph) -> None:
    """The mock graph's resume rewriter updates ``state_by_thread[thread_id]['round']`` —
    pins that the resume payload is itself a typed ``PlanAnswerBatch`` so
    the checkpointer can serialize it losslessly.
    """
    app, graph = app_with_mock_graph
    thread_id = str(uuid.uuid4())
    graph.state_by_thread[thread_id] = {"round": 0}

    client = TestClient(app)
    response = client.post(
        f"/runs/{thread_id}/answer",
        json={"answers": [{"question_id": "q1", "value": "a"}]},
    )
    assert response.status_code == 200
    # The mock's invoke() rebumped the round counter to the user-facing value 1.
    assert graph.state_by_thread[thread_id]["round"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# Acceptance criterion 2 — 404 when the thread has no state.
# ─────────────────────────────────────────────────────────────────────────────


def test_answer_returns_404_when_thread_unknown(app_with_mock_graph) -> None:
    """A thread with no checkpoint raises 404."""
    app, _ = app_with_mock_graph
    client = TestClient(app)

    # ``_MockGraph.get_state`` raises ``KeyError`` for unknown threads —
    # mirrors LangGraph's behaviour. The route converts that to 404.
    response = client.post(
        f"/runs/{uuid.uuid4()}/answer",
        json={"answers": [{"question_id": "q1", "value": "a"}]},
    )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


# ─────────────────────────────────────────────────────────────────────────────
# Acceptance criterion 3 — 422 from Pydantic for malformed bodies.
# ─────────────────────────────────────────────────────────────────────────────


def test_answer_returns_422_when_answers_missing(app_with_mock_graph) -> None:
    """Body without ``answers`` is a Pydantic 422 — raised before the thread check.

    The same ``thread_id`` is pre-populated in the mock and used in the
    request so the 422 path is exercised (otherwise an unknown thread
    short-circuits to 404 before Pydantic validation gets a chance to
    run). Pydantic body validation is performed by FastAPI before the
    endpoint function is invoked, so ``graph.invoke`` must never be
    called.
    """
    app, graph = app_with_mock_graph
    thread_id = str(uuid.uuid4())
    graph.state_by_thread[thread_id] = {"round": 0}

    client = TestClient(app)
    response = client.post(
        f"/runs/{thread_id}/answer",
        json={},
    )
    assert response.status_code == 422, response.text
    assert graph.invokes == [], "graph.invoke must NOT be called on a malformed body"


def test_answer_returns_422_when_question_id_wrong_type(app_with_mock_graph) -> None:
    """``question_id`` must be a string — ints are a Pydantic 422."""
    app, graph = app_with_mock_graph
    thread_id = str(uuid.uuid4())
    graph.state_by_thread[thread_id] = {"round": 0}

    client = TestClient(app)
    response = client.post(
        f"/runs/{thread_id}/answer",
        json={"answers": [{"question_id": 123, "value": "a"}]},
    )
    assert response.status_code == 422
    assert graph.invokes == []


@pytest.mark.parametrize(
    "bad_value",
    [
        pytest.param(123, id="int"),
        pytest.param(1.5, id="float"),
        pytest.param({"key": "val"}, id="dict"),
        pytest.param(None, id="null"),
    ],
)
def test_answer_returns_422_when_value_wrong_shape(
    bad_value: Any, app_with_mock_graph
) -> None:
    """``value`` accepts only ``str | list[str] | bool`` — anything else is 422.

    The Plan subgraph re-emits questions if a ``question_id`` is unknown
    (v0.2 does not enforce question_id set membership), so the route does
    no semantic validation beyond the Pydantic schema.
    """
    app, graph = app_with_mock_graph
    thread_id = str(uuid.uuid4())
    graph.state_by_thread[thread_id] = {"round": 0}

    client = TestClient(app)
    response = client.post(
        f"/runs/{thread_id}/answer",
        json={"answers": [{"question_id": "q1", "value": bad_value}]},
    )
    assert response.status_code == 422
    assert graph.invokes == []


def test_answer_returns_422_when_value_list_contains_non_strings(
    app_with_mock_graph,
) -> None:
    """A ``value`` list with non-strings is rejected so the typed list flows as ``list[str]``."""
    app, graph = app_with_mock_graph
    thread_id = str(uuid.uuid4())
    graph.state_by_thread[thread_id] = {"round": 0}

    client = TestClient(app)
    response = client.post(
        f"/runs/{thread_id}/answer",
        json={"answers": [{"question_id": "q-mc", "value": ["a", 2]}]},
    )
    assert response.status_code == 422
    assert graph.invokes == []


# ─────────────────────────────────────────────────────────────────────────────
# Integration seam — drive the endpoint through a real compiled plan subgraph
# (with MemorySaver). No LLM is called because we script
# ``app.tools.llm.get_chat_model`` to return ready_to_proceed on round 1.
# This is the only test that touches real LangGraph primitives — the rest
# exercise the route in isolation.
# ─────────────────────────────────────────────────────────────────────────────


def test_answer_resumes_real_subgraph_end_to_end(monkeypatch) -> None:
    """End-to-end: compiled plan subgraph + MemorySaver, resumed via the HTTP route.

    Uses ``build_plan_subgraph`` (not ``build_graph``) so the test stays
    focused on the Plan HITL loop — the main graph would also call the
    research and write LLMs, which is out of scope for an endpoint test.
    Scripts the questioner LLM to ask one question, the resume answer
    advances to round 2, which signals ``ready_to_proceed`` and finalizes.
    No real LLM is hit.
    """
    from app.tools import llm

    q1_question = {
        "questions": [
            {
                "kind": "single_choice",
                "id": "q1",
                "prompt": "Scope?",
                "options": [{"option_id": "tech", "label": "Tech"}],
            }
        ],
        "reasoning": "scope",
        "ready_to_proceed": False,
    }

    class _StubChat:
        def __init__(self) -> None:
            self.responses = [
                json.dumps(q1_question),  # Round 1 initial
                json.dumps(q1_question),  # Round 1 resume re-run
                json.dumps(  # Round 2: ready_to_proceed
                    {"questions": [], "reasoning": "done", "ready_to_proceed": True}
                ),
                json.dumps(  # Topic-finalize
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

    monkeypatch.setattr(llm, "get_chat_model", lambda: _StubChat())

    real_graph = build_plan_subgraph(MemorySaver())

    app = FastAPI()
    app.include_router(router)
    app.state.graph = real_graph
    client = TestClient(app)

    thread_id = str(uuid.uuid4())
    config = _config_for(thread_id)

    # 1. Drive the plan subgraph directly to populate the checkpointer
    #    with an interrupt for ``thread_id``. (We bypass the
    #    ``POST /runs`` route because it mints its own ``thread_id`` —
    #    re-targeting the resume to our chosen id via a direct
    #    ``invoke`` keeps the test focused on the answer endpoint.)
    first_invoke = real_graph.invoke({"user_query": "Tell me about X"}, config=config)
    assert "__interrupt__" in first_invoke, "plan subgraph must pause on the first round"

    # 2. Resume via the HTTP route.
    answer = client.post(
        f"/runs/{thread_id}/answer",
        json={"answers": [{"question_id": "q1", "value": "tech"}]},
    )
    assert answer.status_code == 200
    assert answer.json() == {"thread_id": thread_id, "accepted": True}

    # 3. The plan subgraph's checkpointer now records round=1 and the
    #    answer — confirms the resume wired through to the graph state
    #    the typed payload typed ``last_answers``.
    snapshot = real_graph.get_state(config)
    assert snapshot is not None
    assert snapshot.values["round"] == 1
    assert len(snapshot.values["last_answers"]) == 1
    assert snapshot.values["last_answers"][0].question_id == "q1"
    assert snapshot.values["last_answers"][0].value == "tech"


def _config_for(thread_id: str) -> dict[str, Any]:
    """Build the ``RunnableConfig`` shape the route uses, for post-resume inspection."""
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
    }
