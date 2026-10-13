"""Ticket #15 — ``POST /runs/{thread_id}/answer`` FastAPI tests.

These tests pin the three contracts called out in issue #15:

- 200 + advanced state on a valid round submission.
- 404 for an unknown ``thread_id`` (no state in the checkpointer).
- 422 for a malformed body (Pydantic validation).

All three tests run against a real ``plan_subgraph`` compiled with
``MemorySaver`` and the v0.1 ``fake_chat_model`` fixture (via
``patched_chat_model``). No real LLM is touched, no real network is
touched, and the v0.1 test suite still passes.

Why ``build_plan_subgraph`` (not ``build_graph``): the main graph also
calls the research + write LLMs, which is out of scope for an endpoint
test. Driving just the plan subgraph keeps the LLM script to two or
three responses and lets us assert on the subgraph's ``round`` and
``last_answers`` via ``GET /runs/{thread_id}``.
"""

from __future__ import annotations

import json
import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver

from app.api.routes import router
from app.graph.subgraphs.plan import build_plan_subgraph

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures — wire the routes router onto a FastAPI app with a real
# ``plan_subgraph`` compiled against ``MemorySaver``. The LLM is stubbed by
# the v0.1 ``fake_chat_model`` fixture (via ``patched_chat_model``) so the
# plan questioner LLM calls are scripted — no real LLM, no real network.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def app_with_plan_subgraph(patched_chat_model) -> FastAPI:
    """FastAPI app wired to a compiled plan subgraph + ``MemorySaver``.

    ``patched_chat_model`` is the v0.1 fixture from ``conftest.py`` — it
    replaces ``app.tools.llm.get_chat_model`` with the ``fake_chat_model``
    so the plan questioner LLM calls are scripted. ``MemorySaver`` is the
    in-process checkpointer; no Redis, no network.
    """
    graph = build_plan_subgraph(MemorySaver())
    app = FastAPI()
    app.include_router(router)
    app.state.graph = graph
    return app


@pytest.fixture
def client(app_with_plan_subgraph) -> TestClient:
    return TestClient(app_with_plan_subgraph)


def _questioner_payload(questions: list[dict], reasoning: str, ready: bool) -> str:
    """One scripted ``PlanQuestionerBatch`` for the fake questioner LLM."""
    return json.dumps(
        {"questions": questions, "reasoning": reasoning, "ready_to_proceed": ready}
    )


# ─────────────────────────────────────────────────────────────────────────────
# AC1 — POST /answer with a valid PlanAnswerBatch — graph advances.
# ─────────────────────────────────────────────────────────────────────────────


def test_post_answer_resumes_graph(patched_chat_model, client) -> None:
    """A valid round submission advances the graph; ``GET /runs`` shows the new round + answers.

    Three LLM calls are scripted on ``patched_chat_model.responses`` so the
    subgraph advances exactly once (round 0 → round 1) and then pauses at a
    second interrupt where ``round=1`` and ``last_answers=[A1]`` are stable
    in the checkpointer:

    1. ``POST /runs`` runs the subgraph; the first questioner call returns
       Q1 with ``ready_to_proceed=false`` so the subgraph pauses at the
       first interrupt.
    2. ``POST /runs/{thread_id}/answer`` resumes — the second questioner
       call (round-1 resume re-run) returns Q2 with ``ready_to_proceed=false``;
       the node appends the Human turn, bumps ``round`` to 1, writes
       ``last_answers=[A1]``, then the conditional edge routes back.
    3. The third questioner call (round-2 initial) returns Q3 with
       ``ready_to_proceed=false`` and the subgraph pauses at the second
       interrupt — this is the state ``GET /runs`` returns.
    """
    # Round 1 initial — Q1, ready=false → graph pauses at 1st interrupt.
    patched_chat_model.responses.append(
        _questioner_payload(
            [
                {
                    "kind": "single_choice",
                    "id": "q1",
                    "prompt": "Scope?",
                    "options": [{"option_id": "a", "label": "A"}],
                }
            ],
            reasoning="scope",
            ready=False,
        )
    )
    # Round 1 resume re-run — Q2, ready=false. The resume appends a Human
    # turn to messages, bumps round to 1, writes last_answers=[A1] and
    # last_questions=[Q2], then the conditional edge routes back.
    patched_chat_model.responses.append(
        _questioner_payload(
            [{"kind": "confirmation", "id": "q2", "prompt": "Confirm?"}],
            reasoning="confirm",
            ready=False,
        )
    )
    # Round 2 initial — Q3, ready=false. The subgraph pauses at the 2nd
    # interrupt with round=1 and last_answers=[A1] persisted in MemorySaver,
    # which is the state GET /runs returns.
    patched_chat_model.responses.append(
        _questioner_payload(
            [{"kind": "free_text", "id": "q3", "prompt": "Anything else?"}],
            reasoning="details",
            ready=False,
        )
    )

    # 1. Create a run — the subgraph runs to its first interrupt.
    response = client.post("/runs", json={"user_query": "Tell me about X"})
    assert response.status_code == 200, response.text
    thread_id = response.json()["thread_id"]

    # 2. POST a valid answer — the subgraph resumes and re-runs the questioner.
    response = client.post(
        f"/runs/{thread_id}/answer",
        json={"answers": [{"question_id": "q1", "value": "a"}]},
    )
    assert response.status_code == 200, response.text
    assert response.json() == {"thread_id": thread_id, "accepted": True}

    # 3. GET /runs — verify the checkpointer now records the advanced round
    #    and the answer we just submitted.
    response = client.get(f"/runs/{thread_id}")
    assert response.status_code == 200, response.text
    state = response.json()
    assert state["round"] == 1, f"expected round=1, got state={state!r}"
    assert len(state["last_answers"]) == 1, (
        f"expected 1 last_answer, got state={state!r}"
    )
    assert state["last_answers"][0]["question_id"] == "q1"
    assert state["last_answers"][0]["value"] == "a"

    # Pin the seam: the resume re-run recorded Q2 as ``last_questions`` and
    # exactly three questioner LLM calls ran (round-1 emit, resume re-run,
    # round-2 emit). A fourth call would mean the graph finalized instead of
    # pausing at the second interrupt — the persisted round/answers alone
    # can't tell the two apart, so this is what makes the advance visible.
    assert state["last_questions"][0]["id"] == "q2"
    assert len(patched_chat_model.calls) == 3, (
        f"expected 3 questioner LLM calls (paused at 2nd interrupt), "
        f"got {len(patched_chat_model.calls)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# AC2 — POST /answer with unknown thread_id returns 404.
# ─────────────────────────────────────────────────────────────────────────────


def test_post_answer_unknown_thread_returns_404(client) -> None:
    """A ``thread_id`` with no state in ``MemorySaver`` returns 404.

    ``MemorySaver.get_state`` returns a ``StateSnapshot`` with empty
    ``values`` (``{}``, not ``None``) for a thread_id that was never
    checkpointed. The route treats that as "no state" and raises 404
    before invoking the graph.
    """
    unknown_thread_id = str(uuid.uuid4())
    response = client.post(
        f"/runs/{unknown_thread_id}/answer",
        json={"answers": [{"question_id": "q1", "value": "a"}]},
    )
    assert response.status_code == 404, response.text
    assert "not found" in response.json()["detail"].lower()

    # The 404 must short-circuit before ``graph.invoke`` — a resume that
    # ran would checkpoint the thread. The thread is still unknown to the
    # checkpointer, so the route never touched the graph.
    followup = client.get(f"/runs/{unknown_thread_id}")
    assert followup.status_code == 404, (
        "POST /answer must not have created any state for the unknown thread"
    )


# ─────────────────────────────────────────────────────────────────────────────
# AC3 — POST /answer with malformed body returns 422.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({}, id="missing-answers"),
        pytest.param(
            {"answers": [{"question_id": 123, "value": "a"}]},
            id="wrong-question-id-type-int",
        ),
        pytest.param(
            {"answers": [{"question_id": "q1", "value": 123}]},
            id="wrong-value-type-int",
        ),
        pytest.param(
            {"answers": [{"question_id": "q1", "value": {"foo": "bar"}}]},
            id="wrong-value-type-dict",
        ),
    ],
)
def test_post_answer_malformed_body_returns_422(body, client) -> None:
    """Malformed body returns 422 (Pydantic validation, before the graph is touched).

    FastAPI rejects the body at request-parse time — the route handler is
    never invoked, so the graph (and its checkpointer) are not touched.
    A random ``thread_id`` is used because the 422 path runs before the
    thread-existence check.
    """
    thread_id = str(uuid.uuid4())
    response = client.post(f"/runs/{thread_id}/answer", json=body)
    assert response.status_code == 422, f"body={body!r}: {response.text}"

    # Pin "never reaches the graph": if any part of the handler had run
    # (e.g. a future reorder that invokes before validating), the resume
    # would write checkpoint state and GET would stop returning 404.
    followup = client.get(f"/runs/{thread_id}")
    assert followup.status_code == 404, (
        f"body={body!r}: malformed POST must not touch the checkpointer"
    )
