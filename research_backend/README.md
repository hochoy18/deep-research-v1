# research_backend — Deep Research Agent Backend (v0.2)

End-to-end "question → cited report" pipeline with a multi-round Plan HITL loop + SSE event stream (v0.2; no transition gate, no review / polish). Built on LangGraph + FastAPI.

## Stack

- **LangGraph** — stateful multi-stage graph (ADR-0001)
- **OpenAI GPT-4.x** — LLM for plan / outline / draft / citation (via `langchain-openai`)
- **阿里云百炼 DashScope** — web search via OpenAI-compatible chat completion with `extra_body={"enable_search": True}` (ADR-0002)
- **Redis 8.0+** (with RedisJSON + RediSearch) — LangGraph checkpointer (ADR-0003)
- **FastAPI** — `POST /runs`, `GET /runs/{thread_id}`, `POST /runs/{thread_id}/answer`, `GET /runs/{thread_id}/events` (SSE), `GET /health`
- **uv** — dependency management

## Required Redis version

> **Redis 8.0 or newer** is required, with the **RedisJSON** and **RediSearch** modules loaded. The `langgraph-checkpoint-redis` saver uses RediSearch indexes that `RedisSaver.setup()` creates on first startup. Older Redis versions will silently fail to persist thread state.

The fastest local setup is the `redis/redis-stack` Docker image, which bundles both modules:

```bash
docker run -d --name redis-stack -p 6379:6379 redis/redis-stack:latest
```

## Local development

```bash
cd research_backend
uv sync
cp .env.example .env  # then fill in OPENAI_API_KEY + DASHSCOPE_API_KEY
uv run uvicorn app.main:app --reload
```

Then:

```bash
curl http://localhost:8000/health
# {"status":"ok"}

curl -X POST http://localhost:8000/runs \
  -H "content-type: application/json" \
  -d '{"user_query": "对比 Claude 4.5 Sonnet 与 GPT-4o 在多轮对话中的推理深度差异"}'
# {"thread_id":"<uuid>"}

curl http://localhost:8000/runs/<thread_id>
# full State including report (Markdown with [N] citations + ## References)
```

## v0.2 Plan HITL + SSE

The Plan stage interviews the user before any web search runs. `POST /runs`
synchronously drives the graph to its first `interrupt()` and returns; the
graph stays paused until answers are submitted. v0.2 ships **two** new
HTTP endpoints (`POST /runs/{id}/answer`, `GET /runs/{id}/events` — SSE) and
the **5-round Q&A cap**; nothing else from the v0.1 pipeline changes.

### `POST /runs/{thread_id}/answer` — submit one round of answers

The single mechanism that drives the Plan HITL loop forward once the
questioner has emitted a batch. Wraps the body in a typed
`PlanAnswerBatch` and resumes the graph via
`graph.invoke(Command(resume=...))`. Returns `200 {thread_id, accepted: true}`
on success, `404` if the `thread_id` has no checkpointer state, and `422`
(Pydantic) for a malformed body.

**curl**:

```bash
# 1. Start a run — returns thread_id; the graph is paused at round 1.
curl -X POST http://localhost:8000/runs \
  -H "content-type: application/json" \
  -d '{"user_query": "对比 Claude 4.5 与 GPT-4o 的推理深度"}'
# {"thread_id":"<uuid>"}

# 2. Answer the round's questions in one batch — the graph resumes.
curl -X POST http://localhost:8000/runs/<thread_id>/answer \
  -H "content-type: application/json" \
  -d '{"answers": [{"question_id": "q-scope", "value": "tech"}]}'
# {"thread_id":"<uuid>","accepted":true}
```

**Python** (`httpx`, already a project dependency):

```python
import httpx

BASE = "http://localhost:8000"
with httpx.Client(base_url=BASE, timeout=30) as client:
    # 1. Start a run — graph is paused at round 1.
    thread_id = client.post(
        "/runs", json={"user_query": "对比 Claude 4.5 与 GPT-4o 的推理深度"}
    ).json()["thread_id"]

    # 2. Submit one round of answers. ``value`` narrows by question kind:
    #    single_choice / free_text → str, multi_choice → list[str],
    #    confirmation → bool. Unknown ``question_id``s are accepted as-is
    #    in v0.2 — the route does not enforce set membership, and the
    #    Plan subgraph re-emits questions on every resume regardless.
    resp = client.post(
        f"/runs/{thread_id}/answer",
        json={"answers": [{"question_id": "q-scope", "value": "tech"}]},
    )
    resp.raise_for_status()
    assert resp.json() == {"thread_id": thread_id, "accepted": True}
```

### `GET /runs/{thread_id}/events` — tail the Plan lifecycle as SSE

Streams two event types in the v0.5 forward-compatible envelope:

- **`plan_question`** — payload = `PlanInterruptPayload.model_dump()` =
  `{questions: list[PlanQuestion], reasoning: str, round: int}` (1-based
  round). Fired when the Plan subgraph enters `interrupt()` on a new round.
- **`plan_complete`** — payload = `{research_topics: list[ResearchTopic]}`.
  Fired exactly once per connection as soon as `current_step` advances
  past `"plan"` (`"research"`, `"write"`, `"final"`, or `"failed"`). On a
  failed run the topic list is empty (the dedicated `error` event lands
  in v0.5).

Every frame is `event: <name>\ndata: {"thread_id":"...", "ts":<unix-ms>, "payload":{...}}\n\n`.
The stream closes when any of these is observed: the main graph reaches
END (`current_step == "final"`), the graph fails (`current_step == "failed"`,
written by the route before propagating the 500 — see ADR-0008 fast-fail),
the client disconnects, or the thread is unknown. A late joiner whose
stage already advanced sees a replayed `plan_complete` before the close.

**curl**:

```bash
curl -N http://localhost:8000/runs/<thread_id>/events
# Round 1 starts — the Plan subgraph interrupts, so the stream emits
# ``plan_question`` and then idles until you answer:
# event: plan_question
# data: {"thread_id":"<uuid>","ts":1747000000000,"payload":{"questions":[...],"reasoning":"...","round":1}}

# After you POST /runs/<thread_id>/answer (see above), the Plan subgraph
# finalizes topics and the same stream emits ``plan_complete`` before closing:
# event: plan_complete
# data: {"thread_id":"<uuid>","ts":1747000000123,"payload":{"research_topics":[...]}}
```

**Python** (`httpx.Client.stream`):

```python
import json

import httpx

BASE = "http://localhost:8000"
with httpx.Client(base_url=BASE, timeout=None) as client:
    thread_id = client.post(
        "/runs", json={"user_query": "对比 Claude 4.5 与 GPT-4o 的推理深度"}
    ).json()["thread_id"]

    # ``stream()`` keeps the connection open; iterate lines as they arrive.
    with client.stream("GET", f"/runs/{thread_id}/events") as resp:
        resp.raise_for_status()
        event_name: str | None = None
        for line in resp.iter_lines():
            if not line:
                event_name = None  # blank line resets between events
                continue
            if line.startswith("event:"):
                event_name = line.removeprefix("event: ").strip()
                print(f"event: {event_name}")
            elif line.startswith("data:"):
                envelope = json.loads(line.removeprefix("data: "))
                # envelope == {"thread_id", "ts" (unix-ms), "payload"}
                print(f"  thread_id={envelope['thread_id']} ts={envelope['ts']}")
                print(f"  payload={envelope['payload']}")
```

### 5-round cap — `PLAN_MAX_ROUNDS`

The Q&A loop is bounded by `PLAN_MAX_ROUNDS`, defined at the top of
`src/app/graph/subgraphs/plan.py`:

```python
# src/app/graph/subgraphs/plan.py
PLAN_MAX_ROUNDS = 5
```

This is the **one-line tuning knob**: raise or lower the integer to cap
the number of question-emit rounds. After the cap is hit the questioner
LLM is skipped and the topic-finalize LLM runs directly (the cap is
**soft** — the questioner can also exit early by emitting
`ready_to_proceed=true` on any round).

## Tests

```bash
uv run pytest -q
```

Smoke suite finishes in under 30 seconds. All LLM and search calls are mocked — no external API traffic.

## Architecture

```
plan_decompose  →  research_run  →  write_compose  →  finalize
   (LLM, JSON)      (1 search/topic)  (outline+draft+citations)  (no-op)
```

State is persisted in Redis after every node. Threads are identified by UUIDv4.