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

## Plan HITL + SSE (v0.2)

The Plan stage interviews the user before any web search runs. `POST /runs`
synchronously drives the graph to its first `interrupt()` and returns; the
graph stays paused until answers are submitted:

```bash
# 1. Start a run — returns thread_id; the graph is paused at round 1.
curl -X POST http://localhost:8000/runs \
  -H "content-type: application/json" \
  -d '{"user_query": "对比 Claude 4.5 与 GPT-4o 的推理深度"}'
# {"thread_id":"<uuid>"}

# 2. Tail the Plan lifecycle as Server-Sent Events.
curl -N http://localhost:8000/runs/<thread_id>/events
# event: plan_question
# data: {"thread_id":"<uuid>","ts":1747000000000,"payload":{"questions":[...],"reasoning":"...","round":1}}

# 3. Answer the round's questions in one batch — the graph resumes.
curl -X POST http://localhost:8000/runs/<thread_id>/answer \
  -H "content-type: application/json" \
  -d '{"answers": [{"question_id": "q-scope", "value": "tech"}]}'
# {"thread_id":"<uuid>","accepted":true}
```

- **Events**: v0.2 ships two types — `plan_question` (payload = `PlanInterruptPayload.model_dump()`: `{questions, reasoning, round}`) and `plan_complete` (payload = `{research_topics: [...]}`, empty on failure). Every frame uses the v0.5 forward-compatible envelope `{thread_id, ts, payload}` (`ts` = unix-ms); the remaining six event types land in v0.5.
- **Stream close conditions**: main graph reaches END, the graph fails, the client disconnects, or the thread is unknown. A late joiner whose stage already advanced sees a replayed `plan_complete` before the close.
- **Failure semantics**: the graph fast-fails by raising (ADR-0008); `POST /runs` and `POST /runs/{id}/answer` persist `current_step="failed"` before propagating the 500, so an open stream observes the failure, emits the empty `plan_complete`, and closes. A dedicated `error` event lands in v0.5.
- **Round cap**: Q&A stops at `PLAN_MAX_ROUNDS = 5` (`src/app/graph/subgraphs/plan.py`) — after the cap the topic list is finalized even if the questioner never signals `ready_to_proceed`. The LLM can exit early the same way; the cap is soft.

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