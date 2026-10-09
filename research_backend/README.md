# research_backend — Deep Research Agent Backend (v0.1)

End-to-end "question → cited report" pipeline (single-turn happy path, no HITL, no transition gate, no review / polish). Built on LangGraph + FastAPI.

## Stack

- **LangGraph** — stateful multi-stage graph (ADR-0001)
- **OpenAI GPT-4.x** — LLM for plan / outline / draft / citation (via `langchain-openai`)
- **阿里云百炼 DashScope** — web search via OpenAI-compatible chat completion with `extra_body={"enable_search": True}` (ADR-0002)
- **Redis 8.0+** (with RedisJSON + RediSearch) — LangGraph checkpointer (ADR-0003)
- **FastAPI** — `POST /runs`, `GET /runs/{thread_id}`, `GET /health`
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