# Roadmap — Deep Research Agent Backend

本文档把 8 个 ADR 拆成 5 个可交付的版本。每个版本都是可演示的里程碑，前一版本是后一版本的运行前提。配套文档：

- 术语表：`./CONTEXT.md`
- 决策记录：`./adr/0001` ~ `0008`

---

## 分阶段原则

1. **依赖**：后一版本必须能站在前一版本的成果上跑起来，不留半成品
2. **价值**：每个版本都交付一个**可演示的里程碑**，而不是"做完一半再做下一半"
3. **从简到繁**：先单轮 happy path，再加 HITL，再加质量门，最后做运营加固

---

## 版本总览

| 版本 | 主题 | 落地的 ADR | 阶段交付（用户能看到的） |
|---|---|---|---|
| **v0.1 Foundation** | 项目骨架 + 单轮 happy path | 0001、0002、0003、0008（仅 raise） | `POST /runs` 后拿到 final_report，问题→报告全跑通 |
| **v0.2 Plan HITL** | Plan 阶段变成可对话 | 0004、0006（拆子图） | 用户能与 Plan 节点多轮 Q&A，最终拿到 research_plan |
| **v0.3 Transition gate** | Plan→Research 转换门 | 0005 | 用户在转换门全增删改 topic 后再检索 |
| **v0.4 Write quality** | 三维评审 + 润色 | 0007 | Draft 不通过就回炉，三维全过才出 final |
| **v0.5 Production** | 错误字段、SSE、观测性 | 0008（state 错误字段） | 节点失败写 state、SSE 8 类事件、结构化日志带 trace_id |

---

## v0.1 — Foundation

**目标**：端到端跑通"问题 → 报告"单轮流程，无 HITL、无 transition gate、无 review / polish。

**落地的 ADR**：0001（LangGraph）、0002（双供应商）、0003（Redis checkpointer）、0008（仅 raise，不重试）

### 工程交付清单

**项目骨架**
- `research_backend/pyproject.toml`：Python 3.12 + 宽松依赖，列出 `fastapi`、`uvicorn[standard]`、`langgraph`、`langgraph-checkpoint-redis`、`langchain`、`langchain-openai`、`openai`、`redis`、`pydantic`、`python-json-logger`、`pytest`、`httpx`、dev：`pytest-asyncio`、`ruff`
- `research_backend/.env.example`：OpenAI API key、DashScope API key、Redis URL
- `research_backend/README.md`：本地启动步骤（安装 Redis 8.0+、`uv sync`、`uv run uvicorn app.main:app --reload`）

**代码（`research_backend/src/app/`）**
- `config.py`：`os.getenv` 直读 OpenAI / DashScope / Redis 配置
- `models.py`：Pydantic 类型 — `PlanQuestion`、`ResearchTopic`、`SearchResult`、`Outline`、`ReviewNote`、`Reference`
- `state.py`：全局 TypedDict state（`user_query`、`search_results`、`draft`、`final_report`、`current_step` 等）
- `tools/llm.py`：`langchain-openai` ChatOpenAI 实例
- `tools/web_search.py`：DynamicStructuredTool 包装 DashScope `enable_search`
- `graph.py`：单张 `StateGraph`，节点 `plan_decompose → research_run → write_compose → finalize`
- `nodes/plan_decompose.py`：LLM 把 user_query 拆成 topic 列表（无 Q&A）
- `nodes/research_run.py`：每 topic 一次 search，recursion_limit 兜底
- `nodes/write_compose.py`：Outline → Draft → final_report（行内 `[N]` + 文末 References），不做 review / polish
- `api/main.py`：FastAPI app
- `api/routes.py`：`POST /runs`（创建 thread）、`GET /runs/{id}`（拿 state）、`GET /health`

**测试**
- `tests/conftest.py`：`fake_llm` 与 `fake_search` fixture（`monkeypatch` 替换 OpenAI client / DynamicStructuredTool.invoke）
- `tests/test_plan_decompose.py`：smoke test，输入 mock user_query，验证产出 topic 列表结构
- `tests/test_research_run.py`：smoke test，输入 mock topics，验证产出 search_results
- `tests/test_write_compose.py`：smoke test，输入 mock search_results，验证产出 final_report 含 References
- `tests/test_api_health.py`：FastAPI TestClient 验证 `/health`

**基础设施**
- Redis 8.0+ 本地安装指引（README 中）
- `.env.example` 提供完整 key 字段清单

### 验收

- `uv run uvicorn app.main:app --reload` 起得来
- `POST /runs {"user_query": "..."}` 返回 `thread_id`
- `GET /runs/{thread_id}` 最终 state 包含 `final_report` 字段，是带 `[N]` + References 的 Markdown
- 全 smoke test 通过

---

## v0.2 — Plan HITL

**目标**：Plan 阶段从"一次性拆 topic"升级为"多轮结构化问答 + 自适应拆解"。

**落地的 ADR**：0004（多轮结构化 HITL）、0006（主图 + 三子图架构）

### 工程交付清单

**架构迁移**
- 把 v0.1 的单图拆成主图 + 3 个子图：Plan subgraph、Research subgraph、Write subgraph
- 每个子图声明 input / output schema（`src/app/state.py` 新增 `PlanState` / `PlanOutput` 等 TypedDict）
- 主图负责合并子图输出到全局 state

**Plan subgraph**
- 节点：`plan_questioner`（LLM 产出 `{questions, reasoning, ready_to_proceed}`）→ `interrupt()` → 用户回答 → 循环直到 `ready_to_proceed=true`（上限 5 轮）→ 产出 `research_plan`
- 实现：`langgraph.graph.StateGraph` + `from langgraph.types import interrupt, Command`

**API**
- `POST /runs/{id}/answer`：提交 Plan Q&A 答案，触发 graph 恢复

**SSE**
- 加 `GET /runs/{id}/events` 端点
- 落地两类事件：`plan_question`（新问题轮次）、`plan_complete`（Plan HITL 结束）

**测试**
- `tests/test_plan_subgraph.py`：mock LLM，验证多轮 interrupt + 收答案 → 产出 `ready_to_proceed=true`
- `tests/test_api_answer.py`：FastAPI TestClient 验证 `POST /runs/{id}/answer` 恢复 graph

### 验收

- 用户多轮 Q&A 后能拿到 `research_plan`
- Interrupt 暂停期间 `Redis` 中能看到 thread state
- 5 轮上限触发后，graph 强制进入下一步

---

## v0.3 — Transition gate

**目标**：Plan 阶段结束后，在 Research 启动前插入用户审阅 / 编辑 topic 列表的门。

**落地的 ADR**：0005

### 工程交付清单

**Plan subgraph 增加节点**
- 新增 `confirm_plan_gate` 子节点：emit 当前 `research_plan` → `interrupt()` → 接收增删改后的 topic 列表
- Plan subgraph 流程变为：`plan_questioner`（多轮 Q&A）→ `confirm_plan_gate`（转换门）→ 输出最终 `research_plan`

**API**
- `POST /runs/{id}/confirm-plan`：接收编辑后的 topic 列表，触发 graph 恢复

**SSE**
- 新增事件类型：`plan_confirm_required`

**测试**
- `tests/test_transition_gate.py`：mock 输入 research_plan，验证用户全增删改后正确流入 Research
- `tests/test_api_confirm_plan.py`：FastAPI TestClient 验证 `POST /runs/{id}/confirm-plan`

### 验收

- Plan 完成后用户在转换门可以增、删、改 topic
- 用户不改也能点"确认"，原样流入 Research
- SSE 推 `plan_confirm_required` 后 thread state 在 Redis 中可检视

---

## v0.4 — Write quality

**目标**：Write 阶段从"一次性出报告"升级为"outline → draft → review → 修订 → polish"。

**落地的 ADR**：0007

### 工程交付清单

**Write subgraph 拆节点**
- `write_outliner`：从 search_results 产出 Outline（H2 + key_points）
- `write_drafter`：从 Outline 产出 Draft
- `write_reviewer`：LLM judge 沿三维（事实准确性 / 结构清晰度 / 引用完整性）评分；不通过则回 drafter
- `write_polisher`：仅做三件事——过渡 / 语气 / 错字，不动事实 / 结构 / 引用

**数据模型**
- `ReviewNote` Pydantic 类型：包含三维评分 + 修订建议
- `state` 字段增加 `outline`、`draft`、`review_notes`、`polish_iterations`

**测试**
- `tests/test_write_reviewer.py`：mock Draft，验证三维评分逻辑 + 不达阈触发回炉
- `tests/test_write_polisher.py`：验证 Polish 节点不改变事实（输入 Draft 的事实陈述集合 = 输出 final_report 的事实陈述集合）

### 验收

- Review 三维中任一维度不通过，Drafter 自动修订
- 全维度通过后才进入 Polish
- Polish 后 final_report 仍满足三维 Review（即 Polish 不退化）

---

## v0.5 — Production hardening

**目标**：错误处理、SSE 全事件、观测性到位，能在生产环境里查问题。

**落地的 ADR**：0008（state 错误字段）

### 工程交付清单

**State 错误字段**
- `state` 字段增加 `last_error: str | None`、`failed_step: Literal[...] | None`、`trace_id: str`
- 节点错误捕获后写状态，graph 转入 `failed` 终态
- API 层根据 state 返回 4xx / 5xx

**SSE 全事件落地**
- 8 类事件全部实现：`state_update`、`plan_question`、`plan_complete`、`plan_confirm_required`、`search_progress`、`outline_ready`、`draft_chunk`、`review_notes`、`final_report`、`error`
- 每条事件带 `event: <name>` 与 `data: {thread_id, ts, payload}`

**结构化日志**
- stdlib `logging` + JSON formatter（`python-json-logger`）
- 每条日志带 `trace_id`、`thread_id`、`node`、`latency_ms`
- LLM / search 调用的入参 / 出参 / 耗时都打日志

**测试**
- `tests/test_error_handling.py`：mock LLM 抛错，验证 state 写入 `last_error` + `failed_step`
- `tests/test_sse_events.py`：FastAPI TestClient 验证 SSE 流能收到 8 类事件

### 验收

- LLM / search 任何调用失败，state 立即标记 `last_error` + `failed_step`，API 返回明确错误
- 一次完整 pipeline 跑下来，SSE 流能看到全部 8 类事件
- 日志输出 JSON，每条带 `trace_id`，可在日志系统中按 thread_id 反查整条 pipeline

---

## 故意推迟到 v1 之外

以下内容明确不进入 v0.1–v0.5，留到 v1 之后再决策：

- **检索 query 缓存、写阶段缓存**：先跑通后优化
- **多供应商路由 / 多模型 A/B**：等 v0.5 观察成本后再决定是否值得
- **token / 成本预算**：v0.5 加基础指标，v1 再做硬上限
- **鉴权 / 限流**：v1 范围之外
- **Plan / Research / Write 之外的第 4 阶段**：当前 3 阶段足够覆盖需求
- **用户对话历史、个性化**：不在 v1 范围

---

## 下一步：按主流程走

按 `/ask-matt` 的主流程对位，每个版本走一遍：

```
/to-spec（拍本版本 spec）
   ↓
/to-tickets（拆成 5–8 张 tracer-bullet ticket，每张标 blocking edges）
   ↓
/implement × N（每张 ticket 一个新上下文窗口，按 blocking 顺序做）
   ↓
/clear（每张 ticket 完工后清窗口，下一张重新加载上下文）
```

**关键约束**：每个版本内，`grilling → spec → tickets` 必须在**同一上下文**里跑完，不中途 `/clear` 或 `/compact`。每个 `/implement` 才换新窗口，因为 ticket 是自包含的。

**建议起步**：先 `/to-spec` 拍 v0.1 Foundation。spec 里直接引用本 roadmap 的"v0.1 工程交付清单"作为骨架，再细化每个 ticket。