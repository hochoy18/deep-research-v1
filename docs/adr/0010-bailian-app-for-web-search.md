# Bailian 单智能体应用承担 web search

v0.1 的 `web_search` 工具原本通过 DashScope OpenAI 兼容端点 + `extra_body={"enable_search": True}` 调用 LLM，让模型在生成过程中内联触发检索。Free-tier 账号下此路径在少量请求后即返回 403 `AllocationQuota.FreeTierOnly`，流水线无法走通；同时其输出依赖 LLM 自由发挥拼 JSON，结构不可控。改用阿里云百炼**单智能体应用**（`dashscope.Application.call(app_id, prompt)`）承担 web search。

## 决策

Web search 这一段调用从"裸 LLM + `enable_search=True`"切换为"`dashscope.Application.call()` 调一个百炼单智能体应用"。LLM 端（plan / outline / draft / review / polish）的 OpenAI 兼容调用保持不变（仍走 ADR-0002 的双供应商拆分）。

应用 ID 通过新 env var `DASHSCOPE_MCP_APP_ID` 注入到 `Config.dashscope_mcp_app_id`；调用方在 `app.tools.web_search._run_web_search` 里：

1. `dashscope.api_key = cfg.dashscope_api_key`
2. `dashscope.Application.call(app_id=cfg.dashscope_mcp_app_id, prompt=query)`
3. 解析两层 JSON：`output.text` 是 `{"result":{"content":[{"text":"<inner>"}]}}`，`inner` 是 `{"pages":[{snippet, hostname, hostlogo, title, url}, ...]}`
4. 取 `pages[0]` 映射成 `SearchResult`（`url` / `title` / `snippet` / `topic_id` / `score=None`）

## 备选方案

- **继续 `enable_search=True`（已拒）**：
  1. 共享 LLM free-tier quota：单智能体应用额度被其它 LLM 调用消耗后，web search 也 403。
  2. 输出非结构化：模型可能返回 markdown fence 包 JSON、字段缺失、`snippet` 写成自然语言句子；`_run_web_search` 里的 `json.loads` 防御代码覆盖不全。
  3. 无 citation 信息：模型在 `content` 里自带的 `[1]` / `(来源)` 都是它自己编的，无法溯源。

- **多智能体编排（Multi-agent collaboration，已拒）**：百炼支持"多智能体协同"，但 v0.1 只需要"per-topic 单次问答 → 一个 SearchResult"；多智能体把"plan → search → write"全包了，反而把 LLM 编排锁死到百炼，违背 ADR-0002 的"LLM 端独立"原则。v0.4 review 阶段如果发现单智能体的 thought chain 不够用，再考虑。

- **直接拿百炼 `TextEmbedding` / `Generation` 自己写 RAG（已拒）**：等于把"retrieve + rank + cite"重写一遍；单智能体应用已经把这层 server-side 编排做完了，再造一遍是浪费。

## 影响

- `app.config.Config` 新增 `dashscope_mcp_app_id: str` 字段；`get_config()` 读 `DASHSCOPE_MCP_APP_ID`，`override=False` 仍生效。
- `app.tools.web_search._run_web_search` 整段改写：去掉 `OpenAI` 客户端与 `extra_body`；改为 `dashscope.Application.call()` + 嵌套 JSON 解析。
- `app.tools.web_search` 删掉 `DEFAULT_SEARCH_MODEL = "qwen-plus"`——单智能体应用不暴露 model 选择。
- `pyproject.toml` 加 `dashscope>=1.27` 依赖（已加）。
- `tests/test_web_search_wrapper.py` 的 mock 目标从 `app.tools.llm.ChatOpenAI` 改为 `app.tools.web_search.dashscope.Application.call`。
- `research_backend/.env.example` 增加 `DASHSCOPE_MCP_APP_ID=...` 字段示例。
- `docs/adr/0002-two-provider-topology.md` 的 web search 段指向本 ADR，不再描述 `enable_search=True`。
- `src/examples/case_web_search.py`（已存在）现在能真打一发返回 5 个 pages，不再 403。
- v0.1 验收"全 smoke test 通过"不再受 LLM free-tier quota 限制。
