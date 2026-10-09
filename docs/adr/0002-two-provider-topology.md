# 双供应商拓扑（OpenAI 负责 LLM，百炼单智能体应用负责网络搜索）

LLM（用于 plan 提问、检索结果评估、outline / draft / review / polish）使用 OpenAI GPT-4.x，调用方式 `langchain-openai`。网络搜索改走阿里云百炼的**单智能体应用**（详见 [ADR-0010](./0010-bailian-app-for-web-search.md)）——通过 `dashscope.Application.call(app_id, prompt)` 调一个预先配置好 web search 插件的应用，解析其结构化 `pages[]` 输出。两个职责用不同 SDK：`langchain-openai` 走 LLM 端，`dashscope` SDK 走 search 端。

## 备选方案

- **两个客户端（已选）**：GPT 负责 plan / eval / write 的推理质量，百炼单智能体应用负责结构化的 web search（带独立 quota、citation 字段）。
- **全部走 DashScope**：LLM 切到 Qwen，单一 SDK。节省开销且天然获得中文检索能力，但牺牲了 Review 与 Polish 阶段 GPT 的推理质量。
- **LiteLLM 单一接口**：按 model 字符串路由来抽象供应商。最灵活，但为 v1 仅两个供应商的场景引入一层间接与一份依赖；且 web search 走单智能体应用后 LLM 端只有一家，LiteLLM 抽象意义变小。
- **裸 LLM + `extra_body={"enable_search": True}`**（已被 ADR-0010 取代）：quota 共享易撞限、输出非结构化、无 citation 溯源。

## 影响

- `.env` 中需同时配置两家供应商的 API key，外加 `DASHSCOPE_MCP_APP_ID`（百炼单智能体应用 ID）。
- 替换 LLM 供应商只需改 `langchain-openai` 配置；替换 web search 供应商只需在百炼控制台改单智能体应用的插件编排，代码侧无改动。
- plan / eval / write 的推理质量受 GPT 制约；检索质量受单智能体应用的插件配置制约。