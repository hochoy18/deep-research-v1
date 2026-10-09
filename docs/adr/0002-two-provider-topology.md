# 双供应商拓扑（OpenAI 负责 LLM，DashScope 负责网络搜索）

LLM（用于 plan 提问、检索结果评估、outline / draft / review / polish）使用 OpenAI GPT-4.x，调用方式 `langchain-openai`。网络搜索使用阿里云百炼 DashScope，通过其 OpenAI 兼容端点 + `extra_body={"enable_search": True}` 参数开启。两个客户端共用 `openai` Python SDK，但 base URL 不同。

## 备选方案

- **两个客户端（已选）**：GPT 负责 plan / eval / write 的推理质量，DashScope 负责中文友好且成本友好的检索。
- **全部走 DashScope**：LLM 切到 Qwen，单一客户端。节省开销且天然获得中文检索能力，但牺牲了 Review 与 Polish 阶段 GPT 的推理质量。
- **LiteLLM 单一接口**：按 model 字符串路由来抽象供应商。最灵活，但为 v1 仅两个供应商的场景引入一层间接与一份依赖。

## 影响

- `.env` 中需同时配置两家供应商的 API key。
- 替换 LLM 供应商只需改 `langchain-openai` 配置；替换搜索供应商只需改搜索工具的客户端。
- plan / eval / write 的推理质量受 GPT 制约；检索质量受 DashScope 索引制约。