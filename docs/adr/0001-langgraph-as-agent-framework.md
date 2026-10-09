# LangGraph 作为 agent 框架

本流水线是一条有状态的多阶段图，包含两处人机协作暂停和持久化的 checkpoint。选用 LangGraph，因为它原生提供 `interrupt()` 实现 HITL 暂停、原生 checkpointer（含 `langgraph-checkpoint-redis`），并提供类型化的 `StateGraph`，与三个流水线阶段吻合得很自然。

## 备选方案

- **LangGraph（已选）**：通过 `interrupt()` 原生支持 HITL，通过 checkpointer 原生支持持久化，使用类型化的 state schema。
- **LangChain 不带 graph 模块**：更轻量，但 HITL 暂停与持久化都需要手写。
- **自定义 asyncio + PydanticAI**：控制力最强，但需要从零重建 checkpoint。
- **CrewAI / AutoGen**：多 agent 协作场景下表现好，但在严格线性流水线场景下 HITL 语义与持久化都较弱。

## 影响

- 流水线形态被编码在 graph 定义中，可在 LangGraph Studio 中调试。
- 后续切换框架意味着重写整张状态机。