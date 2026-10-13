# Plan→Research 转换门

Plan 阶段结束后，主题列表会再次展示给用户，用户可全增 / 改 / 删，确认后才进入 Research 阶段。该门是第二处 Interrupt，与 Plan 阶段内部的多轮 Q&A 互相独立。

## 备选方案

- **转换门（已选）**：用户审阅模型拆出的主题列表，并在任何搜索开始前纠正。在搜索成本仍为零时早期捕获错误拆解。
- **直接交接**：更简单——Plan 输出直接送 Research。更快，但拆错的主题列表会浪费一整轮搜索预算之后用户才看到。
- **Research 后置门**：门置于 Research 与 Write 之间，请用户确认检索证据后再撰写。有用，但拆错的主题列表仍会产生已浪费的检索。

## 影响

- 后端暴露的 `confirm-plan` 端点与 Plan 内的 `answer` 端点分离，前端处理两种不同的用户动作。
- Plan subgraph 持有 topic schema，是交接产物的单一可信来源。
- 后续要移除该门只需删掉一个 subgraph 节点，但 UX 流程必须相应重设。

## 操作层语义

v0.3 落地 transition gate 时钉死的 7 条操作层决策（v0.3 spec 直接引用）：

1. **位置**：`confirm_plan_gate` 是 Plan subgraph 内的节点，不是主图上的独立子图。Plan subgraph 是 topic schema 的唯一可信来源（见上"影响"）。
2. **Topic ID 语义**：稳定 UUIDv4。Plan 阶段首次拆出 topic 时分配 ID；用户在 gate 上做增 / 改 / 删时，原 topic 的 ID 保留，新增 topic 发新 ID。
3. **编辑 payload 形状**：`POST /runs/{id}/confirm-plan` 的 body 永远是 `{"topics": [...完整最终列表...]}`，不论用户改没改。不引入 diff 协议，前端总是发送最终态。
4. **空 topic 列表兜底**：双重防御——客户端禁用"确认"按钮（`topics.length < 1` 时），服务端 400 + code `EMPTY_PLAN`。不引入"重回 plan_questioner"的 re-loop 路径。
5. **`plan_confirm_required` SSE payload**：内联——event data 直接带 `{topics: [...], reasoning: "..."}`，前端无需额外 RTT 即可渲染表格。
6. **State 双字段**：`research_plan`（LLM 草稿）始终保留；`confirmed_research_plan`（用户最终版）由 confirm-plan 触发后写入。`GET /runs/{id}` 同时返回两者便于事后定位"报告跑偏"是 LLM 的责任还是用户的改动。
7. **审计轨迹**：state 增加 `plan_edit_diff: {added: [...], removed: [...], modified: [{id, before, after}]}`，结构化记录增删改。与第 6 条的双字段配合，事后可完整反查。