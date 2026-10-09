# Plan 阶段的多轮结构化 HITL

Plan 阶段是唯一允许多轮人机协作交互的阶段，所提问题都是类型化的（单选、多选、填空、确认）。Plan 节点产出 `{questions, reasoning, ready_to_proceed}`，通过 `interrupt()` 暂停直到用户提交答案；循环直到 `ready_to_proceed=true`（上限 5 轮），之后产出 research plan。

## 备选方案

- **多轮结构化 Q&A（已选）**：具备适应性——当答案含糊时模型可以追问；类型化 schema 让前端可以原生渲染（选项按钮、文本输入、确认按钮）。
- **一次性 N 个前置问题**：协议更简单，但首轮答案不清晰时失去适应性。
- **自由对话往返**：最灵活，但难以形式化成前端能确定性渲染的类型化 schema。
- **v1 仅 mock**：推迟真正的 HITL UX，等同于没有 Plan 阶段。

## 影响

- Interrupt 载荷是类型化 JSON 对象，前端渲染、用户填写、后端恢复 graph。
- 5 轮上限是软护栏而非硬契约——若问题本身已经清晰，模型可在 1 轮后即 `ready_to_proceed=true`。