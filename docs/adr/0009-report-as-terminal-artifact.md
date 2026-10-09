# Report 作为终态交付产物

v0.1 / v0.4 流水线在 Write 阶段终态交付的 Markdown 产物，被定义为一个独立的术语 `Report`，与未润色的 `Draft` 区分。

## 决策

引入 `Report` 作为新的术语（见 `CONTEXT.md`）。Pipeline run 终态交付物是 Report（带 `[N]` 引文 + 文末 References）。v0.4 多轮写流程里只有一类中间产物：`Draft`（Review 反馈驱动的多轮未润色散文），无中间"已润色"层——Polish 是 Write 子图的最后一步，其输出即为 Report。一次 run 只产生一个 Report。

State 字段对齐（两轮 PR 走完）：wave-1 `final_report` → `final_draft`（与 `Draft` 术语冲突随后被发现）；wave-1.5 `final_draft` → `report`。roadmap v0.1 / v0.4 验收里所有 `final_report` 字面同步改成 `report`。

## 备选方案

- **继续把终态产物当 `Draft` 用（已拒）**：`Draft` 术语定义里写明"未经过评审或润色"，与终态产物的语义不符。把成品叫 `Draft` 会让术语表自己矛盾，且 v0.4 多轮迭代时无法区分哪一轮是终态。
- **`Output` / `Summary`（已拒）**：太通用，丧失区分力。流水线任意节点都产出 output；哪个 output 算"the Report" 在 v0.4 多轮下会模糊。
- **`Final Report`（已拒）**："Final" 前缀暗示单数唯一性。v0.4 多轮 Polish 时谁算 "Final" 是定义难题。改成纯 `Report`，唯一性由"一次 run 唯一"承担，更干净。
- **`Polished Draft`（已拒）**：早期考虑过把"已 Polish 但未 Review"叫 `Polished Draft`，与 v0.4 的 Review→Polish→Report 顺序不匹配——Polish 是 Write 子图最后一步，输出即 Report，没有 Polished-but-not-yet-Report 的中间层。术语表无需为此造一个空概念。

## 影响

- `CONTEXT.md` 写作产物小节新增 `Report` 术语；`Draft._Avoid_` 保留 `report`（仍要避免把未润色散文叫 Report）。
- state 字段 `final_report` / `final_draft` → `report`（两轮 PR：wave-1 走 `final_report → final_draft`，wave-1.5 走 `final_draft → report`）；函数 `compose_final_report` → `compose_report`；测试里 `delta["final_report"]` 等访问点同步改成 `delta["report"]`。
- `docs/roadmap.md` v0.1 验收（line 71）与 v0.4 验收（lines 168-170, 162）里 `final_report` 字面一并改成 `report`，保持代码、术语、文档三者一致。
