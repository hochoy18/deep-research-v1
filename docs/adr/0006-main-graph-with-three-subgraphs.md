# 主图 + 三子图架构（Plan / Research / Write）

本流水线是一张主 `StateGraph`，其三个主节点分别是 `CompiledGraph` 子图：一个用于 Plan，一个用于 Research，一个用于 Write。每个子图声明一份 input / output `TypedDict` schema；主图在边界处把输出字段合并回全局 state。

## 备选方案

- **主图 + 子图（已选）**：阶段边界清晰，每个子图可独立测试与替换，阶段间的 schema 不一致会被类型化 IO 捕获。
- **单张扁平图 8 节点**：连线更简单，但阶段逻辑混杂，难以作为单元测试或替换。
- **节点间共享全局 state**：最简单，但任何节点都可能误读 / 误写属于另一阶段的字段。

## 影响

- 每个子图独占 `src/app/graph/subgraphs/` 下一个文件，并在 `src/app/state.py` 中持有各自的 IO schema。
- 新增阶段意味着编译一张新的子图并插入主图；既有阶段不受影响。
- 子图间通信是单向的（Plan → Research → Write）——除全局 state 外没有反向通道。