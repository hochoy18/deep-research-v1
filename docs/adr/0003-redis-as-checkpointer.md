# Redis 作为 LangGraph checkpointer

LangGraph 的 state（包括两处 HITL Interrupt 的载荷与所有在途流水线数据）通过 `langgraph-checkpoint-redis` 持久化到 Redis。选用 Redis，因为它本身也是调试时快速检视 state 的入口，同时 `langgraph-checkpoint-redis` 的集群与 TTL 支持便于未来扩展。

## 备选方案

- **Redis（已选）**：检视快、TTL 支持、便于未来集群扩展；要求 Redis 8.0+ 且带 RedisJSON + RediSearch 模块。
- **SQLite（`langgraph-checkpoint-sqlite`）**：零配置、单进程，本地开发更简单，但无并发写能力，也难以在进程外检视活跃 state。
- **PostgreSQL（`langgraph-checkpoint-postgres`）**：生产级并发写、工具成熟，但 v1 阶段引入数据库搭建成本。

## 影响

- 本地开发需 Redis 8.0+，并安装 RedisJSON + RediSearch 模块；包内的 `RedisSaver.setup()` 必须在启动时调用一次以创建 RediSearch 索引，否则写入会静默失败而不建索引。
- state 与 TTL 按 thread 划定，废弃的 thread 最终会过期。
- 后续切换到 SQLite 或 Postgres 仅需替换 checkpointer，无需改动 graph 代码。