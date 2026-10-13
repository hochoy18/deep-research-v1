import asyncio

from langgraph.checkpoint.redis import RedisSaver

from app.api.events import stream_plan_events
from app.config import _getenv
from app.graph import build_graph

redis_url = _getenv("REDIS_URL", "redis://localhost:6379")

# ``from_conn_string`` is a context manager in the installed
# langgraph-checkpoint-redis, so build the saver directly instead.
saver = RedisSaver(redis_url=redis_url)
saver.setup()
graph = build_graph(saver)
thread_id = '10000001'
config =  {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 10,
    }

USER_QUERY = "深度调研 欧美国家 酒店旅游业 AI Agent 的落地方案"

graph.invoke({"user_query": USER_QUERY}, config=config)


class MockRequest:
    """Stand-in for fastapi.Request — stream_plan_events only calls is_disconnected()."""

    async def is_disconnected(self) -> bool:
        return False

request = MockRequest()

async def main():
    async for event in stream_plan_events(request, graph, config, thread_id):
        print(event.decode("utf-8"), end="")

asyncio.run(main())