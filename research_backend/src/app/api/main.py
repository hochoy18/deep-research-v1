"""FastAPI application (ticket #8).

The lifespan handler opens the Redis checkpointer, runs ``setup()`` once, and
compiles the graph onto ``app.state.graph``. ``RedisSaver.setup()`` creates the
RediSearch indexes that the saver requires on first startup.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router as runs_router
from app.config import get_config
from app.graph import build_graph

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_config()
    # Import lazily so tests can swap the checkpointer without a Redis dependency.
    from langgraph.checkpoint.redis import RedisSaver

    saver = RedisSaver.from_conn_string(cfg.redis_url)
    saver.setup()
    app.state.saver = saver
    app.state.graph = build_graph(saver)
    log.info("Application started; graph wired to %s", cfg.redis_url)
    try:
        yield
    finally:
        # RedisSaver doesn't expose a close() — let the OS reclaim sockets on shutdown.
        app.state.graph = None
        app.state.saver = None


app = FastAPI(title="Deep Research Backend", version="0.1.0", lifespan=lifespan)
app.include_router(runs_router)