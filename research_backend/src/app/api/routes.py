"""HTTP routes — health, run creation, run inspection."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.nodes.research_run import RECURSION_LIMIT

router = APIRouter()


class CreateRunRequest(BaseModel):
    user_query: str = Field(..., min_length=1)


class RunCreatedResponse(BaseModel):
    thread_id: str


def _build_config(thread_id: str) -> dict[str, Any]:
    """RunnableConfig used for both ``invoke`` and ``get_state`` calls."""
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
    }


def _to_jsonable(value: Any) -> Any:
    """Coerce a single value (Pydantic model / container / primitive) to JSON-friendly form."""
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    return value


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/runs", response_model=RunCreatedResponse)
async def create_run(payload: CreateRunRequest, request: Request) -> RunCreatedResponse:
    graph = request.app.state.graph
    if graph is None:  # pragma: no cover — only when lifespan hasn't run
        raise HTTPException(status_code=503, detail="Graph not initialized")

    thread_id = str(uuid.uuid4())
    graph.invoke({"user_query": payload.user_query}, config=_build_config(thread_id))
    return RunCreatedResponse(thread_id=thread_id)


@router.get("/runs/{thread_id}")
async def get_run(thread_id: str, request: Request) -> dict[str, Any]:
    graph = request.app.state.graph
    if graph is None:  # pragma: no cover
        raise HTTPException(status_code=503, detail="Graph not initialized")

    snapshot = graph.get_state(_build_config(thread_id))
    if snapshot is None or snapshot.values is None:
        raise HTTPException(status_code=404, detail="Thread not found")

    return _to_jsonable(snapshot.values)