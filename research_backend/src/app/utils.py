"""Small utilities shared between nodes."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def strip_code_fences(raw: str) -> str:
    """Strip Markdown code fences (``` or ```json) from LLM output."""
    if raw.startswith("```"):
        raw = raw.strip("`").removeprefix("json").strip()
    return raw


def parse_llm_json(raw: str, node_name: str) -> dict[str, Any]:
    """Strip fences and parse the LLM's output as JSON. Raise on failure."""
    cleaned = strip_code_fences(raw).strip()
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{node_name}: LLM returned non-JSON: {raw!r}") from exc
    if not isinstance(result, dict):
        raise TypeError(
            f"{node_name}: LLM JSON must be an object, got {type(result).__name__}"
        )
    return result