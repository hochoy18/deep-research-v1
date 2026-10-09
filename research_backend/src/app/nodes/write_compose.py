"""Write node — turn search_results into a Markdown Report (tickets #6 + #7).

Stage 1 (ticket #6) produces an outline and a draft Markdown body. Stage 2
(ticket #7) deterministically decorates the draft with ``[N]`` reference
markers and appends a ``## References`` section so every cited SearchResult
appears in the reference list at the bottom.

We split the node into two helpers so stage 1 can be exercised in isolation
without forcing every test to also build a citation inserter.
"""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.models import Outline, OutlineSection, Reference, SearchResult
from app.state import State
from app.tools import llm as llm_module
from app.utils import parse_llm_json

_OUTLINE_SYSTEM_PROMPT = (
    "You are a research report writer. You will receive a user question and a "
    "list of web search results grouped by topic. Respond ONLY with a JSON "
    "object matching the schema below — no prose, no Markdown fences.\n\n"
    "Schema:\n"
    '{"outline": [{"title": "<H2>", "summary": "<one sentence>", '
    '"key_points": ["...", "..."]}, ...],\n'
    ' "draft": "<Markdown body with one H2 section per outline entry, '
    'NO inline [N] citations, NO References section>"}'
)


class DraftResponse(BaseModel):
    """Typed schema the LLM must satisfy for the outline + draft."""

    outline: list[OutlineSection] = Field(..., description="1–5 outline sections")
    draft: str = Field(..., description="Markdown body without inline citations or References")


def _search_results_as_bullets(results: list[SearchResult]) -> str:
    return "\n".join(
        f"- topic_id={r.topic_id} url={r.url} title={r.title!r} snippet={r.snippet!r}"
        for r in results
    )


def _build_outline_prompt(state: State) -> list:
    user_query = state.get("user_query", "")
    search_results = state.get("search_results") or []
    topics = state.get("research_topics") or []
    topics_block = "\n".join(f"- {t.id}: {t.title} — {t.description}" for t in topics) or "(no topics)"
    results_block = _search_results_as_bullets(search_results) or "(no search results)"
    user_text = (
        f"User question: {user_query}\n\n"
        f"Topics to cover:\n{topics_block}\n\n"
        f"Search results:\n{results_block}\n\n"
        "Produce the outline + draft JSON."
    )
    return [
        SystemMessage(content=_OUTLINE_SYSTEM_PROMPT),
        HumanMessage(content=user_text),
    ]


def compose_outline_and_draft(state: State) -> dict:
    """Stage 1 — produce ``outline`` and ``draft`` from ``search_results``."""
    chat = llm_module.get_chat_model()
    response = chat.invoke(_build_outline_prompt(state))
    payload = parse_llm_json(response.content, "Write node")
    parsed = DraftResponse(**payload)

    outline_sections = parsed.outline
    if not outline_sections:
        raise RuntimeError(f"Write node: 'outline' must be a non-empty list: {parsed.outline!r}")

    draft = parsed.draft.strip()
    if not draft:
        raise RuntimeError(f"Write node: 'draft' must be a non-empty string: {parsed.draft!r}")
    if re.search(r"\[\d+\]", draft):
        # Stage 1 must not produce citations; reject so we don't double-stamp.
        raise RuntimeError("Write node: draft already contains [N] markers; stage 1 should not add them")

    return {
        "outline": Outline(sections=outline_sections),
        "draft": draft,
        "current_step": "write",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — deterministic citation + References insertion (ticket #7)
# ─────────────────────────────────────────────────────────────────────────────


def _build_reference_index(search_results: list[SearchResult]) -> dict[str, int]:
    """Map each search result's URL to its 1-based reference number, in order."""
    return {r.url: idx + 1 for idx, r in enumerate(search_results)}


def _render_references(refs: list[Reference]) -> str:
    return "\n".join(f"[{r.id}] {r.title} — {r.url}" for r in refs)


def _ensure_blank_line_before_references(body: str) -> str:
    body = body.rstrip()
    if not body.endswith("\n\n"):
        body += "\n\n"
    return body


_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


def _insert_reference_marker_per_sentence(section_body: str, marker: str) -> str:
    """Append ``[N]`` after every sentence in ``section_body``.

    Splits on sentence-ending punctuation (``.``, ``!``, ``?``) so each factual
    claim carries its own inline citation.
    """
    section_body = section_body.rstrip()
    if not section_body:
        return f" {marker}"
    sentences = _SENTENCE_END_RE.split(section_body)
    annotated = [f"{s.rstrip()} {marker}" for s in sentences if s.strip()]
    return " ".join(annotated)


def _assign_citation_marker(
    section_index: int,
    search_results: list[SearchResult],
    ref_index: dict[str, int],
    topics_index: dict[str, int],
) -> str | None:
    """Return the ``[N]`` marker for the Nth section's owning search result, or None."""
    topic_ids_in_order = sorted(topics_index, key=topics_index.get)
    if section_index >= len(topic_ids_in_order):
        return None
    topic_id = topic_ids_in_order[section_index]
    matching = [r for r in search_results if r.topic_id == topic_id]
    if not matching:
        return None
    primary = matching[0]
    return f"[{ref_index[primary.url]}]"


def _build_topics_index(topics: list) -> dict[str, int]:
    return {t.id: i for i, t in enumerate(topics)}


def compose_report(state: State) -> dict:
    """Stage 2 — decorate the draft with ``[N]`` markers and a References section."""
    draft = state.get("draft") or ""
    search_results = state.get("search_results") or []
    topics = state.get("research_topics") or []

    if not draft:
        raise RuntimeError("Write node stage 2: state has no draft")
    if not search_results:
        raise RuntimeError("Write node stage 2: state has no search_results")

    sections = re.split(r"(?m)^## ", draft)
    intro = sections[0].strip()
    bodies: list[tuple[str, str]] = []
    for chunk in sections[1:]:
        lines = chunk.split("\n", 1)
        heading = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""
        bodies.append((heading, body))

    ref_index = _build_reference_index(search_results)
    topics_index = _build_topics_index(topics)

    rendered: list[str] = []
    if intro:
        rendered.append(intro)
    for i, (heading, body) in enumerate(bodies):
        marker = _assign_citation_marker(i, search_results, ref_index, topics_index)
        if marker is not None:
            body = _insert_reference_marker_per_sentence(body, marker)
        rendered.append(f"## {heading}\n\n{body}".rstrip())

    body_with_references = _ensure_blank_line_before_references("\n\n".join(rendered))
    refs = [Reference(id=idx + 1, title=r.title, url=r.url) for idx, r in enumerate(search_results)]
    references_block = f"## References\n\n{_render_references(refs)}"

    report = f"{body_with_references}{references_block}\n"
    return {
        "report": report,
        "current_step": "final",
    }


def write_compose(state: State) -> dict:
    """Public node entrypoint — runs stage 1 then stage 2 in one go.

    Returns a state delta that includes BOTH stages' outputs so the graph state
    accumulates ``outline``, ``draft``, ``report``, and ``current_step``
    in one node transition.
    """
    stage1_delta = compose_outline_and_draft(state)
    stage2_delta = compose_report({**state, **stage1_delta})
    return {**stage1_delta, **stage2_delta}
