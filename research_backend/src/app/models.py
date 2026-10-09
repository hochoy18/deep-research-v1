"""Typed Pydantic models for the pipeline's data shapes.

These are the canonical types referenced by `state.py`, the node modules, and
the API responses. Naming follows the glossary in ``CONTEXT.md``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ResearchTopic(BaseModel):
    """A single Topic carved out of the user_query."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(..., description="Stable identifier, e.g. 'topic-1'")
    title: str = Field(..., description="Short human-readable topic title")
    description: str = Field(..., description="What this topic is meant to investigate")
    expected_finding_type: str = Field(
        ..., description="What kind of evidence this topic should surface"
    )


class SearchResult(BaseModel):
    """A single web result returned by the DashScope web-search tool."""

    model_config = ConfigDict(frozen=True)

    topic_id: str = Field(..., description="Owning ResearchTopic.id")
    url: str = Field(..., description="Canonical URL of the result")
    title: str = Field(..., description="Result title")
    snippet: str = Field(..., description="Short text excerpt")
    score: float | None = Field(default=None, description="Relevance score from the search service")


class OutlineSection(BaseModel):
    """One H2 chapter of the final report outline."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(..., description="H2 chapter title")
    summary: str = Field(..., description="One-sentence summary of the chapter")
    key_points: list[str] = Field(
        default_factory=list, description="Bulleted points the chapter should cover"
    )


class Outline(BaseModel):
    """The full report outline (a sequence of H2 sections)."""

    model_config = ConfigDict(frozen=True)

    sections: list[OutlineSection] = Field(
        default_factory=list, description="H2 chapters of the report, in order"
    )


class Reference(BaseModel):
    """A numbered reference appearing in the final report's References section."""

    model_config = ConfigDict(frozen=True)

    id: int = Field(..., description="1-based reference number, used as `[N]` in body")
    title: str = Field(..., description="Title of the source")
    url: str = Field(..., description="URL of the reference")


class SearchResultsByTopic(BaseModel):
    """Search results grouped by the ResearchTopic that produced them."""

    model_config = ConfigDict(frozen=True)

    results: dict[str, list[SearchResult]] = Field(
        default_factory=dict,
        description="Mapping from ResearchTopic.id to its list of SearchResult",
    )