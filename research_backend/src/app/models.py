"""Typed Pydantic models for the pipeline's data shapes.

These are the canonical types referenced by `state.py`, the node modules, and
the API responses. Naming follows the glossary in ``CONTEXT.md``.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ResearchTopic(BaseModel):
    """A single Topic carved out of the user_query."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(..., description="Stable identifier, e.g. 'topic-1'")
    title: str = Field(..., description="Short human-readable topic title")
    description: str = Field(..., description="What this topic is meant to investigate")
    expected_finding_type: str = Field(
        ..., description="What kind of evidence this topic should surface"
    )

    @field_validator("id")
    @classmethod
    def _id_not_blank(cls, value: str) -> str:
        """Reject empty or whitespace-only topic ids.

        Topic ids are the join key across the gate diff (``PlanEditDiff``),
        the SSE payload (``PlanConfirmInterruptPayload``), and the state
        audit trail. A blank id would silently merge two unrelated edits.
        """
        if not value or not value.strip():
            raise ValueError("ResearchTopic.id must be a non-empty, non-whitespace string")
        return value


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


# ─────────────────────────────────────────────────────────────────────────────
# Plan HITL data shapes (v0.2 — issue #11)
#
# The Plan subgraph asks structured questions in multiple rounds until the
# questioner signals ``ready_to_proceed``. Each round the questioner emits a
# ``PlanQuestionerBatch``; the user replies with a ``PlanAnswerBatch``; the
# API/SSE layer hands the user a ``PlanInterruptPayload`` describing what to
# fill in. All shapes are additive to v0.1 — ``plan_decompose`` still runs as
# before and no node currently emits these.
# ─────────────────────────────────────────────────────────────────────────────


class PlanQuestionOption(BaseModel):
    """One option for a single- or multi-choice plan question."""

    model_config = ConfigDict(frozen=True)

    option_id: str = Field(..., description="Stable option identifier; the user picks this")
    label: str = Field(..., description="Human-readable option label")


class SingleChoicePlanQuestion(BaseModel):
    """A question whose answer is exactly one ``option_id``."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["single_choice"] = "single_choice"
    id: str = Field(..., description="Stable question identifier, used to match answers")
    prompt: str = Field(..., description="The question text shown to the user")
    options: list[PlanQuestionOption] = Field(
        ..., description="Options the user can pick from (one will be chosen)"
    )


class MultiChoicePlanQuestion(BaseModel):
    """A question whose answer is a list of ``option_id``s."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["multi_choice"] = "multi_choice"
    id: str = Field(..., description="Stable question identifier, used to match answers")
    prompt: str = Field(..., description="The question text shown to the user")
    options: list[PlanQuestionOption] = Field(
        ..., description="Options the user can pick from (any subset may be chosen)"
    )


class FreeTextPlanQuestion(BaseModel):
    """A question whose answer is a free-form string."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["free_text"] = "free_text"
    id: str = Field(..., description="Stable question identifier, used to match answers")
    prompt: str = Field(..., description="The question text shown to the user")
    placeholder: str | None = Field(
        default=None, description="Optional placeholder text shown in the input box"
    )


class ConfirmationPlanQuestion(BaseModel):
    """A yes/no confirmation question."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["confirmation"] = "confirmation"
    id: str = Field(..., description="Stable question identifier, used to match answers")
    prompt: str = Field(..., description="The question text shown to the user")


# Discriminated union over the four kinds. Each member shares ``id``/``prompt``
# and a ``kind`` literal; the runtime type of ``options``/``placeholder`` and
# the accepted answer ``value`` narrows by ``kind`` (see ``PlanUserAnswer``).
PlanQuestion = Annotated[
    SingleChoicePlanQuestion
    | MultiChoicePlanQuestion
    | FreeTextPlanQuestion
    | ConfirmationPlanQuestion,
    Field(discriminator="kind"),
]


class PlanUserAnswer(BaseModel):
    """The user's reply to a single ``PlanQuestion``."""

    model_config = ConfigDict(frozen=True)

    question_id: str = Field(..., description="The ``PlanQuestion.id`` being answered")
    value: str | list[str] | bool = Field(
        ...,
        description=(
            "Answer value; the runtime type is narrowed by the matching question's "
            "``kind``: single_choice/free_text → str, multi_choice → list[str], "
            "confirmation → bool"
        ),
    )


class PlanQuestionerBatch(BaseModel):
    """One batch the questioner emits in a single round."""

    model_config = ConfigDict(frozen=True)

    questions: list[PlanQuestion] = Field(
        default_factory=list, description="The questions to ask in this round"
    )
    reasoning: str = Field(
        default="", description="Why the questioner is asking these questions right now"
    )
    ready_to_proceed: bool = Field(
        default=False,
        description="True once the questioner is satisfied and the plan can be built",
    )


class PlanAnswerBatch(BaseModel):
    """One round of user answers returned to the plan subgraph."""

    model_config = ConfigDict(frozen=True)

    round: int = Field(..., description="1-based round index this batch answers")
    answers: list[PlanUserAnswer] = Field(
        default_factory=list, description="The user's answers to this round's questions"
    )


class PlanInterruptPayload(BaseModel):
    """The API + SSE contract: questions waiting for the user's input."""

    model_config = ConfigDict(frozen=True)

    questions: list[PlanQuestion] = Field(..., description="Questions the user must answer")
    reasoning: str = Field(..., description="Why the questioner is asking these questions")
    round: int = Field(..., description="1-based round index this interrupt belongs to")


# ─────────────────────────────────────────────────────────────────────────────
# Plan → Research transition gate data shapes (v0.3 — issue #21)
#
# The transition gate is the dumb pass-through between Plan and Research:
# Plan emits ``research_topics``; the user reviews the list at the gate;
# the API receives the final list via ``ConfirmedResearchTopics`` and the
# diff vs the LLM draft via ``PlanEditDiff``. The API/SSE layer hands the
# user a ``PlanConfirmInterruptPayload`` describing the draft so the front
# end can render the table without an extra RTT. See ADR-0005.
#
# These are pure data contracts — no logic. The gate node is issue #23;
# the API surface is issue #24; the SSE event is issue #25; the gate's
# behavior tests are issue #26.
# ─────────────────────────────────────────────────────────────────────────────


class ConfirmedResearchTopics(BaseModel):
    """The body of ``POST /runs/{id}/confirm-plan`` — the user's final topic list.

    The front end always sends the complete final list (no diff protocol —
    see ADR-0005 §3). ``topics`` is required at the model level so a
    missing field fails closed at parse time (422). An explicit empty
    list (``"topics": []``) still parses — the API layer rejects it with
    code ``EMPTY_PLAN`` (issue #24).
    """

    model_config = ConfigDict(frozen=True)

    topics: list[ResearchTopic] = Field(
        ...,
        description="Final research topics the user confirmed (non-empty enforced at the API layer)",
    )


class ModifiedTopic(BaseModel):
    """One entry in ``PlanEditDiff.modified`` — same id, edited content.

    ``before`` and ``after`` carry full ``ResearchTopic`` snapshots so the
    audit trail can attribute any field-level drift to the user's edit
    without needing the LLM draft to be looked up elsewhere.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(..., description="Topic id of the entry that was edited")
    before: ResearchTopic = Field(
        ..., description="The ResearchTopic as the LLM drafted it"
    )
    after: ResearchTopic = Field(
        ..., description="The ResearchTopic after the user's edit"
    )


class PlanEditDiff(BaseModel):
    """The structured audit trail of the user's edits at the transition gate.

    Per ADR-0005 §7 the state records added / removed / modified topics so
    a "report drift" complaint can be attributed to either the LLM or the
    user's edits without re-running either side.
    """

    model_config = ConfigDict(frozen=True)

    added: list[ResearchTopic] = Field(
        default_factory=list,
        description="Topics the user added that weren't in the LLM draft",
    )
    removed: list[str] = Field(
        default_factory=list,
        description="Topic ids the user removed at the gate",
    )
    modified: list[ModifiedTopic] = Field(
        default_factory=list,
        description="Topics the user edited (id is shared with before/after)",
    )


class PlanConfirmInterruptPayload(BaseModel):
    """The API + SSE contract: the draft topic list waiting for confirmation.

    Inline on the SSE ``plan_confirm_required`` event so the front end can
    render the review table without an extra RTT (ADR-0005 §5). Distinct
    from ``PlanInterruptPayload``, which carries structured Q&A questions
    during the Plan HITL loop and is emitted on ``plan_interrupt``.
    """

    model_config = ConfigDict(frozen=True)

    topics: list[ResearchTopic] = Field(
        ..., description="The draft topics the user should review"
    )
    reasoning: str = Field(
        default="",
        description="Why the planner proposes these topics (shown above the table)",
    )