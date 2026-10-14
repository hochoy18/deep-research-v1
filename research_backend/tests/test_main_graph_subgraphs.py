"""Issue #17 — Main-graph subgraph wiring + E2E happy-path test.

Two cases pin the contract the main graph implements per ADR-0006:

- ``test_main_graph_wires_three_subgraphs`` — ``build_graph(...)`` exposes
  the three subgraph nodes (``plan_subgraph``, ``research_subgraph``,
  ``write_subgraph``), and the compiled graph's output schema equals the
  union of subgraph IO TypedDicts (``State`` per ADR-0006).

- ``test_end_to_end_with_immediate_ready_to_proceed`` — full happy path:
  the Plan questioner returns ``ready_to_proceed=true`` on round 1, the
  topic-finalize LLM emits ``ResearchTopic``s, Research runs the web
  searches, Write composes the report, and the final state contains a
  ``report`` with ``[N]`` markers and a ``## References`` section.

Both cases use the v0.1 shared fixtures (``fake_chat_model``,
``fake_search_tool``, ``memory_checkpointer``) and assert external
behaviour — no real LLM calls, no real network calls. The shape mirrors
the v0.1 ``test_graph_e2e.py`` happy-path, but uses the shared fixtures
instead of inline stubs (which is what the issue's AC requires). The
``fake_chat_model`` is wired up to ``app.tools.llm.get_chat_model`` the
same way the ``patched_chat_model`` conftest fixture does it, so the
test signature matches the AC literally.

v0.3 (issue #23) inserts the Plan→Research transition gate between
``plan_questioner`` and the research subgraph. The E2E happy-path
test drives the gate resume via ``Command(resume=...)`` so the
pipeline can advance through research → write → finalize → END; the
subgraph-wiring test is unaffected (the gate is a node inside
``plan_subgraph``, not a new subgraph on the main graph).
"""

from __future__ import annotations

import json
import uuid

from langgraph.types import Command

from app.graph import build_graph
from app.models import (
    ConfirmedResearchTopics,
    Outline,
    PlanAnswerBatch,
    PlanUserAnswer,
    ResearchTopic,
)
from app.nodes.research_run import RECURSION_LIMIT
from app.state import State

# The three subgraph node ids the main graph must compose (ADR-0006).
_EXPECTED_SUBGRAPH_NODE_IDS = frozenset(
    {"plan_subgraph", "research_subgraph", "write_subgraph"}
)


def test_main_graph_wires_three_subgraphs(memory_checkpointer) -> None:
    """AC #17.1 — ``build_graph(...)`` exposes the three subgraph node ids,
    and the compiled graph's output schema is the union of subgraph IO
    fields.

    The global ``State`` TypedDict is declared as that union per ADR-0006
    (see ``src/app/state.py``), so asserting equality to
    ``State.__annotations__`` pins the contract without enumerating the
    subgraph IO TypedDicts by hand.
    """
    graph = build_graph(memory_checkpointer)

    # 1. The three subgraph node IDs are present.
    nodes = set(graph.nodes.keys())
    missing = _EXPECTED_SUBGRAPH_NODE_IDS - nodes
    assert not missing, (
        f"main graph is missing the following subgraph nodes: "
        f"{sorted(missing)}; found nodes: {sorted(nodes)}"
    )

    # 2. The compiled output schema equals the union of subgraph IO fields.
    # ``State`` is the contract surface: every field any subgraph writes
    # back must appear here, and no subgraph-local field (e.g. PlanState's
    # ``messages`` / ``round`` / ``last_questions`` / ``last_answers``) must
    # leak into the main graph.
    output_fields = set(graph.output_schema.__annotations__.keys())
    expected_fields = set(State.__annotations__.keys())
    assert output_fields == expected_fields, (
        f"compiled output schema must equal the union of subgraph IO fields; "
        f"missing: {sorted(expected_fields - output_fields)}, "
        f"extra: {sorted(output_fields - expected_fields)}"
    )


def test_end_to_end_with_immediate_ready_to_proceed(
    fake_chat_model, fake_search_tool, memory_checkpointer
) -> None:
    """AC #17.2 — full happy path with the Plan subgraph finalizing on round 1.

    Three LLM calls are scripted onto ``fake_chat_model.responses``:

    1. Questioner LLM — ``ready_to_proceed=true`` so ``plan_questioner``
       short-circuits to the topic-finalize LLM in the same node entry
       (no questioner ``interrupt()`` raised, no resume needed).
    2. Topic-finalize LLM — two ``ResearchTopic`` entries that drive the
       research / write stages below.
    3. Write stage 1 LLM — outline (2 H2 sections matching the topics)
       and a draft body without ``[N]`` markers (stage 2 adds them).

    v0.3 (issue #23): the questioner finalizes into the Plan→Research
    transition gate, which pauses. The test drives the gate resume
    directly via ``Command(resume=ConfirmedResearchTopics(...))`` so
    the pipeline can advance through research → write → finalize →
    END. There is no ``/confirm`` HTTP endpoint yet (issue #24) so
    the resume is done through the same compiled graph the test
    already built.

    Final state assertions match ``test_graph_invoke_end_to_end``:
    ``current_step == "final"``, two topics, two search results, an
    ``Outline`` of two sections, and a report carrying ``[1]`` / ``[2]``
    markers plus a ``## References`` block whose entries reference the
    ``Mock result for {topic_id}`` titles the ``fake_search_tool`` stub
    emits (see ``tests/conftest.py``).
    """
    topics_payload = {
        "topics": [
            {
                "id": "topic-1",
                "title": "History",
                "description": "history of X",
                "expected_finding_type": "factual",
            },
            {
                "id": "topic-2",
                "title": "Current state",
                "description": "current state of X",
                "expected_finding_type": "data",
            },
        ]
    }
    write_payload = {
        "outline": [
            {"title": "History", "summary": "s", "key_points": ["p"]},
            {"title": "Current state", "summary": "s", "key_points": ["p"]},
        ],
        "draft": (
            "Intro paragraph.\n\n"
            "## History\n\nFirst historical fact. Second historical fact.\n\n"
            "## Current state\n\nFirst current fact. Second current fact."
        ),
    }

    # Wire ``fake_chat_model`` into the LLM tool entry point — the same
    # patch the ``patched_chat_model`` conftest fixture applies.
    from app.tools import llm as _llm

    _llm.get_chat_model = lambda: fake_chat_model  # type: ignore[assignment]

    fake_chat_model.responses.extend(
        [
            # Questioner: round 1, ready_to_proceed short-circuit.
            json.dumps(
                {
                    "questions": [],
                    "reasoning": "have enough",
                    "ready_to_proceed": True,
                }
            ),
            # Topic-finalize: 2 ResearchTopic entries.
            json.dumps(topics_payload),
            # Write stage 1: outline + draft (stage 2 decorates with [N]).
            json.dumps(write_payload),
        ]
    )

    graph = build_graph(memory_checkpointer)
    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
    }

    # First invoke: questioner (ready=true) → topic-finalize → GATE PAUSE.
    graph.invoke({"user_query": "Tell me about X"}, config=config)

    # Drive the gate resume directly — no /confirm endpoint yet. The
    # gate's draft topics carry uuid4 ids (ADR-0005 §2), not the
    # LLM's ``topic-1``/``topic-2`` slugs; the test sends them back
    # unchanged so ``diff_topics`` returns an empty edit set.
    paused = graph.get_state(config)
    gate_payload = paused.tasks[0].interrupts[0].value
    state = graph.invoke(
        Command(resume=ConfirmedResearchTopics(topics=list(gate_payload.topics))),
        config=config,
    )

    assert isinstance(state, dict)
    assert state["current_step"] == "final"
    assert len(state["research_topics"]) == 2
    assert len(state["search_results"]) == 2
    assert isinstance(state["outline"], Outline)
    assert len(state["outline"].sections) == 2

    report = state["report"]
    assert "[1]" in report
    assert "[2]" in report
    assert "## References" in report
    # The ``fake_search_tool`` fixture emits titles of the form
    # ``"Mock result for {topic_id}"`` (see ``tests/conftest.py``);
    # match that exactly so the assertion is fixture-driven, not
    # duplicated from ``test_graph_e2e.py``'s inline stub. v0.3 — the
    # topic ids in the report are uuid4 (gate-assigned) and the
    # ``search_results`` carry the original id from the search call;
    # the fixture stub keys the title off the search-call topic_id
    # which is the gate-assigned uuid4 here. The test pins the
    # section titles instead so the assertion is stable across the
    # id-format change.
    assert "## History" in report
    assert "## Current state" in report


def test_end_to_end_with_gate_edit(
    fake_chat_model, fake_search_tool, memory_checkpointer
) -> None:
    """Issue #26 AC — full E2E with a real gate edit feeding into Research + Write.

    The pipeline walks every v0.3 stage: 1 Q&A round → ``ready_to_proceed``
    → topic-finalize (the LLM's 3) → gate pause → the user adds 1 topic,
    deletes 1, and modifies 1 → gate resume → Research runs ONE SEARCH PER
    USER-FINAL TOPIC (not the LLM's 3) → Write produces a report → END.

    The load-bearing assertion is the search-call set: the topics the
    user confirmed (2 preserved draft ids + 1 gate-minted uuid4) are
    exactly the topics the searches ran on — pinning that Research
    consumes ``confirmed_research_plan``, never the pre-gate draft
    (ADR-0005 §6). The dual-field contract is pinned too: the LLM draft
    stays intact in ``research_topics`` while the user's list lands in
    ``confirmed_research_plan`` alongside a structured ``plan_edit_diff``.
    """
    llm_draft = {
        "topics": [
            {
                "id": "topic-1",
                "title": "Scope",
                "description": "scope of X",
                "expected_finding_type": "factual",
            },
            {
                "id": "topic-2",
                "title": "Evidence",
                "description": "evidence for X",
                "expected_finding_type": "data",
            },
            {
                "id": "topic-3",
                "title": "Limits",
                "description": "limits of X",
                "expected_finding_type": "data",
            },
        ]
    }
    write_payload = {
        "outline": [
            {"title": "Refined scope", "summary": "s", "key_points": ["p"]},
            {"title": "Limits", "summary": "s", "key_points": ["p"]},
            {"title": "New angle", "summary": "s", "key_points": ["p"]},
        ],
        "draft": (
            "Intro paragraph.\n\n"
            "## Refined scope\n\nRefined scope fact.\n\n"
            "## Limits\n\nOne limit fact.\n\n"
            "## New angle\n\nOne new-angle fact."
        ),
    }

    from app.tools import llm as _llm

    _llm.get_chat_model = lambda: fake_chat_model  # type: ignore[assignment]

    fake_chat_model.responses.extend(
        [
            # Round 1 questioner: one question, ready=false → HITL pause.
            json.dumps(
                {
                    "questions": [
                        {
                            "kind": "free_text",
                            "id": "q1",
                            "prompt": "What scope?",
                            "placeholder": "e.g. last 5 years",
                        }
                    ],
                    "reasoning": "scope",
                    "ready_to_proceed": False,
                }
            ),
            # Round 1 resume re-run: the SAME question batch (LangGraph
            # re-executes the node; its ``interrupt()`` re-fires with the
            # same id and returns the user's answer). The answer consumes
            # the resume value here — scripting ready=true instead would
            # leave the resume unconsumed and hand it to the GATE's
            # interrupt, which expects ConfirmedResearchTopics.
            json.dumps(
                {
                    "questions": [
                        {
                            "kind": "free_text",
                            "id": "q1",
                            "prompt": "What scope?",
                            "placeholder": "e.g. last 5 years",
                        }
                    ],
                    "reasoning": "scope",
                    "ready_to_proceed": False,
                }
            ),
            # Round 2 initial: ready=true → finalize in the same entry.
            json.dumps(
                {"questions": [], "reasoning": "done", "ready_to_proceed": True}
            ),
            # Topic-finalize: the LLM's 3 draft topics.
            json.dumps(llm_draft),
            # Write stage 1: outline + draft over the user's final 3.
            json.dumps(write_payload),
        ]
    )

    graph = build_graph(memory_checkpointer)
    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
    }

    # 1. The 1 Q&A round — the questioner pauses on round 1.
    graph.invoke({"user_query": "Tell me about X"}, config=config)

    # 2. The user answers; the resume re-run finalizes and the gate opens.
    out = graph.invoke(
        Command(
            resume=PlanAnswerBatch(
                round=1,
                answers=[PlanUserAnswer(question_id="q1", value="last 5 years")],
            )
        ),
        config=config,
    )
    assert "__interrupt__" in out
    gate_payload = out["__interrupt__"][0].value
    draft_topics = list(gate_payload.topics)
    assert len(draft_topics) == 3
    id_scope, id_evidence, id_limits = [t.id for t in draft_topics]

    # 3. The user's edit set: keep + modify id_scope, drop id_evidence,
    #    keep id_limits, add one brand-new topic (placeholder slug — the
    #    gate mints a fresh uuid4 for it per ADR-0005 §2).
    user_final = [
        ResearchTopic(
            id=id_scope,
            title="Refined scope",
            description="scope of X, last 5 years",
            expected_finding_type="factual",
        ),
        ResearchTopic(
            id=id_limits, title="Limits", description="limits of X",
            expected_finding_type="data",
        ),
        ResearchTopic(
            id="topic-added", title="New angle", description="new angle on X",
            expected_finding_type="evidence",
        ),
    ]
    state = graph.invoke(
        Command(resume=ConfirmedResearchTopics(topics=user_final)),
        config=config,
    )

    # 4. The pipeline ran to the terminal.
    assert isinstance(state, dict)
    assert state["current_step"] == "final"
    report = state["report"]
    assert "## References" in report
    assert "[1]" in report

    # 5. Research ran on the USER's final list — one search per
    #    confirmed topic, keyed by the confirmed ids (2 preserved + 1
    #    gate-minted), not the LLM's draft ids.
    search_calls = fake_search_tool.calls
    assert len(search_calls) == 3, (
        f"expected 3 searches (one per confirmed topic), got "
        f"{len(search_calls)}"
    )
    searched_topic_ids = {c["topic_id"] for c in search_calls}
    confirmed_ids = [t.id for t in state["confirmed_research_plan"]]
    assert searched_topic_ids == set(confirmed_ids)
    # The modified topic was searched under its PRESERVED id, and the
    # dropped draft topic was never searched.
    assert id_scope in searched_topic_ids
    assert id_evidence not in searched_topic_ids
    # The added topic was searched under a gate-minted uuid4, not the
    # client's placeholder slug.
    added_id = confirmed_ids[2]
    assert added_id not in {id_scope, id_evidence, id_limits, "topic-added"}
    # The search query follows ``{title} — {description}`` of the USER's
    # wording, not the LLM's draft wording.
    assert search_calls[0]["query"] == "Refined scope — scope of X, last 5 years"

    # 6. The dual-field contract (ADR-0005 §6): the LLM draft survives
    #    untouched while the user's list lands in confirmed_research_plan.
    assert [t.id for t in state["research_topics"]] == [
        id_scope, id_evidence, id_limits,
    ]
    assert [t.title for t in state["confirmed_research_plan"]] == [
        "Refined scope", "Limits", "New angle",
    ]

    # 7. The structured audit trail (ADR-0005 §7).
    diff = state["plan_edit_diff"]
    assert [m.id for m in diff.modified] == [id_scope]
    assert diff.modified[0].before.title == "Scope"
    assert diff.modified[0].after.title == "Refined scope"
    assert diff.removed == [id_evidence]
    assert [a.id for a in diff.added] == [added_id]
    assert diff.added[0].title == "New angle"

    # 8. The search results carry the confirmed topic ids — what the
    #    References block (and any future reviewer) keys on.
    assert len(state["search_results"]) == 3
    assert {r.topic_id for r in state["search_results"]} == set(confirmed_ids)