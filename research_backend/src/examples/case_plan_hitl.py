"""Case example — Plan HITL loop with a real LLM and popup dialogs (issue #12),
plus the Plan→Research transition gate round-trip (issue #23).

Builds the plan subgraph standalone and drives it through the real
LangGraph interrupt / resume loop. The questioner LLM is a live
``get_chat_model()`` instance, not a stub. On every questioner
interrupt the example prints the typed ``PlanInterruptPayload`` and
pops a GUI dialog for each question; the user's typed answer is
wrapped in a ``PlanAnswerBatch`` and resumed via ``Command(resume=...)``.
The loop exits as soon as the questioner sets ``ready_to_proceed=true``
(or the 5-round soft cap is hit) and the plan_questioner node
materialises the topics.

v0.3 (issue #23) inserts the Plan→Research transition gate between
``plan_questioner`` and ``END``. After the topics are finalised the
gate reads them, emits a typed ``PlanConfirmInterruptPayload``, and
pauses. The example then prints the draft and pops a console prompt
that lets the user choose between three scripted edit shapes:

- **accept** — send the draft back unchanged (the diff is empty;
  the gate writes ``confirmed_research_plan`` and ``plan_edit_diff``
  as zeros and the subgraph reaches ``END``).
- **delete one topic** — drop the topic with the given 1-based
  index from the draft; the gate records a ``removed`` entry.
- **modify one topic** — change the ``title`` of the topic with
  the given 1-based index; the gate records a ``modified`` entry.

After the resume the example prints the resulting
``PlanEditDiff`` (or its absence, if the user accepted unchanged)
and the final ``current_step`` / ``confirmed_research_plan`` it
persists.

Run it as a script::

    PYTHONPATH=src uv run python -m examples.case_plan_hitl

``OPENAI_API_KEY`` (or whatever provider ``get_chat_model`` is wired to)
must be set in the environment. Each question pops a tkinter dialog:

- ``confirmation`` — Yes / No buttons.
- ``single_choice`` — radio buttons, one ``option_id`` returned.
- ``multi_choice`` — check buttons, a list of ``option_id``s returned.
- ``free_text`` — single-line text input (``Cancel`` returns empty).

The dialog helpers (``_ask_yes_no``, ``_ask_string``, ``_ask_choice``)
are module-level so tests can monkeypatch them to script answers
without needing a display.
"""

from __future__ import annotations

import tkinter as tk
import uuid
from tkinter import messagebox, simpledialog
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from rich import print as rprint

from app.graph.subgraphs.plan import build_plan_subgraph
from app.models import (
    ConfirmedResearchTopics,
    PlanAnswerBatch,
    PlanUserAnswer,
    ResearchTopic,
)

# ─────────────────────────────────────────────────────────────────────────────
# The user query this example drives. The LLM sees this verbatim as the
# opening of every question-emit prompt.
# ─────────────────────────────────────────────────────────────────────────────

USER_QUERY = "深度调研 欧美国家 酒店旅游业 AI Agent 的落地方案"


# ─────────────────────────────────────────────────────────────────────────────
# tkinter — lazy root, kept visible-but-tiny so the window manager
# registers the process as a GUI app (otherwise the dialogs never
# appear on macOS — the process runs in the background with no UI).
# ─────────────────────────────────────────────────────────────────────────────

_tk_root: tk.Tk | None = None


def _get_tk_root() -> tk.Tk:
    global _tk_root
    if _tk_root is None:
        _tk_root = tk.Tk()
        # 1x1 in the top-left corner — effectively invisible but the
        # window manager treats the process as a foreground GUI app
        # so child Toplevels actually show. ``withdraw()`` here would
        # make the dialogs vanish on macOS.
        _tk_root.geometry("1x1+0+0")
    return _tk_root


# ─────────────────────────────────────────────────────────────────────────────
# Dialog helpers — module-level so tests can monkeypatch them.
# ─────────────────────────────────────────────────────────────────────────────


def _ask_yes_no(title: str, prompt: str) -> bool:
    """Pop a Yes/No dialog. Returns the boolean answer."""
    rprint(f"  [bold cyan]→ {title}[/bold cyan] (waiting for click)")
    return bool(messagebox.askyesno(title=title, message=prompt, parent=_get_tk_root()))


def _ask_string(title: str, prompt: str) -> str:
    """Pop a free-text input dialog. ``Cancel`` returns an empty string."""
    rprint(f"  [bold cyan]→ {title}[/bold cyan] (waiting for input)")
    result = simpledialog.askstring(
        title=title, prompt=prompt, parent=_get_tk_root()
    )
    return (result or "").strip()


def _ask_choice(
    title: str,
    prompt: str,
    options: list[tuple[str, str]],
    multi: bool,
) -> Any:
    """Pop a single- or multi-choice dialog.

    ``options`` is a list of ``(option_id, label)`` pairs. With
    ``multi=False`` returns the chosen ``option_id`` (or ``""`` on
    ``Cancel``). With ``multi=True`` returns the list of chosen
    ``option_id``s (possibly empty on ``Cancel``).
    """
    rprint(
        f"  [bold cyan]→ {title}[/bold cyan] (waiting for selection, "
        f"{'multi' if multi else 'single'})"
    )
    root = _get_tk_root()
    dialog = tk.Toplevel(root)
    dialog.title(title)
    dialog.geometry("560x420")
    # macOS / X11: keep the dialog on top of every other window so
    # the user actually sees it.
    dialog.attributes("-topmost", True)

    tk.Label(
        dialog,
        text=prompt,
        font=("Helvetica", 12, "bold"),
        wraplength=520,
        justify="left",
        anchor="w",
    ).pack(pady=12, padx=12, fill="x")

    body = tk.Frame(dialog)
    body.pack(fill="both", expand=True, padx=12)

    if multi:
        tk.Label(
            body,
            text="(Select all that apply)",
            font=("Helvetica", 9, "italic"),
        ).pack(anchor="w")
        checks: dict[str, tk.BooleanVar] = {
            oid: tk.BooleanVar(value=False) for oid, _ in options
        }
        for oid, label in options:
            tk.Checkbutton(
                body,
                text=f"{oid}: {label}",
                variable=checks[oid],
                wraplength=480,
                justify="left",
                anchor="w",
            ).pack(fill="x", pady=2)
    else:
        radio_var = tk.StringVar(value=options[0][0])
        for oid, label in options:
            tk.Radiobutton(
                body,
                text=f"{oid}: {label}",
                variable=radio_var,
                value=oid,
                wraplength=480,
                justify="left",
                anchor="w",
            ).pack(fill="x", pady=2)

    result: list = []

    def on_ok() -> None:
        if multi:
            result.extend(oid for oid, var in checks.items() if var.get())
        else:
            result.append(radio_var.get())
        dialog.destroy()

    def on_cancel() -> None:
        dialog.destroy()

    button_frame = tk.Frame(dialog)
    button_frame.pack(pady=10)
    tk.Button(button_frame, text="OK", command=on_ok, width=10).pack(
        side="left", padx=5
    )
    tk.Button(button_frame, text="Cancel", command=on_cancel, width=10).pack(
        side="left", padx=5
    )

    dialog.transient(root)
    dialog.grab_set()
    dialog.lift()
    dialog.focus_force()
    root.update_idletasks()  # force the WM to actually draw it
    dialog.wait_window()

    if multi:
        return result
    return result[0] if result else ""


# ─────────────────────────────────────────────────────────────────────────────
# Adapter — match a ``PlanQuestion`` to the right dialog.
# ─────────────────────────────────────────────────────────────────────────────


def _prompt_for_answer(question) -> Any:
    """Pop a dialog matched to ``question.kind`` and return a typed answer.

    Return type matches the ``PlanUserAnswer.value`` shape: bool for
    ``confirmation``, list[str] for ``multi_choice``, str otherwise.
    """
    kind = question.kind
    if kind == "confirmation":
        return _ask_yes_no(
            title=f"Question {question.id}",
            prompt=f"{question.prompt}\n\n(Yes / No)",
        )
    if kind in ("single_choice", "multi_choice"):
        opts = [(o.option_id, o.label) for o in question.options]
        return _ask_choice(
            title=f"Question {question.id}",
            prompt=question.prompt,
            options=opts,
            multi=(kind == "multi_choice"),
        )
    if kind == "free_text":
        placeholder = getattr(question, "placeholder", None)
        full_prompt = question.prompt
        if placeholder:
            full_prompt = f"{question.prompt}\n\n(placeholder: {placeholder})"
        return _ask_string(title=f"Question {question.id}", prompt=full_prompt)
    raise RuntimeError(f"Unknown question kind: {kind!r}")


def _collect_answers(questions) -> list[PlanUserAnswer]:
    """Pop one dialog per question and return the typed ``PlanUserAnswer``s."""
    return [
        PlanUserAnswer(question_id=q.id, value=_prompt_for_answer(q))
        for q in questions
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-printing.
# ─────────────────────────────────────────────────────────────────────────────


def _print_interrupt(round_label: str, interrupt_value) -> None:
    """Pretty-print a ``PlanInterruptPayload`` and the questions to answer."""
    rprint(
        f"\n[cyan]{round_label}[/cyan] (round={interrupt_value.round}, "
        f"reasoning={interrupt_value.reasoning!r})"
    )
    for q in interrupt_value.questions:
        option_ids = [o.option_id for o in getattr(q, "options", [])]
        rprint(f"  → {q.kind} {q.id!r}: {q.prompt}  options={option_ids}")
        print()


def _print_gate_interrupt(payload) -> None:
    """Pretty-print a ``PlanConfirmInterruptPayload`` (gate, v0.3 — issue #23)."""
    rprint(
        f"\n[cyan]Gate[/cyan] (reasoning={payload.reasoning!r}, "
        f"{len(payload.topics)} topic(s) proposed)"
    )
    for idx, topic in enumerate(payload.topics, start=1):
        rprint(
            f"  [{idx}] {topic.id} — {topic.title} "
            f"({topic.expected_finding_type})"
        )
        rprint(f"      {topic.description}")


# ─────────────────────────────────────────────────────────────────────────────
# Gate prompt — three scripted edit shapes (issue #23 AC). Module-level so
# tests can monkeypatch.
# ─────────────────────────────────────────────────────────────────────────────


def _prompt_gate_edit(
    draft: list[ResearchTopic],
) -> ConfirmedResearchTopics | None:
    """Pop a console prompt that lets the user pick one of three edit shapes.

    Returns a ``ConfirmedResearchTopics`` with the user's final list, or
    ``None`` if the prompt is unavailable (non-interactive shell). The
    shape is intentionally scripted — the case example is meant to
    exercise the gate round-trip, not to surface a fully-fledged editor.
    """
    try:
        choice = input(
            "\n[gate] accept / delete <i> / modify <i> / list (default=accept): "
        ).strip()
    except EOFError:
        return None

    parts = choice.split()
    if not parts or parts[0] == "accept":
        return ConfirmedResearchTopics(topics=list(draft))
    if parts[0] == "list":
        for idx, t in enumerate(draft, start=1):
            print(f"  [{idx}] {t.id} — {t.title}")
        return _prompt_gate_edit(draft)
    if parts[0] == "delete" and len(parts) == 2 and parts[1].isdigit():
        i = int(parts[1]) - 1
        if not 0 <= i < len(draft):
            print(f"  index out of range (1..{len(draft)})")
            return _prompt_gate_edit(draft)
        final = [t for j, t in enumerate(draft) if j != i]
        return ConfirmedResearchTopics(topics=final)
    if parts[0] == "modify" and len(parts) == 2 and parts[1].isdigit():
        i = int(parts[1]) - 1
        if not 0 <= i < len(draft):
            print(f"  index out of range (1..{len(draft)})")
            return _prompt_gate_edit(draft)
        new_title = input(f"  new title for [{i + 1}] (was {draft[i].title!r}): ").strip()
        if not new_title:
            new_title = draft[i].title
        final = list(draft)
        final[i] = ResearchTopic(
            id=draft[i].id,
            title=new_title,
            description=draft[i].description,
            expected_finding_type=draft[i].expected_finding_type,
        )
        return ConfirmedResearchTopics(topics=final)
    print(f"  unrecognised choice {choice!r}; please retry")
    return _prompt_gate_edit(draft)


def _print_final_diff(diff) -> None:
    """Pretty-print the gate's ``PlanEditDiff`` (or 'no edits')."""
    if diff.added or diff.removed or diff.modified:
        rprint("\n[green]Plan edit diff:[/green]")
    if diff.added:
        rprint(f"  [+] added ({len(diff.added)}):")
        for t in diff.added:
            rprint(f"      - {t.id} — {t.title}")
    if diff.removed:
        rprint(f"  [-] removed ({len(diff.removed)}):")
        for tid in diff.removed:
            rprint(f"      - {tid}")
    if diff.modified:
        rprint(f"  [~] modified ({len(diff.modified)}):")
        for entry in diff.modified:
            rprint(f"      - {entry.id}: {entry.before.title!r} → {entry.after.title!r}")
    if not (diff.added or diff.removed or diff.modified):
        rprint("\n[green]Plan edit diff: (no edits — user accepted the draft)[/green]")


# ─────────────────────────────────────────────────────────────────────────────
# Driver — invoke the subgraph, pop dialogs, resume until finalize.
# ─────────────────────────────────────────────────────────────────────────────


def _run_hitl_demo() -> None:
    graph = build_plan_subgraph(checkpointer=MemorySaver())
    config = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": 25,
    }

    rprint(f"[bold]User query:[/bold] {USER_QUERY}\n")

    round_num = 1
    result = graph.invoke({"user_query": USER_QUERY}, config=config)

    # Phase 1 — drive the questioner loop until ready_to_proceed (or
    # the soft cap forces topic-finalize). Each iteration is one
    # full round: question-emit → interrupt → user answers → resume.
    # The interrupt value is always a ``PlanInterruptPayload`` here.
    while "__interrupt__" in result:
        interrupt_value = result["__interrupt__"][0].value
        # Phase 2 — once the questioner finalizes, the next interrupt
        # is the gate's ``PlanConfirmInterruptPayload``. Branch by
        # payload type so the driver knows whether to pop a question
        # dialog or a topic-edit prompt.
        if not _is_questioner_interrupt(interrupt_value):
            break
        _print_interrupt(f"Round {round_num} interrupt", interrupt_value)
        answers = _collect_answers(interrupt_value.questions)
        answer_batch = PlanAnswerBatch(round=round_num, answers=answers)
        result = graph.invoke(Command(resume=answer_batch), config=config)
        round_num += 1

    # Phase 2 — drive the Plan→Research transition gate. The questioner
    # either said ready_to_proceed (and the gate now pauses) or the
    # soft cap fired topic-finalize (and the gate pauses). Either way
    # the next ``__interrupt__`` is the gate's typed payload.
    if "__interrupt__" in result:
        gate_payload = result["__interrupt__"][0].value
        _print_gate_interrupt(gate_payload)
        confirmed = _prompt_gate_edit(list(gate_payload.topics))
        if confirmed is None:
            rprint(
                "\n[yellow]Gate: non-interactive shell — accepting the "
                "draft unchanged.[/yellow]"
            )
            confirmed = ConfirmedResearchTopics(topics=list(gate_payload.topics))
        result = graph.invoke(Command(resume=confirmed), config=config)

    rprint("\n[green]Plan completed.[/green]")
    rprint(f"[green]current_step:[/green] {result['current_step']}")
    rprint(f"[green]rounds used:[/green] {round_num - 1}")
    rprint("[green]research_topics:[/green]")
    for topic in result["research_topics"]:
        rprint(f"  - {topic.id}: {topic.title} — {topic.description}")

    # Phase 3 — print the gate's audit trail if the gate ran. The
    # ``plan_edit_diff`` field is on the state only after the gate
    # confirmed; ``confirmed_research_plan`` carries the user's
    # final list.
    diff = result.get("plan_edit_diff")
    if diff is not None:
        _print_final_diff(diff)
    confirmed_plan = result.get("confirmed_research_plan")
    if confirmed_plan is not None:
        rprint("\n[green]confirmed_research_plan:[/green]")
        for topic in confirmed_plan:
            rprint(f"  - {topic.id}: {topic.title}")


def _is_questioner_interrupt(value: Any) -> bool:
    """True if ``value`` is a questioner ``PlanInterruptPayload`` (not the gate).

    Two interrupt types share the same channel in v0.3: the questioner
    emits ``PlanInterruptPayload`` (carrying questions) and the gate
    emits ``PlanConfirmInterruptPayload`` (carrying topics). The
    discriminator is the presence of a ``topics`` field — see
    ``app.api.events._is_gate_interrupt_value`` for the same check on
    the SSE side. Re-implemented here to keep the example
    self-contained (it doesn't depend on the API layer).
    """
    return hasattr(value, "questions") and not hasattr(value, "topics")


if __name__ == "__main__":
    _run_hitl_demo()
