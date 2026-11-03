"""
Static checks on the full-text assessment codebook.

Run from notebook 06 to document the branching logic, and again at app start-up so
that a later edit to the xlsx cannot silently break it. Three severities:

  error    the tool cannot behave correctly; the app refuses to start
  warning  probably a mistake, but the tool still works
  info     worth knowing, nothing to fix (e.g. a question with no escape label
           whose condition guarantees an answer exists)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from src.full_text_assessment.codebook import (
    EXCLUSIVE_LABELS, LEVELS, MULTI_SEP, NA, QID_PATTERN, Codebook, Comparison, BoolOp,
    Question,
)

ERROR, WARNING, INFO = "error", "warning", "info"


@dataclass(frozen=True)
class Issue:
    severity: str
    qid: str
    check: str
    message: str


def _comparisons(node) -> list[Comparison]:
    if node is None:
        return []
    if isinstance(node, Comparison):
        return [node]
    assert isinstance(node, BoolOp)
    return [c for child in node.children for c in _comparisons(child)]


def _find_cycles(codebook: Codebook) -> list[list[str]]:
    cycles: list[list[str]] = []
    state: dict[str, int] = {}          # 1 = on the stack, 2 = finished

    def visit(qid: str, path: list[str]) -> None:
        state[qid] = 1
        for parent in codebook.by_id[qid].parents:
            if parent not in codebook.by_id:
                continue
            if state.get(parent) == 1:
                cycles.append(path[path.index(parent):] + [parent] if parent in path
                              else [qid, parent])
            elif parent not in state:
                visit(parent, path + [parent])
        state[qid] = 2

    for q in codebook.questions:
        if q.id not in state:
            visit(q.id, [q.id])
    return cycles


def audit(codebook: Codebook) -> list[Issue]:
    issues: list[Issue] = []

    def add(severity: str, qid: str, check: str, message: str) -> None:
        issues.append(Issue(severity, qid, check, message))

    for qid, message in codebook.parse_errors:
        add(ERROR, qid, "parse", message)

    # ── identity and structure ───────────────────────────────────────────────
    # fixIDs are the key of every answer: they must be unique, also ignoring case (a
    # CSV opened in Excel, or a typo in a condition, must not be able to confuse two).
    ids = [q.id for q in codebook.questions]
    for qid in sorted({i for i in ids if ids.count(i) > 1}):
        add(ERROR, qid, "unique fixID", f"fixID {qid} is used {ids.count(qid)} times")
    folded: dict[str, set[str]] = {}
    for qid in ids:
        folded.setdefault(qid.casefold(), set()).add(qid)
    for variants in folded.values():
        if len(variants) > 1:
            names = sorted(variants)
            add(ERROR, names[0], "unique fixID",
                f"fixIDs differ only in case: {', '.join(names)}")
    for qid in ids:
        if not re.fullmatch(QID_PATTERN, qid):
            add(ERROR, qid, "fixID", f"fixID {qid!r} does not match {QID_PATTERN} "
                                     "(section key, a dot, then letters, digits or _)")
    shown = [q.display_id for q in codebook.questions if q.display_id]
    for did in sorted({d for d in shown if shown.count(d) > 1}):
        add(WARNING, did, "unique ID", f"display ID {did} is used {shown.count(did)} times")

    position = {q.id: i for i, q in enumerate(codebook.questions)}
    for q in codebook.questions:
        if q.level not in LEVELS:
            add(ERROR, q.id, "level", f"Level is {q.level!r}; expected one of {sorted(LEVELS)}")
        if q.section_key not in codebook.sections:
            add(WARNING, q.id, "section", f"section key {q.section_key!r} is not in Sheet2")
        elif codebook.sections[q.section_key] != q.section:
            add(WARNING, q.id, "section",
                f"section name {q.section!r} differs from Sheet2 "
                f"({codebook.sections[q.section_key]!r})")
        if not q.id.startswith(q.section_key + "."):
            add(WARNING, q.id, "section", f"fixID does not start with its section key {q.section_key!r}")

    # ── labels ───────────────────────────────────────────────────────────────
    for q in codebook.questions:
        if q.pool != q.id:
            owner = codebook.by_id.get(q.pool)
            if owner is None:
                hint = codebook.by_display.get(q.pool)
                add(ERROR, q.id, "label pool",
                    f"'same labels as {q.pool}': no such fixID"
                    + (f"; {q.pool} is a display ID, use {hint.id}" if hint else ""))
                continue
            if owner.pool != owner.id:
                add(ERROR, q.id, "label pool", f"{q.pool} itself borrows labels; chains are not supported")
            if owner.kind != q.kind:
                add(WARNING, q.id, "label pool",
                    f"shares labels with {q.pool} but is {q.kind} while {q.pool} is {owner.kind}")
            if owner.updatable != q.updatable:
                add(WARNING, q.id, "label pool",
                    f"shares labels with {q.pool} but only one of them is updatable")
            continue

        texts = [label.text for label in q.labels]
        folded = [t.casefold() for t in texts]
        for text in sorted({t for t in texts if folded.count(t.casefold()) > 1}):
            add(ERROR, q.id, "labels", f"duplicate label {text!r}")
        for text in texts:
            if MULTI_SEP.strip() in text:
                add(ERROR, q.id, "labels", f"label {text!r} contains the separator '|'")
            if text.endswith("*"):
                add(WARNING, q.id, "labels", f"label {text!r} still ends with '*'")
        # Escape labels may sit uncoded next to coded ones: they are stored as text,
        # so they never pass for a point on the numeric scale.
        codes = [label.code for label in q.labels
                 if label.code is not None or label.text.casefold() not in EXCLUSIVE_LABELS]
        if any(c is not None for c in codes):
            if None in codes:
                add(ERROR, q.id, "labels", "mixes coded ('N = ...') and uncoded labels "
                    "(only escape labels may be uncoded)")
            if len(set(codes)) != len(codes):
                add(ERROR, q.id, "labels", "duplicate codes")
            if q.updatable:
                add(WARNING, q.id, "labels", "coded labels on an updatable question")
        if q.kind in ("single", "multi") and not q.labels and not q.updatable:
            add(ERROR, q.id, "labels", f"{q.raw_type} with no labels to choose from")

        escapes = [t for t in texts if t.casefold() in EXCLUSIVE_LABELS]
        if not escapes and q.kind in ("multi", "free", "integer", "year"):
            if q.condition is not None:
                add(INFO, q.id, "escape label",
                    f"no 'not reported'/'unclear'-type label; relies on its condition "
                    f"({q.condition_text}) to guarantee an answer exists")
            else:
                add(WARNING, q.id, "escape label",
                    "unconditional, and no 'not reported'/'unclear'-type label to "
                    "answer with when the paper is silent")

    # ── conditions ───────────────────────────────────────────────────────────
    for q in codebook.questions:
        for comp in _comparisons(q.condition):
            parent = codebook.by_id.get(comp.qid)
            if parent is None:
                hint = codebook.by_display.get(comp.qid)
                add(ERROR, q.id, "condition",
                    f"uses display ID {comp.qid}; conditions must use fixIDs: {hint.id}" if hint
                    else f"refers to unknown question {comp.qid}")
                continue
            if comp.qid == q.id:
                add(ERROR, q.id, "condition", "refers to itself")
                continue
            if position[comp.qid] > position[q.id]:
                add(WARNING, q.id, "order", f"parent {comp.qid} comes after it in the xlsx")
            if q.level == "paper" and parent.level == "dataset":
                add(ERROR, q.id, "level",
                    f"paper-level question depends on dataset-level {comp.qid}")
            if parent.kind not in ("single", "multi", "free", "integer", "year"):
                continue
            known = {label.value.casefold() for label in codebook.labels(parent)}
            known |= {label.text.casefold() for label in codebook.labels(parent)}
            for value in comp.values:
                if value.casefold() in known:
                    continue
                if value == NA:
                    add(INFO, q.id, "condition",
                        f"mentions 'NA', which is no label of {comp.qid}; harmless, "
                        "because a skipped parent already skips its children")
                elif parent.updatable:
                    add(WARNING, q.id, "condition",
                        f"value {value!r} is not (yet) a label of {comp.qid}")
                else:
                    add(ERROR, q.id, "condition",
                        f"value {value!r} is not a label of {comp.qid} "
                        f"(labels: {[l.value for l in codebook.labels(parent)]})")
            if parent.kind == "free" and not parent.labels:
                add(WARNING, q.id, "condition",
                    f"depends on free-text {comp.qid}, which has no labels to compare with")

    for cycle in _find_cycles(codebook):
        add(ERROR, cycle[0], "cycle", "dependency cycle: " + " -> ".join(cycle))

    order = {ERROR: 0, WARNING: 1, INFO: 2}
    return sorted(issues, key=lambda i: (order[i.severity], position.get(i.qid, 999)))


def issues_frame(issues: list[Issue]) -> pd.DataFrame:
    return pd.DataFrame([vars(i) for i in issues],
                        columns=["severity", "qid", "check", "message"])


def question_table(codebook: Codebook) -> pd.DataFrame:
    """One row per question: the logic at a glance, for the notebook."""
    rows = []
    for q in codebook.questions:
        labels = codebook.labels(q)
        rows.append({
            "fixID": q.id,
            "ID": q.display_id,
            "level": q.level,
            "type": q.raw_type,
            "condition": q.condition_text or "",
            "parents": ", ".join(q.parents),
            "children": ", ".join(c.id for c in codebook.children(q.id)),
            "label pool": q.pool if q.pool != q.id else "",
            "n labels": len(labels),
            "escape labels": ", ".join(l.text for l in labels
                                       if l.text.casefold() in EXCLUSIVE_LABELS),
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# The branching logic as a graph
# ──────────────────────────────────────────────────────────────────────────────

_OP_SYMBOL = {"=": "=", "!=": "≠", "in": "∈", "not in": "∉"}


def edge_label(comp: Comparison, width: int = 16) -> str:
    """Compact text for one comparison, e.g. "∈ yes, direct / yes, indirect"."""
    values = [v if len(v) <= width else v[:width - 1] + "…" for v in comp.values]
    return f"{_OP_SYMBOL[comp.op]} " + " / ".join(values)


def dependency_graph(codebook: Codebook):
    """networkx DiGraph, parent -> child, one edge per comparison in the condition."""
    import networkx as nx

    graph = nx.DiGraph()
    for q in codebook.questions:
        graph.add_node(q.id, section=q.section_key, level=q.level, text=q.text,
                       conditional=q.condition is not None)
    for q in codebook.questions:
        for comp in _comparisons(q.condition):
            if comp.qid in codebook.by_id:
                first = comp.values[0]
                short = f"{_OP_SYMBOL[comp.op]} {first}" + (
                    f" +{len(comp.values) - 1}" if len(comp.values) > 1 else "")
                graph.add_edge(comp.qid, q.id, label=edge_label(comp), short=short,
                               joined=isinstance(q.condition, BoolOp))
    return graph


def to_mermaid(codebook: Codebook, text_width: int = 38) -> str:
    """The branching logic as a Mermaid flowchart, one subgraph per section."""
    def node_id(qid: str) -> str:
        return qid.replace(".", "_")

    def esc(text: str) -> str:
        return text.replace('"', "#quot;").replace("|", "/")

    lines = ["flowchart LR"]
    for key, name in codebook.sections.items():
        questions = codebook.section_questions(key)
        if not questions:
            continue
        lines.append(f'  subgraph {key}["{esc(name)}"]')
        for q in questions:
            text = q.text if len(q.text) <= text_width else q.text[:text_width - 1] + "…"
            label = f"{q.id} <small>({q.display_id})</small><br/><small>{esc(text)}</small>"
            shape = ('["{}"]' if q.level == "paper" else '[["{}"]]').format(label)
            lines.append(f"    {node_id(q.id)}{shape}")
        lines.append("  end")
    for q in codebook.questions:
        for comp in _comparisons(q.condition):
            lines.append(f'  {node_id(comp.qid)} -->|"{esc(edge_label(comp, 22))}"| {node_id(q.id)}')
    dataset = [node_id(q.id) for q in codebook.level("dataset")]
    lines.append("  classDef dataset fill:#fdeee7,stroke:#eb6834")
    lines.append("  classDef paper fill:#e8f1fb,stroke:#2a78d6")
    lines.append(f"  class {','.join(dataset)} dataset")
    lines.append("  class " + ",".join(node_id(q.id) for q in codebook.level("paper")) + " paper")
    return "\n".join(lines)


PAPER_COLOR, DATASET_COLOR = "#2a78d6", "#eb6834"      # categorical slots 1 and 2
INK, INK_MUTED, RULE = "#0b0b0b", "#52514e", "#b9b8b2"


def draw_dependency_graph(codebook: Codebook, text_width: int = 30):
    """
    The branching logic as a static figure.

    One column per section (so sections are read by position, not colour); within a
    column, questions go down in xlsx order and step right by their depth in the
    section's branching. Colour and shape both give the level: blue circles are
    paper-level, orange squares dataset-level. Edges carry the triggering values;
    a parent with several children on the same values is labelled once.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    graph = dependency_graph(codebook)
    keys = [k for k in codebook.sections if codebook.section_questions(k)]
    col_w, dx, row_h = 3.3, 0.28, 1.0

    depth: dict[str, int] = {}
    for q in codebook.dependency_order(codebook.questions):
        same = [depth[p] + 1 for p in q.parents
                if p in depth and codebook.by_id[p].section_key == q.section_key]
        depth[q.id] = max(same, default=0)

    pos: dict[str, tuple[float, float]] = {}
    for i, key in enumerate(keys):
        for row, q in enumerate(codebook.section_questions(key)):
            pos[q.id] = (i * col_w + depth[q.id] * dx, -row * row_h)
    n_rows = max(len(codebook.section_questions(k)) for k in keys)

    fig, ax = plt.subplots(figsize=(len(keys) * col_w * 0.95, n_rows * 0.62 + 1.6))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    import textwrap

    for i, key in enumerate(keys):
        title = "\n".join(textwrap.wrap(codebook.sections[key], 26))
        ax.text(i * col_w - 0.1, 0.85, title, fontsize=10.5, fontweight="bold",
                color=INK, va="bottom", ha="left")
        ax.plot([i * col_w - 0.1, i * col_w + col_w - 0.5], [0.7, 0.7], color=RULE, lw=0.8)

    # Within a section: a tree. Each parent drops one vertical spine; every child
    # gets a short horizontal arrow off it. Each distinct trigger is labelled once,
    # just above the first child it opens.
    cross_notes: dict[str, list[str]] = {}
    labelled: set[tuple[str, str]] = set()
    edges = sorted(graph.edges(data=True), key=lambda e: -pos[e[1]][1])
    noted = {c for p, c, _ in edges   # these carry an "if …" note under their text
             if codebook.by_id[p].section_key != codebook.by_id[c].section_key}
    for parent, child, data in edges:
        (x0, y0), (x1, y1) = pos[parent], pos[child]
        if codebook.by_id[parent].section_key != codebook.by_id[child].section_key:
            cross_notes.setdefault(child, []).append(f"{parent} {data['short']}")
            ax.annotate("", xy=(x1 - 0.08, y1), xytext=(x0 + 0.08, y0),
                        arrowprops=dict(arrowstyle="-|>", color=INK_MUTED, lw=0.8,
                                        linestyle=(0, (4, 3)), shrinkA=3, shrinkB=4,
                                        connectionstyle="arc3,rad=-0.3"))
            continue
        ax.plot([x0, x0], [y0 - 0.12, y1], color=RULE, lw=1.0, zorder=1)
        ax.annotate("", xy=(x1 - 0.09, y1), xytext=(x0, y1),
                    arrowprops=dict(arrowstyle="-|>", color=RULE, lw=1.0,
                                    shrinkA=0, shrinkB=0), zorder=1)
        if (parent, data["label"]) not in labelled:
            labelled.add((parent, data["label"]))
            ax.text(x0 + 0.07, y1 + (0.3 if parent in noted else 0.42), data["label"],
                    fontsize=6.4, color=INK_MUTED,
                    ha="left", va="center", style="italic", zorder=2,
                    bbox=dict(boxstyle="round,pad=0.1", fc="#fcfcfb", ec="none"))

    for q in codebook.questions:
        x, y = pos[q.id]
        dataset = q.level == "dataset"
        ax.scatter([x], [y], s=70, marker="s" if dataset else "o", zorder=3,
                   color=DATASET_COLOR if dataset else PAPER_COLOR,
                   edgecolors="#fcfcfb", linewidths=2)
        text = q.text if len(q.text) <= text_width else q.text[:text_width - 1] + "…"
        ax.text(x + 0.12, y + 0.02, q.id, fontsize=8.2, fontweight="bold", color=INK,
                va="bottom", ha="left", zorder=4)
        ax.text(x + 0.12, y - 0.02, f"{q.display_id} · {text}" if q.display_id else text,
                fontsize=6.6, color=INK_MUTED, va="top", ha="left", zorder=4)
        if q.id in cross_notes:
            joiner = f" {q.condition.op.upper()} " if isinstance(q.condition, BoolOp) else ""
            note = "\n".join(textwrap.wrap("if " + joiner.join(cross_notes[q.id]), 46))
            ax.text(x + 0.12, y - 0.24, note, fontsize=6.0, linespacing=1.1,
                    color=INK_MUTED, style="italic", va="top", ha="left", zorder=4)

    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=PAPER_COLOR, markersize=8, label="paper-level"),
        Line2D([], [], marker="s", ls="", color=DATASET_COLOR, markersize=8, label="dataset-level"),
        Line2D([], [], color=RULE, lw=1, label="condition within a section"),
        Line2D([], [], color=INK_MUTED, lw=1, ls="--", label="condition across sections"),
    ], loc="lower left", bbox_to_anchor=(0, -0.02), ncol=4, frameon=False, fontsize=8.5,
        labelcolor=INK_MUTED)
    ax.set_xlim(-0.3, len(keys) * col_w)
    ax.set_ylim(-(n_rows + 0.3) * row_h, 1.9)
    ax.axis("off")
    fig.tight_layout()
    return fig
