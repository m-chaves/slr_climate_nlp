"""
Full-text assessment tool.

    streamlit run src/full_text_assessment/app.py

One screen per paper: the pre-filled paper info, the PDF (side by side or in an
external viewer), and one tab per codebook section. Only the questions that apply are
shown; the rest are written as NA. Every change is saved immediately to
`data/processed/full_text_assessment/` -- see `store.py` for the file layout.

State model: the answers being edited live in `st.session_state["work"]`, one entry
per (reviewer, paper). Widgets are views on that dict: they are initialised from it and
write back to it in their `on_change` callbacks. Keeping the dict (rather than widget
state) as the source of truth is what lets a question disappear behind the branching
and come back with its answer, and lets you hop between papers and datasets freely.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.full_text_assessment import store, timing
from src.full_text_assessment.audit import ERROR, audit, issues_frame
from src.full_text_assessment.codebook import (
    CODEBOOK_PATH, EXCLUSIVE_LABELS, NA, PENDING, SHOWN, SKIPPED, Codebook, Question,
    diff_snapshots, is_blank, load_codebook,
)
from src.full_text_assessment.prefill import load_paper_info

OTHER = "✎ other (free text)"
VALUE = "enter a value"
STATUS_ICON = {"": "⬜", store.IN_PROGRESS: "🟡", store.COMPLETE: "✅", "needs review": "⚠️"}
NEW_REVIEWER = "➕ new reviewer…"
REVIEWER_FILE = store.DATA_DIR / ".reviewer"

st.set_page_config(page_title="Full-text assessment", page_icon="📄", layout="wide")
ss = st.session_state


# ──────────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────────

def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_resource
def _codebook(xlsx_mtime: float, log_mtime: float) -> Codebook:
    return load_codebook(label_log=store.load_label_log())


def codebook() -> Codebook:
    return _codebook(_mtime(CODEBOOK_PATH), _mtime(store.LABELS_PATH))


@st.cache_data
def _paper_info(mtime: float) -> pd.DataFrame:
    return load_paper_info()


def paper_info() -> pd.DataFrame:
    from src.full_text_assessment.prefill import PAPER_INFO_PATH
    return _paper_info(_mtime(PAPER_INFO_PATH))


def blank_bucket(cb: Codebook, level: str) -> dict:
    return {"answers": {q.id: ([] if q.kind == "multi" else None) for q in cb.level(level)},
            "free": {q.id: q.kind == "free" and not q.labels for q in cb.level(level)}}


def load_work(cb: Codebook, reviewer: str, paper_id: str) -> dict:
    """The editable state for one paper, from disk the first time it is opened."""
    key = (reviewer, paper_id)
    work = ss.setdefault("work", {})
    if key in work:
        return work[key]
    paper = store.get_paper(cb, paper_id, reviewer)
    stored_ds = store.get_datasets(cb, paper_id, reviewer)
    datasets = {d["idx"]: {"label": d["label"], "answers": d["answers"], "free": d["free"]}
                for d in stored_ds}
    entry = {"answers": paper["answers"], "free": paper["free"], "notes": paper["notes"],
             "status": paper["status"], "datasets": datasets,
             "current_ds": min(datasets) if datasets else None, "saved_ds": {},
             # a paper never saved is, by definition, reviewed against today's codebook
             "codebook_version": paper["codebook_version"] or cb.version,
             "retired": paper["retired"],
             "retired_ds": {d["label"]: d["retired"] for d in stored_ds if d["retired"]}}
    entry["saved"] = paper_signature(cb, entry)
    for idx in datasets:
        entry["saved_ds"][idx] = dataset_signature(cb, entry, idx)
    work[key] = entry
    return entry


# ──────────────────────────────────────────────────────────────────────────────
# Saving (autosave at the end of every run)
# ──────────────────────────────────────────────────────────────────────────────

def paper_signature(cb: Codebook, work: dict) -> str:
    stored = store.finalize(cb, work["answers"], "paper")
    return json.dumps([stored, work["free"], work["notes"], work["status"],
                       work["codebook_version"]], sort_keys=True, default=str)


def dataset_signature(cb: Codebook, work: dict, idx: int) -> str:
    ds = work["datasets"][idx]
    context = cb.visibility(work["answers"], cb.level("paper"))
    stored = store.finalize(cb, {**work["answers"], **ds["answers"]}, "dataset", context)
    return json.dumps([stored, ds["free"], ds["label"]], sort_keys=True, default=str)


def autosave(cb: Codebook, reviewer: str, paper_id: str, work: dict) -> None:
    problems = completion_problems(cb, work)
    if work["status"] == store.COMPLETE and problems:
        work["status"] = store.IN_PROGRESS
    sig = paper_signature(cb, work)
    if sig != work["saved"]:
        if not work["status"]:
            work["status"] = store.IN_PROGRESS
            sig = paper_signature(cb, work)
        store.save_paper(cb, paper_id, reviewer, work["answers"], work["free"],
                         status=work["status"], notes=work["notes"],
                         codebook_version=work["codebook_version"])
        work["saved"] = sig
    if cb.datasets_applicable(work["answers"]) is not True:
        return
    for idx, ds in work["datasets"].items():
        sig = dataset_signature(cb, work, idx)
        if sig != work["saved_ds"].get(idx):
            store.save_dataset(cb, paper_id, reviewer, idx, ds["label"], ds["answers"],
                               ds["free"], work["answers"])
            work["saved_ds"][idx] = sig


def year_problems(cb: Codebook, answers: dict, level: str) -> list[str]:
    years = [q for q in cb.level(level) if q.kind == "year"]
    if len(years) != 2:
        return []
    start, end = (answers.get(q.id) for q in years)
    if isinstance(start, str) and isinstance(end, str) and start.isdigit() and end.isdigit():
        if int(end) < int(start):
            return [f"{years[1].label_ids} ({end}) is before {years[0].label_ids} ({start})"]
    return []


def completion_problems(cb: Codebook, work: dict) -> list[str]:
    """Everything standing between this paper and 'complete'."""
    problems = []
    if work["codebook_version"] != cb.version:
        problems.append("the codebook changed since your last review of this paper: "
                        "check the changes and confirm them (banner above the tabs)")
    problems += [f"{cb.by_id[qid].label_ids} unanswered"
                 for qid in store.missing(cb, work["answers"], "paper")]
    applicable = cb.datasets_applicable(work["answers"])
    context = cb.visibility(work["answers"], cb.level("paper"))
    if applicable is True:
        if not work["datasets"]:
            problems.append("no dataset added yet")
        for idx, ds in sorted(work["datasets"].items()):
            merged = {**work["answers"], **ds["answers"]}
            gaps = store.missing(cb, merged, "dataset", context)
            if gaps:
                names = ", ".join(cb.by_id[g].label_ids for g in gaps)
                problems.append(f"{ds['label']}: {names} unanswered")
            problems += [f"{ds['label']}: {p}" for p in year_problems(cb, ds["answers"], "dataset")]
    elif applicable is False and work["datasets"]:
        problems.append(f"{len(work['datasets'])} dataset(s) recorded, but the paper "
                        "uses none: delete them in the Data tab")
    return problems


# ──────────────────────────────────────────────────────────────────────────────
# Widget callbacks: widget state -> work dict
# ──────────────────────────────────────────────────────────────────────────────

def _log_new_labels(cb: Codebook, q: Question, values: list[str]) -> None:
    """Log labels typed for the first time. Old names of renamed labels never count."""
    pool = cb.pool_owner(q).id
    known = {label.value.casefold() for label in cb.labels(q)}
    for value in values:
        if value and value.casefold() not in known and not cb.registry.renamed_away(pool, value):
            store.log_label(pool, value, ss["reviewer"], ss["paper_id"])


def _current_names(cb: Codebook, q: Question, values: list[str]) -> list[str]:
    """Map old names of renamed labels to their current name, and say so."""
    out: list[str] = []
    for value in values:
        current = cb.resolve(q, value)
        if current != value:
            ss.setdefault("label_notices", []).append(
                f"{q.label_ids}: “{value}” was renamed to “{current}”; saved as “{current}”.")
        if current not in out:
            out.append(current)
    return out


def on_single(cb: Codebook, q: Question, bucket: dict, key: str) -> None:
    value = ss.get(key)
    if value is not None and q.updatable:
        current = _current_names(cb, q, [value])[0]
        if current != value:
            ss[key] = value = current
        _log_new_labels(cb, q, [value])
    bucket["answers"][q.id] = value


def on_multi(cb: Codebook, q: Question, bucket: dict, key: str) -> None:
    raw = [str(v).strip() for v in ss.get(key) or [] if str(v).strip()]
    selected = _current_names(cb, q, raw) if q.updatable else raw
    if selected != raw:
        ss[key] = selected
    previous = bucket["answers"].get(q.id) or []
    previous = previous if isinstance(previous, list) else []
    exclusive = [v for v in selected if v.casefold() in EXCLUSIVE_LABELS]
    if exclusive and len(selected) > 1:
        newest = [v for v in selected if v not in previous]
        if newest and newest[-1].casefold() in EXCLUSIVE_LABELS:
            selected = [newest[-1]]          # just picked "not reported": it wins
        else:
            selected = [v for v in selected if v.casefold() not in EXCLUSIVE_LABELS]
        ss[key] = selected
    if q.updatable:
        _log_new_labels(cb, q, selected)
    bucket["answers"][q.id] = selected


def on_free_choice(q: Question, bucket: dict, key: str, text_key: str) -> None:
    choice = ss.get(key)
    if choice == OTHER:
        bucket["free"][q.id] = True
        bucket["answers"][q.id] = ss.get(text_key) or None
    else:
        bucket["free"][q.id] = False
        bucket["answers"][q.id] = choice


def on_free_text(q: Question, bucket: dict, key: str) -> None:
    bucket["free"][q.id] = True
    bucket["answers"][q.id] = (ss.get(key) or "").strip() or None


def on_number_mode(q: Question, bucket: dict, key: str, num_key: str) -> None:
    mode = ss.get(key)
    if mode == VALUE:
        number = ss.get(num_key)
        bucket["answers"][q.id] = str(int(number)) if number is not None else None
    else:
        bucket["answers"][q.id] = mode


def on_number(q: Question, bucket: dict, key: str) -> None:
    number = ss.get(key)
    bucket["answers"][q.id] = str(int(number)) if number is not None else None


# ──────────────────────────────────────────────────────────────────────────────
# Rendering questions
# ──────────────────────────────────────────────────────────────────────────────

def _init(key: str, **default) -> dict:
    """A widget's default, only on first render: afterwards its own state wins."""
    return {} if key in ss else default


def _current(bucket: dict, q: Question):
    value = bucket["answers"].get(q.id)
    if value == NA:                          # stored NA, but the question applies now
        return [] if q.kind == "multi" else None
    return value


def render_question(cb: Codebook, q: Question, bucket: dict, prefix: str) -> None:
    key = f"{prefix}|{q.id}"
    labels = cb.labels(q)
    values = [label.value for label in labels]
    current = _current(bucket, q)
    shared = f" · labels shared with {cb.pool_owner(q).id}" if q.pool != q.id else ""
    if q.pool == q.id and any(o.pool == q.id for o in cb.questions if o is not q):
        shared = " · labels shared with " + ", ".join(o.id for o in cb.questions
                                                        if o.pool == q.id and o is not q)
    badge = ss.get("changes", {}).get(q.id)
    st.markdown(f"**{q.display_id or q.id}** <small>`{q.id}`</small> &nbsp; {q.text}"
                + (f" &nbsp; :orange-badge[{badge}]" if badge else ""), unsafe_allow_html=True)
    if q.updatable or shared:
        st.caption(("updatable: type to add a label" if q.updatable else "") + shared)

    if q.kind == "single":
        options = values + ([current] if current and current not in values else [])
        index = options.index(current) if current in options else None
        if q.updatable:
            st.selectbox(q.id, options, key=key, format_func=q.label_text,
                         **_init(key, index=index),
                         accept_new_options=True, placeholder="choose, or type a new label",
                         label_visibility="collapsed", on_change=on_single,
                         args=(cb, q, bucket, key))
        else:
            horizontal = len(options) <= 4 and all(len(q.label_text(o)) < 40 for o in options)
            st.radio(q.id, options, key=key, format_func=q.label_text,
                     **_init(key, index=index),
                     horizontal=horizontal, label_visibility="collapsed",
                     on_change=on_single, args=(cb, q, bucket, key))

    elif q.kind == "multi":
        current = current if isinstance(current, list) else []
        options = values + [c for c in current if c not in values]
        st.multiselect(q.id, options, key=key, **_init(key, default=current),
                       accept_new_options=q.updatable, label_visibility="collapsed",
                       placeholder="choose" + (", or type a new label" if q.updatable else ""),
                       on_change=on_multi, args=(cb, q, bucket, key))

    elif q.kind == "free":
        text_key = key + "|text"
        if values:
            is_free = bool(bucket["free"].get(q.id)) and bool(current)
            options = values + [OTHER]
            if is_free:
                index = len(values)
            elif current in values:
                index = values.index(current)
            else:
                index = None
            choice = st.radio(q.id, options, key=key, horizontal=True,
                              **_init(key, index=index),
                              label_visibility="collapsed", on_change=on_free_choice,
                              args=(q, bucket, key, text_key))
            if choice == OTHER:
                st.text_area(q.id, key=text_key,
                             **_init(text_key, value=current if is_free else ""),
                             label_visibility="collapsed", placeholder="type the answer",
                             on_change=on_free_text, args=(q, bucket, text_key))
        else:
            st.text_area(q.id, key=text_key, **_init(text_key, value=current or ""),
                         label_visibility="collapsed", placeholder="type the answer",
                         on_change=on_free_text, args=(q, bucket, text_key))

    else:                                    # integer / year
        num_key = key + "|num"
        is_value = bool(current) and current not in values
        options = [VALUE] + values
        index = 0 if is_value else (options.index(current) if current in options else None)
        mode = st.radio(q.id, options, key=key, horizontal=True, **_init(key, index=index),
                        label_visibility="collapsed", on_change=on_number_mode,
                        args=(q, bucket, key, num_key))
        if mode == VALUE:
            bounds = {"min_value": 1900, "max_value": 2100} if q.kind == "year" else {"min_value": 0}
            number = int(current) if is_value and str(current).isdigit() else None
            st.number_input(q.id, step=1, key=num_key, **_init(num_key, value=number),
                            label_visibility="collapsed",
                            placeholder="YYYY" if q.kind == "year" else "integer",
                            on_change=on_number, args=(q, bucket, num_key), **bounds)


def render_questions(cb: Codebook, questions: list[Question], bucket: dict,
                     state: dict[str, str], prefix: str) -> None:
    skipped, lost = [], []
    for q in questions:
        if state[q.id] == SHOWN:
            with st.container(border=True):
                render_question(cb, q, bucket, prefix)
        elif state[q.id] == PENDING:
            parents = ", ".join(cb.by_id[p].label_ids for p in q.parents if p in cb.by_id)
            st.caption(f"⏳ **{q.label_ids}** {q.text[:90]}{'…' if len(q.text) > 90 else ''} "
                       f"— appears once {parents} is answered")
        else:
            skipped.append(q.id)
            if not is_blank(bucket["answers"].get(q.id)) and bucket["answers"].get(q.id) != NA:
                lost.append(q.id)
    if lost:
        lost = [cb.by_id[qid].label_ids for qid in lost]
        st.warning(f"{', '.join(lost)} had answers and are now **NA** because of the "
                   "branching. Undo the change in this session to get them back.")
    if skipped:
        names = ", ".join(cb.by_id[qid].label_ids for qid in skipped)
        st.caption(f"{len(skipped)} skipped → NA: {names}")


# ──────────────────────────────────────────────────────────────────────────────
# Data section
# ──────────────────────────────────────────────────────────────────────────────

def add_dataset(cb: Codebook, work: dict, sel_key: str | None = None) -> None:
    idx = max(work["datasets"], default=0) + 1
    work["datasets"][idx] = {"label": f"dataset {idx}", **blank_bucket(cb, "dataset")}
    work["saved_ds"][idx] = dataset_signature(cb, work, idx)   # nothing to save yet
    work["current_ds"] = idx
    if sel_key:
        ss[sel_key] = idx


def delete_dataset(work: dict, idx: int | None) -> None:
    targets = list(work["datasets"]) if idx is None else [idx]
    store.delete_datasets(ss["paper_id"], ss["reviewer"], idx)
    for i in targets:
        work["datasets"].pop(i, None)
        work["saved_ds"].pop(i, None)
    work["current_ds"] = min(work["datasets"]) if work["datasets"] else None


def render_data_section(cb: Codebook, key: str, work: dict, prefix: str) -> None:
    paper_qs = cb.section_questions(key, "paper")
    paper_state = cb.visibility(work["answers"], cb.level("paper"))
    render_questions(cb, paper_qs, work, paper_state, prefix + "|p")

    applicable = cb.datasets_applicable(work["answers"])
    if applicable is None:
        st.info("Answer the question above to describe the paper's datasets.")
        return
    if applicable is False:
        if work["datasets"]:
            st.warning(f"This paper has {len(work['datasets'])} dataset(s) recorded, but "
                       "the answer above says it uses none.")
            st.button("Delete the recorded datasets", on_click=delete_dataset,
                      args=(work, None), type="primary")
        else:
            st.caption("No datasets to describe for this paper.")
        return

    if not work["datasets"]:
        add_dataset(cb, work)
    if work["current_ds"] not in work["datasets"]:
        work["current_ds"] = min(work["datasets"])

    context = paper_state

    def ds_label(idx: int) -> str:
        ds = work["datasets"][idx]
        gaps = store.missing(cb, {**work["answers"], **ds["answers"]}, "dataset", context)
        return f"{ds['label']}" + (f" ({len(gaps)} left)" if gaps else " ✓")

    # The selector's state is the source of truth for the dataset being edited.
    sel_key = f"{prefix}|dsel"
    if ss.get(sel_key) in work["datasets"]:
        work["current_ds"] = ss[sel_key]
    else:
        ss[sel_key] = work["current_ds"]

    left, right = st.columns([4, 1])
    with left:
        st.segmented_control("Dataset", list(work["datasets"]), key=sel_key,
                             format_func=ds_label, required=True)
    with right:
        st.button("➕ Add dataset", on_click=add_dataset, args=(cb, work, sel_key),
                  use_container_width=True)

    idx = work["current_ds"]
    ds = work["datasets"][idx]
    with st.expander(f"Name or delete “{ds['label']}”"):
        name_key = f"{prefix}|d{idx}|label"

        def on_name() -> None:
            ds["label"] = (ss.get(name_key) or "").strip() or f"dataset {idx}"

        st.text_input("Short name (used for the tabs, and to match datasets across "
                      "reviewers)", value=ds["label"], key=name_key, on_change=on_name)
        st.button(f"Delete “{ds['label']}”", on_click=delete_dataset, args=(work, idx))

    merged = {**work["answers"], **ds["answers"]}
    ds_state = cb.visibility(merged, cb.level("dataset"), context)
    render_questions(cb, cb.section_questions(key, "dataset"), ds, ds_state,
                     f"{prefix}|d{idx}")
    for problem in year_problems(cb, ds["answers"], "dataset"):
        st.error(problem)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

def remembered_reviewer() -> str:
    """From the URL, else this machine's last choice; matched to known names up to case."""
    if "reviewer" in st.query_params:
        name = st.query_params["reviewer"]
    else:
        try:
            name = REVIEWER_FILE.read_text(encoding="utf-8").strip()
        except OSError:
            name = ""
    return store.canonical_reviewer(name) if name else ""


def set_reviewer(name: str) -> None:
    ss["reviewer"] = name
    if name:
        store.DATA_DIR.mkdir(parents=True, exist_ok=True)
        REVIEWER_FILE.write_text(name, encoding="utf-8")


def on_reviewer_select() -> None:
    choice = ss.get("reviewer_select")
    if choice and choice != NEW_REVIEWER:
        set_reviewer(choice)


def on_new_reviewer() -> None:
    name = store.canonical_reviewer(ss.get("reviewer_new") or "")
    if len(name) >= 2:
        set_reviewer(name)
        ss["reviewer_select"] = name
        ss["reviewer_new"] = ""


def go_to(paper_id: str) -> None:
    ss["paper_id"] = paper_id
    ss["paper_select"] = paper_id


def statuses(cb: Codebook, reviewer: str) -> dict[str, str]:
    """Stored status per paper, or "needs review" when saved under an older codebook."""
    papers = store.load_papers()
    mine = papers[papers["reviewer"] == reviewer]
    return {row["paper_id"]: ("needs review" if row["codebook_version"] != cb.version
                              else row["status"])
            for _, row in mine.iterrows()}


def reset_form_state() -> None:
    """
    Forget every answer widget's remembered value, so the form is rebuilt from the
    answers just loaded. Only the answer form (keys starting "w|"): reviewer, paper,
    filters and timer are kept.
    """
    for key in [k for k in ss.keys() if str(k).startswith("w|")]:
        del ss[key]


def rename_label(cb: Codebook, q: Question, old: str, new: str) -> None:
    try:
        n = store.rename_label(cb, cb.pool_owner(q).id, old, new, ss["reviewer"],
                               ss.get("paper_id", ""))
    except ValueError as exc:
        ss["rename_error"] = str(exc)
        return
    ss["work"] = {}                          # everything is saved; reload with new names
    reset_form_state()
    ss["rn_new"] = ""
    ss["rename_done"] = f"Renamed “{old}” → “{new}” in {n} stored answer(s)."


def reload_from_disk() -> None:
    """Re-read answers, codebook, labels and paper info from the files."""
    ss["work"] = {}
    reset_form_state()
    _codebook.clear()
    _paper_info.clear()
    ss["reload_done"] = True


def sidebar(cb: Codebook, info: pd.DataFrame) -> None:
    with st.sidebar:
        st.header("Full-text assessment")
        if "reviewer" not in ss:
            ss["reviewer"] = remembered_reviewer()
        known = store.reviewers()
        if ss["reviewer"] and ss["reviewer"] not in known:
            known = [ss["reviewer"]] + known
        if ss.get("reviewer_select") not in known + [NEW_REVIEWER]:
            ss["reviewer_select"] = ss["reviewer"] if ss["reviewer"] else NEW_REVIEWER
        st.selectbox("Reviewer", known + [NEW_REVIEWER], key="reviewer_select",
                     on_change=on_reviewer_select)
        if ss["reviewer_select"] == NEW_REVIEWER:
            st.text_input("Your full name", key="reviewer_new", on_change=on_new_reviewer,
                          placeholder="first and last name")
            st.caption("Use the same full name every time: it is how your answers and "
                       "time are recorded.")
            ss["reviewer"] = ""              # nobody is recorded until a name is entered
        reviewer = ss["reviewer"]
        if not reviewer:
            return

        status = statuses(cb, reviewer)
        done = sum(1 for s in status.values() if s == store.COMPLETE)
        st.progress(done / max(len(info), 1),
                    text=f"{done} / {len(info)} complete · "
                         f"{sum(1 for s in status.values() if s == store.IN_PROGRESS)} in progress")

        with st.expander("Filter papers", expanded=False):
            wanted_status = st.multiselect(
                "Status", ["not started", "in progress", "complete", "needs review"],
                key="f_status")
            areas = sorted(a for a in info["research_area"].unique() if a)
            wanted_area = st.multiselect("Research area (screening)", areas, key="f_area")
            search = st.text_input("Title contains", key="f_search").strip().lower()

        def keep(row) -> bool:
            s = status.get(row["paper_id"], "") or "not started"
            return ((not wanted_status or s in wanted_status)
                    and (not wanted_area or row["research_area"] in wanted_area)
                    and (not search or search in row["title"].lower()))

        ids = [r["paper_id"] for _, r in info.iterrows() if keep(r)]
        current = ss.get("paper_id")
        if current not in set(info["paper_id"]):
            current = ids[0] if ids else info["paper_id"].iloc[0]
        if current not in ids:
            ids = [current] + ids
        titles = dict(zip(info["paper_id"], info["title"]))
        years = dict(zip(info["paper_id"], info["year"]))

        def fmt(pid: str) -> str:
            return f"{STATUS_ICON.get(status.get(pid, ''), '⬜')} {years[pid]} · {titles[pid][:80]}"

        if ss.get("paper_select") not in ids:
            ss["paper_select"] = current
        st.selectbox(f"Paper ({len(ids)} shown)", ids, key="paper_select", format_func=fmt,
                     on_change=lambda: ss.__setitem__("paper_id", ss["paper_select"]))
        ss["paper_id"] = ss["paper_select"]

        position = ids.index(ss["paper_id"])
        c1, c2, c3 = st.columns(3)
        c1.button("◀ Prev", on_click=go_to, args=(ids[position - 1],),
                  disabled=position == 0, use_container_width=True)
        c2.button("Next ▶", on_click=go_to, args=(ids[min(position + 1, len(ids) - 1)],),
                  disabled=position == len(ids) - 1, use_container_width=True)
        unfinished = [p for p in ids[position + 1:] + ids[:position]
                      if status.get(p) != store.COMPLETE]
        c3.button("Next ⬜", help="next paper that is not complete",
                  on_click=go_to, args=(unfinished[0] if unfinished else ss["paper_id"],),
                  disabled=not unfinished, use_container_width=True)

        st.toggle("Show the PDF side by side", key="side_by_side")

        with st.expander("Rename a label"):
            st.caption("A rename rewrites every stored answer, for every paper and "
                       "reviewer (and in both questions of a shared label pool). "
                       "The new name must not already exist. Typed free-text "
                       "answers are never changed.")
            if msg := ss.pop("rename_done", None):
                st.success(msg)
            if msg := ss.pop("rename_error", None):
                st.error(msg)
            updatable = [q.id for q in cb.questions if q.updatable and q.pool == q.id]
            qid = st.selectbox("Question", updatable, key="rn_q",
                               format_func=lambda i: f"{cb.by_id[i].label_ids} · "
                                                     f"{cb.by_id[i].text[:45]}")
            if qid is not None:
                q = cb.by_id[qid]
                old = st.selectbox("Label", [l.value for l in cb.labels(q) if l.code is None],
                                   key="rn_old")
                new = st.text_input("New name", key="rn_new").strip()
                problem = store.rename_error(cb, q.id, old, new) if old and new else None
                if problem:
                    st.error(problem)
                st.button("Rename", disabled=not (old and new) or bool(problem),
                          on_click=rename_label, args=(cb, q, old, new))

        st.button("Reload from disk", on_click=reload_from_disk,
                  help="Re-read answers, codebook and labels from the files, e.g. after "
                       "pulling another reviewer's work or editing the xlsx. Nothing is "
                       "lost: every change is already saved.")
        if ss.pop("reload_done", False):
            st.caption("Reloaded from disk.")


# ──────────────────────────────────────────────────────────────────────────────
# Paper header
# ──────────────────────────────────────────────────────────────────────────────

ID_LINKS = {
    "doi": "https://doi.org/{}",
    "arxiv_id": "https://arxiv.org/abs/{}",
    "acl_id": "https://aclanthology.org/{}/",
    "scopus_eid": "https://www.scopus.com/record/display.uri?eid={}&origin=resultslist",
    "wos_id": "https://www.webofscience.com/wos/woscc/full-record/{}",
}


def open_external(path: Path) -> str | None:
    """Open a file in the OS viewer; returns an error message, or None."""
    try:
        if shutil.which("wslpath") and shutil.which("explorer.exe"):
            win = subprocess.run(["wslpath", "-w", str(path)], capture_output=True,
                                 text=True, check=True).stdout.strip()
            subprocess.Popen(["explorer.exe", win])
        elif shutil.which("xdg-open"):
            subprocess.Popen(["xdg-open", str(path)])
        elif shutil.which("open"):
            subprocess.Popen(["open", str(path)])
        else:
            return "no viewer found (tried explorer.exe, xdg-open, open)"
    except (OSError, subprocess.CalledProcessError) as exc:
        return str(exc)
    return None


PRESENCE = {"Misinformation": ("MIS.is_present", "misinformation"),
            "Polarization": ("POL.is_present", "polarization")}


def consistency_warnings(cb: Codebook, row: pd.Series, answers: dict) -> list[str]:
    area = row["research_area"]
    out = []
    for construct, (qid, name) in PRESENCE.items():
        if area in (construct, "Both") and qid in cb.by_id and answers.get(qid) == "no":
            out.append(f"Screening assigned **{area}**, but {cb.by_id[qid].label_ids} "
                       f"says {name} is not present.")
    return out


# ── timer ────────────────────────────────────────────────────────────────────

@st.cache_resource
def _last_tick_by_reviewer() -> dict[str, float]:
    """Shared by every browser tab: two open tabs must not count the same time twice."""
    return {}


def credit_time(now: float) -> None:
    """Credit the paper being timed with the time since the last tick."""
    timer = ss.get("timer")
    if not timer:
        return
    shared = _last_tick_by_reviewer()
    start = max(timer["last_tick"], shared.get(timer["reviewer"], 0.0))
    seconds = timing.accrue(start, now, ss.get("last_interaction", now),
                            ss.get("timer_paused", False))
    if seconds > 0:
        store.add_time(timer["reviewer"], timer["paper"], seconds)
    shared[timer["reviewer"]] = max(shared.get(timer["reviewer"], 0.0), now)
    timer["last_tick"] = now


def toggle_pause() -> None:
    now = time.time()
    credit_time(now)
    if ss.get("timer_paused"):
        ss["timer_paused"] = False
        ss["last_interaction"] = now
    else:
        ss["timer_paused"] = True


def sync_timer(reviewer: str, paper_id: str, work: dict, now: float) -> None:
    """Time the open paper unless it is marked complete; reopening it starts the clock."""
    if work["status"] == store.COMPLETE:
        ss["timer"] = None
    elif not ss.get("timer"):
        ss["timer"] = {"reviewer": reviewer, "paper": paper_id, "last_tick": now}


@st.fragment(run_every=timing.TICK)
def timer_widget(reviewer: str, paper_id: str, complete: bool) -> None:
    now = time.time()
    credit_time(now)
    total = store.seconds_on(reviewer, paper_id)
    if complete:
        state = "⏹ stopped: paper marked complete (reopen it to count again)"
    elif ss.get("timer_paused"):
        state = "⏸ paused (any interaction resumes)"
    elif timing.is_idle(now, ss.get("last_interaction", now)):
        state = (f"💤 stopped: no interaction for {timing.IDLE_LIMIT // 60} min "
                 "(any interaction resumes)")
    else:
        state = "counting"
    left, right = st.columns([4, 1])
    left.caption(f"⏱ **{timing.format_duration(total)}** on this paper · {state}")
    if not complete:
        right.button("▶ Resume" if ss.get("timer_paused") else "⏸ Pause", key="timer_toggle",
                     on_click=toggle_pause, use_container_width=True)


# ── codebook changes ─────────────────────────────────────────────────────────

def question_changes(cb: Codebook, work: dict) -> dict[str, str]:
    """fixID -> badge text for questions new or changed since the paper was reviewed."""
    if work["codebook_version"] == cb.version:
        return {}
    old = store.snapshot_of(work["codebook_version"])
    if old is None:
        return {}
    diff = diff_snapshots(old, cb.snapshot())
    badges = {qid: "🆕 new question" for qid in diff.added}
    badges.update({qid: "✏️ changed: " + ", ".join(what) for qid, what in diff.changed.items()})
    return badges


def codebook_changes_banner(cb: Codebook, work: dict) -> None:
    retired = {**work["retired"], **{f"{label} · {k}": v for label, d in work["retired_ds"].items()
                                     for k, v in d.items()}}
    if work["codebook_version"] == cb.version and not retired:
        return

    def confirm() -> None:
        work["codebook_version"] = cb.version

    with st.container(border=True):
        if work["codebook_version"] != cb.version:
            st.warning("⚠️ **The codebook changed since you last reviewed this paper.** "
                       "Check the questions marked below, answer any new ones, then confirm.")
            old = store.snapshot_of(work["codebook_version"])
            if old is None:
                st.markdown("The earlier codebook version is not on record, so the changes "
                            "can't be listed: please look over all answers.")
            else:
                diff = diff_snapshots(old, cb.snapshot())
                lines = [f"- 🆕 **{cb.by_id[q].label_ids}** {cb.by_id[q].text[:90]}"
                         for q in diff.added]
                lines += [f"- ✏️ **{cb.by_id[q].label_ids}** changed: {', '.join(what)}"
                          for q, what in diff.changed.items()]
                lines += [f"- 🗑 **{q}** ({old[q].get('display_id', '')}) removed: "
                          f"{old[q].get('text', '')[:80]}" for q in diff.removed]
                st.markdown("\n".join(lines) or "No question-level differences.")
            st.button("✅ I've reviewed the codebook changes", on_click=confirm)
        if retired:
            with st.expander(f"Retired questions: {len(retired)} answer(s) kept from "
                             "questions no longer in the codebook"):
                for qid, value in retired.items():
                    st.markdown(f"- **{qid}**: {value}")


def paper_header(row: pd.Series, work: dict, cb: Codebook) -> None:
    st.subheader(row["title"])
    ids = " · ".join(f"[{c}: {row[c]}]({ID_LINKS[c].format(row[c])})"
                     for c in ID_LINKS if row[c])
    st.markdown(f"**{row['year']}** · {row['published_in']}  \n"
                f"type: {row['paper_type']} · research area (screening): "
                f"**{row['research_area'] or '—'}**  \n{ids}  \n"
                f"`paper_id` {row['paper_id']} · `pdf` {row['pdf_path'] or 'not available'}")

    others = store.load_papers()
    others = others[(others["paper_id"] == row["paper_id"])
                    & (others["reviewer"] != ss["reviewer"])]
    if not others.empty:
        st.caption("Also assessed by: " + ", ".join(
            f"{r['reviewer']} ({r['status']})" for _, r in others.iterrows()))

    c1, c2, c3 = st.columns([1, 1, 3])
    pdf = PROJECT_ROOT / row["pdf_path"] if row["pdf_path"] else None
    if c1.button("📄 Open PDF", disabled=pdf is None or not pdf.exists(),
                 use_container_width=True):
        if error := open_external(pdf):
            st.error(f"Could not open the PDF: {error}")

    problems = completion_problems(cb, work)

    def mark(status: str) -> None:
        work["status"] = status

    if work["status"] == store.COMPLETE:
        c2.button("↩ Reopen", on_click=mark, args=(store.IN_PROGRESS,),
                  use_container_width=True)
        c3.success("Complete")
    else:
        c2.button("✅ Mark complete", on_click=mark, args=(store.COMPLETE,),
                  disabled=bool(problems), use_container_width=True)
        with c3:
            if problems:
                with st.expander(f"{len(problems)} thing(s) left before this paper is complete"):
                    st.markdown("\n".join(f"- {p}" for p in problems))
            else:
                st.info("Every question is answered: you can mark it complete.")

    timer_widget(ss["reviewer"], row["paper_id"], work["status"] == store.COMPLETE)
    codebook_changes_banner(cb, work)
    for warning in consistency_warnings(cb, row, work["answers"]):
        st.warning(warning)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cb = codebook()
    errors = [i for i in audit(cb) if i.severity == ERROR]
    if errors:
        st.error("The codebook has errors; fix the xlsx and reload. "
                 f"({CODEBOOK_PATH.relative_to(PROJECT_ROOT)})")
        st.dataframe(issues_frame(errors), hide_index=True)
        st.stop()

    if ss.get("recorded_version") != cb.version:
        store.record_version(cb)
        ss["recorded_version"] = cb.version
    for notice in ss.pop("label_notices", []):
        st.info(notice)

    info = paper_info()
    sidebar(cb, info)
    reviewer = ss.get("reviewer", "")
    if not reviewer:
        st.info("Choose your name in the sidebar, or add yourself as a new reviewer "
                "with your full name, to start.")
        st.stop()
    st.query_params["reviewer"] = reviewer
    paper_id = ss["paper_id"]
    st.query_params["paper"] = paper_id

    # Every full rerun is a user action: credit the paper that was open up to now,
    # then restart the clock (and end any pause) on the paper open now, unless it is
    # complete. Marking complete / reopening is a full rerun, so it stops / restarts here.
    now = time.time()
    credit_time(now)
    ss["last_interaction"] = now
    ss["timer_paused"] = False
    ss["timer"] = None

    row = info[info["paper_id"] == paper_id].iloc[0]
    work = load_work(cb, reviewer, paper_id)
    sync_timer(reviewer, paper_id, work, now)
    ss["changes"] = question_changes(cb, work)
    prefix = f"w|{reviewer}|{paper_id}"

    pdf = PROJECT_ROOT / row["pdf_path"] if row["pdf_path"] else None
    if ss.get("side_by_side") and pdf is not None and pdf.exists():
        pdf_col, form_col = st.columns([1, 1], gap="medium")
        with pdf_col:
            try:
                st.pdf(str(pdf), height=1100)
            except Exception as exc:  # the viewer component is optional
                st.error(f"The embedded viewer is unavailable ({exc}). "
                         "Use “Open PDF” instead.")
    else:
        form_col = st.container()

    with form_col:
        paper_header(row, work, cb)
        keys = [k for k in cb.sections if cb.section_questions(k)]
        tabs = st.tabs([cb.sections[k] for k in keys], key="section_tabs")
        state = cb.visibility(work["answers"], cb.level("paper"))
        for key, tab in zip(keys, tabs):
            with tab:
                if cb.section_questions(key, "dataset"):
                    render_data_section(cb, key, work, prefix)
                else:
                    render_questions(cb, cb.section_questions(key), work, state, prefix + "|p")
        notes_key = f"{prefix}|notes"
        st.text_area("Reviewer notes (free, not part of the codebook)", value=work["notes"],
                     key=notes_key,
                     on_change=lambda: work.__setitem__("notes", ss.get(notes_key) or ""))

    autosave(cb, reviewer, paper_id, work)
    sync_timer(reviewer, paper_id, work, now)  # autosave reopens a complete paper left incomplete


if "paper_id" not in ss and "paper" in st.query_params:
    ss["paper_id"] = st.query_params["paper"]
    ss["paper_select"] = ss["paper_id"]

main()
