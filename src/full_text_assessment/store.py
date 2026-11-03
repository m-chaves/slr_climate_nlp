"""
Reading and writing full-text assessment answers.

Two wide CSVs, one column per question **fixID**, keyed by reviewer:

    papers.csv     one row per (paper_id, reviewer)                -- paper-level questions
    datasets.csv   one row per (paper_id, reviewer, dataset_idx)   -- dataset-level questions

and alongside them:

    labels_added.csv       append-only log of labels added or renamed while reading
    codebook_versions.json every codebook version seen, as {version: {fixID: fingerprint}}
    papers_archive.csv     rows as they were before their first save under a newer
    datasets_archive.csv   codebook version (nothing is lost when questions change)
    time_log.csv           seconds spent, one row per (reviewer, paper_id)

Conventions, so the files stay easy to consume downstream:

  - every cell is a string; files are read with `keep_default_na=False` so the
    literal "NA" (skipped by the branching logic) survives the round trip;
  - multi-select answers are joined with " | ";
  - free-text questions that also offer labels get a companion `<fixID>__free` column,
    "True" when the answer is typed text rather than one of the labels;
  - label values are always written under their *current* name: renames rewrite the
    stored answers, and every save resolves names again;
  - columns of questions removed from the codebook are kept, never dropped;
  - rows are sorted by key, so diffs stay local and git merges rarely conflict;
  - every write goes to a temp file first and then replaces the original.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.full_text_assessment.codebook import (
    LABEL_LOG_COLUMNS, MULTI_SEP, NA, PROJECT_ROOT, SHOWN, SKIPPED, Answer, Codebook,
    Question, is_blank,
)

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "full_text_assessment"
PAPERS_PATH = DATA_DIR / "papers.csv"
DATASETS_PATH = DATA_DIR / "datasets.csv"
LABELS_PATH = DATA_DIR / "labels_added.csv"
VERSIONS_PATH = DATA_DIR / "codebook_versions.json"
PAPERS_ARCHIVE_PATH = DATA_DIR / "papers_archive.csv"
DATASETS_ARCHIVE_PATH = DATA_DIR / "datasets_archive.csv"
TIME_PATH = DATA_DIR / "time_log.csv"

PAPER_KEY = ["paper_id", "reviewer"]
DATASET_KEY = ["paper_id", "reviewer", "dataset_idx"]
# codebook_version: the version the reviewer last confirmed for this paper;
# saved_with: the version in force when the row was last written.
PAPER_META = PAPER_KEY + ["status", "codebook_version", "saved_with", "last_modified", "notes"]
DATASET_META = DATASET_KEY + ["dataset_label", "saved_with", "last_modified"]
TIME_COLUMNS = ["reviewer", "paper_id", "seconds", "first_opened", "last_seen"]

IN_PROGRESS, COMPLETE = "in progress", "complete"
FREE_SUFFIX = "__free"


# ──────────────────────────────────────────────────────────────────────────────
# Files
# ──────────────────────────────────────────────────────────────────────────────

def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def read_csv(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns or [])
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column in columns or []:
        if column not in df.columns:
            df[column] = ""
    return df


def write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write via a temp file in the same directory, then replace: never half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            df.to_csv(handle, index=False)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _write_text_atomic(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(tmp, path)


def has_free_column(question: Question) -> bool:
    return question.kind == "free" and bool(question.labels)


def answer_columns(codebook: Codebook, level: str) -> list[str]:
    columns = []
    for q in codebook.level(level):
        columns.append(q.id)
        if has_free_column(q):
            columns.append(q.id + FREE_SUFFIX)
    return columns


def _schema(codebook: Codebook, level: str, existing: list[str]) -> list[str]:
    """Metadata, then the codebook's columns, then any stale columns we must not drop."""
    meta = PAPER_META if level == "paper" else DATASET_META
    columns = meta + answer_columns(codebook, level)
    return columns + [c for c in existing if c not in columns]


# ──────────────────────────────────────────────────────────────────────────────
# Codebook versions
# ──────────────────────────────────────────────────────────────────────────────

def load_versions() -> dict[str, dict]:
    if not VERSIONS_PATH.exists():
        return {}
    return json.loads(VERSIONS_PATH.read_text(encoding="utf-8"))


def record_version(codebook: Codebook) -> str:
    """Add the current codebook to `codebook_versions.json` if it is new. Returns it."""
    version = codebook.version
    versions = load_versions()
    if version not in versions:
        versions[version] = {"recorded_at": now(), "questions": codebook.snapshot()}
        _write_text_atomic(json.dumps(versions, indent=1, ensure_ascii=False, sort_keys=True),
                           VERSIONS_PATH)
    return version


def snapshot_of(version: str) -> dict[str, dict] | None:
    entry = load_versions().get(version)
    return entry["questions"] if entry else None


# ──────────────────────────────────────────────────────────────────────────────
# Encoding answers
# ──────────────────────────────────────────────────────────────────────────────

def encode(question: Question, answer: Answer) -> str:
    if answer is None:
        return ""
    if isinstance(answer, list):
        return MULTI_SEP.join(str(a).strip() for a in answer if str(a).strip())
    return str(answer).strip()


def resolve_answer(codebook: Codebook, question: Question, answer: Answer,
                   is_free: bool = False) -> Answer:
    """An answer with every label under its current name (renames applied)."""
    if answer is None or answer == NA or is_free:
        return answer
    if isinstance(answer, list):
        out: list[str] = []
        for part in answer:
            part = codebook.resolve(question, str(part).strip())
            if part and part not in out:
                out.append(part)
        return out
    return codebook.resolve(question, str(answer))


def decode(codebook: Codebook, question: Question, cell: str) -> Answer:
    """A CSV cell back to an in-memory answer, with label renames applied."""
    cell = "" if cell is None else str(cell)
    if cell == "":
        return [] if question.kind == "multi" else None
    if cell == NA:
        return NA
    if question.kind == "multi":
        return resolve_answer(codebook, question, cell.split(MULTI_SEP))
    if question.kind == "single" and question.updatable:
        return codebook.resolve(question, cell)
    return cell


def _row_to_answers(codebook: Codebook, level: str, row: pd.Series) -> tuple[dict, dict]:
    answers, free = {}, {}
    for q in codebook.level(level):
        answers[q.id] = decode(codebook, q, row.get(q.id, ""))
        if has_free_column(q):
            free[q.id] = str(row.get(q.id + FREE_SUFFIX, "")) == "True"
        elif q.kind == "free":
            free[q.id] = True
    return answers, free


def _retired(codebook: Codebook, level: str, row: pd.Series) -> dict[str, str]:
    """Non-empty answers stored under fixIDs that are no longer in the codebook."""
    meta = set(PAPER_META if level == "paper" else DATASET_META)
    current = set(answer_columns(codebook, level))
    return {c: str(v) for c, v in row.items()
            if c not in meta and c not in current and not c.endswith(FREE_SUFFIX)
            and str(v).strip() and str(v) != NA}


def finalize(codebook: Codebook, answers: dict[str, Answer], level: str,
             context: dict[str, str] | None = None) -> dict[str, Answer]:
    """Answers as they will be stored: skipped questions become NA."""
    questions = codebook.level(level)
    state = codebook.visibility(answers, questions, context)
    out: dict[str, Answer] = {}
    for q in questions:
        if state[q.id] == SKIPPED:
            out[q.id] = NA
        elif state[q.id] == SHOWN:
            answer = answers.get(q.id)
            out[q.id] = None if answer == NA else answer
        else:
            out[q.id] = None
    return out


def unseen_questions(codebook: Codebook, version: str) -> set[str]:
    """
    fixIDs a paper reviewed under `version` has never seen: all of them for a paper
    never saved, none when that version is not on record (we can't tell).
    """
    if not version:
        return {q.id for q in codebook.questions}
    old = snapshot_of(version)
    return set() if old is None else {q.id for q in codebook.questions if q.id not in old}


def fill_defaults(codebook: Codebook, bucket: dict, level: str, qids: set[str]) -> None:
    """Give the blank questions in `qids` their default response, in place."""
    for q in codebook.level(level):
        if q.default is None or q.id not in qids or not is_blank(bucket["answers"].get(q.id)):
            continue
        bucket["answers"][q.id] = list(q.default) if isinstance(q.default, list) else q.default
        if q.kind == "free":
            bucket["free"][q.id] = q.default not in {label.value for label in q.labels}


def missing(codebook: Codebook, answers: dict[str, Answer], level: str,
            context: dict[str, str] | None = None) -> list[str]:
    """
    Ids of shown questions with no answer yet.

    Pending questions are left out: their blank parent is already on the list, and
    they either appear or become NA once it is answered.
    """
    questions = codebook.level(level)
    state = codebook.visibility(answers, questions, context)
    return [q.id for q in questions
            if state[q.id] == SHOWN and (is_blank(answers.get(q.id)) or answers.get(q.id) == NA)]


def _encode_row(codebook: Codebook, level: str, stored: dict, free: dict) -> dict:
    row = {}
    for q in codebook.level(level):
        is_free = bool(free.get(q.id)) and stored[q.id] not in (None, NA)
        row[q.id] = encode(q, resolve_answer(codebook, q, stored[q.id],
                                             is_free or (q.kind == "free" and not q.labels)))
        if has_free_column(q):
            row[q.id + FREE_SUFFIX] = "True" if is_free else ""
    return row


# ──────────────────────────────────────────────────────────────────────────────
# Upsert + archive
# ──────────────────────────────────────────────────────────────────────────────

def _key_mask(df: pd.DataFrame, key: list[str], row: dict) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for column in key:
        mask &= df[column] == row[column]
    return mask


def _archive_previous(df: pd.DataFrame, mask: pd.Series, version: str,
                      archive_path: Path) -> None:
    """
    Keep the old row when it is first overwritten under a newer codebook version.

    A changed condition can turn a stored answer into NA on the next save; the
    archive keeps what was there. Once per version change, not on every autosave.
    """
    previous = df[mask]
    if previous.empty:
        return
    old = previous.iloc[0]
    if not old.get("saved_with") or old.get("saved_with") == version:
        return
    archive = read_csv(archive_path)
    entry = pd.DataFrame([{**old.to_dict(), "archived_at": now()}])
    archive = pd.concat([archive, entry], ignore_index=True).fillna("")
    write_csv_atomic(archive, archive_path)


def _upsert(path: Path, df: pd.DataFrame, key: list[str], row: dict,
            columns: list[str], archive_path: Path) -> None:
    mask = _key_mask(df, key, row)
    _archive_previous(df, mask, row["saved_with"], archive_path)
    if mask.any():
        # Columns the codebook no longer writes (removed questions) keep their value.
        previous = df[mask].iloc[0]
        row = {**{c: previous[c] for c in df.columns if c not in row}, **row}
    df = df[~mask]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = df.reindex(columns=columns).fillna("")
    write_csv_atomic(_sorted(df, key), path)


def _sorted(df: pd.DataFrame, key: list[str]) -> pd.DataFrame:
    if "dataset_idx" in key:
        df = df.assign(_idx=pd.to_numeric(df["dataset_idx"], errors="coerce"))
        return df.sort_values([c for c in key if c != "dataset_idx"] + ["_idx"],
                              kind="stable").drop(columns="_idx")
    return df.sort_values(key, kind="stable")


# ──────────────────────────────────────────────────────────────────────────────
# Papers
# ──────────────────────────────────────────────────────────────────────────────

def load_papers() -> pd.DataFrame:
    return read_csv(PAPERS_PATH, PAPER_META)


def load_datasets() -> pd.DataFrame:
    return read_csv(DATASETS_PATH, DATASET_META)


def get_paper(codebook: Codebook, paper_id: str, reviewer: str,
              papers: pd.DataFrame | None = None) -> dict:
    """
    One paper for one reviewer:
    {"answers", "free", "status", "notes", "codebook_version", "last_modified", "retired"}.
    """
    papers = load_papers() if papers is None else papers
    match = papers[(papers["paper_id"] == paper_id) & (papers["reviewer"] == reviewer)]
    if match.empty:
        answers, free = _row_to_answers(codebook, "paper", pd.Series(dtype=str))
        return {"answers": answers, "free": free, "status": "", "notes": "",
                "codebook_version": "", "last_modified": "", "retired": {}}
    row = match.iloc[0]
    answers, free = _row_to_answers(codebook, "paper", row)
    return {"answers": answers, "free": free, "status": row["status"],
            "notes": row["notes"], "codebook_version": row["codebook_version"],
            "last_modified": row["last_modified"],
            "retired": _retired(codebook, "paper", row)}


def save_paper(codebook: Codebook, paper_id: str, reviewer: str,
               answers: dict[str, Answer], free: dict[str, bool],
               status: str = IN_PROGRESS, notes: str = "",
               codebook_version: str | None = None) -> None:
    """
    `codebook_version` is the version the reviewer has confirmed; it defaults to the
    current one, which is right for a first save.
    """
    papers = load_papers()
    stored = finalize(codebook, answers, "paper")
    current = codebook.version
    row = {"paper_id": paper_id, "reviewer": reviewer, "status": status,
           "codebook_version": codebook_version or current, "saved_with": current,
           "last_modified": now(), "notes": notes,
           **_encode_row(codebook, "paper", stored, free)}
    _upsert(PAPERS_PATH, papers, PAPER_KEY, row,
            _schema(codebook, "paper", list(papers.columns)), PAPERS_ARCHIVE_PATH)


# ──────────────────────────────────────────────────────────────────────────────
# Datasets
# ──────────────────────────────────────────────────────────────────────────────

def get_datasets(codebook: Codebook, paper_id: str, reviewer: str,
                 datasets: pd.DataFrame | None = None) -> list[dict]:
    """The reviewer's datasets for a paper, ordered by `dataset_idx`."""
    datasets = load_datasets() if datasets is None else datasets
    rows = datasets[(datasets["paper_id"] == paper_id) & (datasets["reviewer"] == reviewer)]
    out = []
    for _, row in rows.iterrows():
        answers, free = _row_to_answers(codebook, "dataset", row)
        out.append({"idx": int(row["dataset_idx"]), "label": row["dataset_label"],
                    "answers": answers, "free": free,
                    "retired": _retired(codebook, "dataset", row)})
    return sorted(out, key=lambda d: d["idx"])


def save_dataset(codebook: Codebook, paper_id: str, reviewer: str, idx: int,
                 label: str, answers: dict[str, Answer], free: dict[str, bool],
                 paper_answers: dict[str, Answer]) -> None:
    datasets = load_datasets()
    context = codebook.visibility(paper_answers, codebook.level("paper"))
    stored = finalize(codebook, {**paper_answers, **answers}, "dataset", context)
    row = {"paper_id": paper_id, "reviewer": reviewer, "dataset_idx": str(idx),
           "dataset_label": label, "saved_with": codebook.version, "last_modified": now(),
           **_encode_row(codebook, "dataset", stored, free)}
    _upsert(DATASETS_PATH, datasets, DATASET_KEY, row,
            _schema(codebook, "dataset", list(datasets.columns)), DATASETS_ARCHIVE_PATH)


def delete_datasets(paper_id: str, reviewer: str, idx: int | None = None) -> int:
    """Remove one dataset (or all of a paper's, for this reviewer). Returns rows removed."""
    datasets = load_datasets()
    mask = (datasets["paper_id"] == paper_id) & (datasets["reviewer"] == reviewer)
    if idx is not None:
        mask &= datasets["dataset_idx"] == str(idx)
    removed = int(mask.sum())
    if removed:
        write_csv_atomic(datasets[~mask], DATASETS_PATH)
    return removed


# ──────────────────────────────────────────────────────────────────────────────
# Labels
# ──────────────────────────────────────────────────────────────────────────────

def load_label_log() -> pd.DataFrame:
    return read_csv(LABELS_PATH, LABEL_LOG_COLUMNS)


def log_label(pool: str, label: str, added_by: str, paper_id: str,
              renamed_to: str = "") -> None:
    """Append one addition (or rename, when `renamed_to` is set) to the label log."""
    log = load_label_log()
    row = {"pool": pool, "label": label.strip(), "added_by": added_by,
           "added_at": now(), "paper_id": paper_id, "renamed_to": renamed_to.strip()}
    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    write_csv_atomic(log.reindex(columns=LABEL_LOG_COLUMNS).fillna(""), LABELS_PATH)


def _rename_cell(cell: str, kind: str, old: str, new: str) -> str:
    if cell in ("", NA):
        return cell
    if kind == "multi":
        parts: list[str] = []
        for part in cell.split(MULTI_SEP):
            part = new if part.strip() == old else part.strip()
            if part and part not in parts:
                parts.append(part)
        return MULTI_SEP.join(parts)
    return new if cell == old else cell


def rename_error(codebook: Codebook, pool: str, old: str, new: str) -> str | None:
    """Why this rename is not allowed, or None when it is."""
    old, new = old.strip(), new.strip()
    if not new:
        return "the new name is empty"
    if new.casefold() == old.casefold():
        return "the new name is the same as the old one"
    owner = codebook.by_id[pool]
    current = {label.value.casefold() for label in codebook.labels(owner)}
    if old.casefold() not in current:
        return f"“{old}” is not a current label of {pool}"
    if new.casefold() in current:
        return f"“{new}” already exists in {pool}; renaming onto an existing label is not allowed"
    return None


def rename_label(codebook: Codebook, pool: str, old: str, new: str, by: str,
                 paper_id: str = "") -> int:
    """
    Rename a label everywhere: log it, then rewrite every stored answer.

    Covers every reviewer and every question drawing on `pool` (so ML.task and
    DA.task together). Typed free-text answers are never touched. Returns the number
    of cells changed.

    Raises ValueError when the new name is empty, equals the old one, or already
    exists in the pool (ignoring case): renaming never merges two labels.
    """
    old, new = old.strip(), new.strip()
    rename_problem = rename_error(codebook, pool, old, new)
    if rename_problem:
        raise ValueError(rename_problem)
    log_label(pool, old, by, paper_id, renamed_to=new)
    changed = 0
    for level, path in (("paper", PAPERS_PATH), ("dataset", DATASETS_PATH)):
        df = read_csv(path)
        if df.empty:
            continue
        touched = False
        for q in codebook.level(level):
            if codebook.pool_owner(q).id != pool or q.id not in df.columns:
                continue
            if q.kind == "free" and not q.labels:
                continue
            free_col = q.id + FREE_SUFFIX
            for i, cell in df[q.id].items():
                if free_col in df.columns and df.at[i, free_col] == "True":
                    continue
                renamed = _rename_cell(cell, q.kind, old, new)
                if renamed != cell:
                    df.at[i, q.id] = renamed
                    changed += 1
                    touched = True
        if touched:
            write_csv_atomic(df, path)
    return changed


# ──────────────────────────────────────────────────────────────────────────────
# Reviewers
# ──────────────────────────────────────────────────────────────────────────────

def reviewers() -> list[str]:
    """Everyone who has saved at least one paper, in first-seen order."""
    names: list[str] = []
    for name in list(load_papers()["reviewer"]) + list(read_csv(TIME_PATH, TIME_COLUMNS)["reviewer"]):
        if name and name not in names:
            names.append(name)
    return names


def canonical_reviewer(name: str, known: list[str] | None = None) -> str:
    """Trim and collapse whitespace; reuse an existing spelling that matches up to case."""
    name = re.sub(r"\s+", " ", name or "").strip()
    for existing in known if known is not None else reviewers():
        if existing.casefold() == name.casefold():
            return existing
    return name


# ──────────────────────────────────────────────────────────────────────────────
# Time
# ──────────────────────────────────────────────────────────────────────────────

def load_time() -> pd.DataFrame:
    return read_csv(TIME_PATH, TIME_COLUMNS)


def add_time(reviewer: str, paper_id: str, seconds: float) -> float:
    """Add seconds to (reviewer, paper). Returns the new total for that paper."""
    df = load_time()
    mask = (df["reviewer"] == reviewer) & (df["paper_id"] == paper_id)
    stamp = now()
    if mask.any():
        i = df.index[mask][0]
        total = float(df.at[i, "seconds"] or 0) + seconds
        df.at[i, "seconds"] = f"{total:.1f}"
        df.at[i, "last_seen"] = stamp
    else:
        total = seconds
        df = pd.concat([df, pd.DataFrame([{"reviewer": reviewer, "paper_id": paper_id,
                                           "seconds": f"{total:.1f}", "first_opened": stamp,
                                           "last_seen": stamp}])], ignore_index=True)
    write_csv_atomic(df.reindex(columns=TIME_COLUMNS).sort_values(["paper_id", "reviewer"],
                                                                  kind="stable"), TIME_PATH)
    return total


def seconds_on(reviewer: str, paper_id: str) -> float:
    df = load_time()
    match = df[(df["reviewer"] == reviewer) & (df["paper_id"] == paper_id)]
    return float(match.iloc[0]["seconds"] or 0) if not match.empty else 0.0


def time_per_paper() -> pd.DataFrame:
    """Minutes per paper (rows) and reviewer (columns), for the analysis."""
    df = load_time()
    if df.empty:
        return pd.DataFrame()
    df = df.assign(minutes=df["seconds"].astype(float) / 60)
    return df.pivot_table(index="paper_id", columns="reviewer", values="minutes",
                          aggfunc="sum").round(1)


# ──────────────────────────────────────────────────────────────────────────────
# For analysis
# ──────────────────────────────────────────────────────────────────────────────

def load_resolved(codebook: Codebook, level: str) -> pd.DataFrame:
    """
    The answers CSV with label renames applied.

    Renames already rewrite the stored answers, so this is a safety net (e.g. for a
    file restored from an old commit). Multi-selects stay " | "-joined strings.
    """
    df = load_papers() if level == "paper" else load_datasets()
    for q in codebook.level(level):
        if q.id in df.columns and q.updatable:
            df[q.id] = df[q.id].map(lambda cell, q=q: encode(q, decode(codebook, q, cell)))
    return df
