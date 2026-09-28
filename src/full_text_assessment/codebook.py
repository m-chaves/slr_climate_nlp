"""
The full-text assessment codebook: questions, branching logic, and label pools.

`documents/full-text_assessment_questions.xlsx` is the single source of truth. This
module turns each row into a `Question`, parses its `Condition` into a small AST, and
decides -- for a given set of answers -- whether each question is *shown*, *skipped*
(written as NA), or *pending* (a parent is still unanswered).

Condition grammar (no `eval`; anything else is rejected):

    condition  := comparison (("AND" | "OR") comparison)*
    comparison := QID ("=" | "!=") value
                | QID ["not"] "in" "(" value ("," value)* ")"
    value      := 'text' | "text" | integer

Evaluation is three-valued (True / False / unknown), which gives the rules we want
without special cases:

  - a parent that is *skipped* makes every comparison on it False, so a skipped
    question skips its children -- including `!=` conditions, where a naive
    "NA != x" would wrongly come out True;
  - a parent that is *unanswered* makes the comparison unknown, so the child waits
    ("pending") rather than being shown or skipped prematurely.

Multi-select answers are sets. `X in (..)` is true when any selected label is in the
list; `X not in (..)` is true when at least one selected label is outside it. `=` and
`!=` are the one-element versions of the same. Comparisons are case-insensitive.

Labels written as "N = description" are *coded*: the answer stored is the code N, and
conditions compare against it (`MOT.theory=4`).

Questions are keyed by their **fixID** (`MIS.def`), which never changes when questions
are added, removed or reordered. The xlsx `ID` column (`MIS.3`) is kept as
`display_id` for the UI only. Conditions, label pools and the answer CSVs all use
fixIDs. `Codebook.version` fingerprints the questions, so answers saved under an older
codebook can be diffed against the current one (see `diff_snapshots`).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODEBOOK_PATH = PROJECT_ROOT / "documents" / "full-text_assessment_questions.xlsx"

NA = "NA"                      # value written for a question the logic skipped
MULTI_SEP = " | "              # joins multi-select answers in the CSVs

# Labels that mean "no substantive answer". In a multi-select they cannot be combined
# with any other label.
EXCLUSIVE_LABELS = {
    "na", "not reported", "unclear", "none",
    "no responses are presented", "no truth status established",
}

KINDS = {                      # xlsx "Type" (lower-cased, "updatable " stripped) -> kind
    "single-select": "single",
    "multi-select": "multi",
    "free text": "free",
    "integer": "integer",
    "yyyy": "year",
}
LEVELS = {"paper", "dataset"}

QID_PATTERN = r"[A-Z]+\.[A-Za-z0-9_]+"
_POOL_REF = re.compile(r"^\s*same labels as\s+(" + QID_PATTERN + r")\s*$", re.IGNORECASE)
_CODED = re.compile(r"^\s*(\d+)\s*=\s*(.+)$", re.DOTALL)


# ──────────────────────────────────────────────────────────────────────────────
# Questions and labels
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Label:
    text: str                  # what the reviewer sees
    code: str | None = None    # "4" for "4 = explicit link plus validation"

    @property
    def value(self) -> str:
        """What is stored in the CSV and compared against in conditions."""
        return self.code if self.code is not None else self.text


@dataclass
class Question:
    id: str                    # the fixID: stable key everywhere
    section_key: str
    section: str
    counter: int
    text: str
    level: str                 # "paper" | "dataset"
    kind: str                  # "single" | "multi" | "free" | "integer" | "year"
    updatable: bool
    labels: list[Label]        # initial labels, from the xlsx
    condition_text: str | None
    condition: "Node | None" = None
    pool: str = ""             # label pool id: own id, or the id named in "same labels as"
    raw_type: str = ""
    raw_labels: str = ""
    display_id: str = ""       # the xlsx "ID" column (MIS.3); changes when rows move

    @property
    def label_ids(self) -> str:
        """'MIS.def (MIS.3)' for the UI."""
        return f"{self.id} ({self.display_id})" if self.display_id and self.display_id != self.id else self.id

    @property
    def coded(self) -> bool:
        return any(label.code is not None for label in self.labels)

    @property
    def parents(self) -> list[str]:
        return self.condition.qids() if self.condition else []

    def label_text(self, value: str) -> str:
        """Display text for a stored value (the code maps back to its description)."""
        for label in self.labels:
            if label.value == value:
                return label.text
        return value


# ──────────────────────────────────────────────────────────────────────────────
# Condition parsing
# ──────────────────────────────────────────────────────────────────────────────

class ConditionError(ValueError):
    """A condition string that does not match the grammar."""


_TOKEN = re.compile(r"""
    \s*(?:
        (?P<qid>(?-i:[A-Z]+\.[A-Za-z0-9_]+))
      | (?P<notin>not\s+in)\b
      | (?P<kw>in|and|or)\b
      | (?P<op>!=|=)
      | '(?P<sq>[^']*)'
      | "(?P<dq>[^"]*)"
      | (?P<num>\d+)
      | (?P<punct>[(),])
    )""", re.VERBOSE | re.IGNORECASE)


def _tokenize(text: str) -> list[tuple[str, str]]:
    tokens, pos = [], 0
    text = text.strip()
    while pos < len(text):
        match = _TOKEN.match(text, pos)
        if not match or match.end() == pos:
            raise ConditionError(f"cannot parse {text!r} at position {pos}: {text[pos:]!r}")
        pos = match.end()
        kind = match.lastgroup
        raw = match.group(kind)
        if kind in ("sq", "dq", "num"):
            tokens.append(("value", raw))
        elif kind == "notin":
            tokens.append(("op", "not in"))
        elif kind == "kw":
            low = raw.lower()
            tokens.append(("op", "in") if low == "in" else ("bool", low))
        else:
            tokens.append((kind, raw))
    return tokens


@dataclass(frozen=True)
class Comparison:
    qid: str
    op: str                    # "=", "!=", "in", "not in"
    values: tuple[str, ...]

    def qids(self) -> list[str]:
        return [self.qid]


@dataclass(frozen=True)
class BoolOp:
    op: str                    # "and" | "or"
    children: tuple["Node", ...]

    def qids(self) -> list[str]:
        seen: list[str] = []
        for child in self.children:
            for qid in child.qids():
                if qid not in seen:
                    seen.append(qid)
        return seen


Node = Comparison | BoolOp


def parse_condition(text: str) -> Node:
    """Parse a condition string. Raises `ConditionError` on anything off-grammar."""
    tokens = _tokenize(text)
    if not tokens:
        raise ConditionError("empty condition")
    pos = 0

    def expect(kind: str, value: str | None = None) -> str:
        nonlocal pos
        if pos >= len(tokens):
            raise ConditionError(f"{text!r}: unexpected end, expected {value or kind}")
        tok_kind, tok_value = tokens[pos]
        if tok_kind != kind or (value is not None and tok_value != value):
            raise ConditionError(f"{text!r}: expected {value or kind}, got {tok_value!r}")
        pos += 1
        return tok_value

    def comparison() -> Comparison:
        nonlocal pos
        qid = expect("qid")
        op = expect("op")
        if op in ("=", "!="):
            return Comparison(qid, op, (expect("value"),))
        expect("punct", "(")
        values = [expect("value")]
        while pos < len(tokens) and tokens[pos] == ("punct", ","):
            pos += 1
            values.append(expect("value"))
        expect("punct", ")")
        return Comparison(qid, op, tuple(values))

    # AND binds tighter than OR.
    def conjunction() -> Node:
        nonlocal pos
        parts = [comparison()]
        while pos < len(tokens) and tokens[pos] == ("bool", "and"):
            pos += 1
            parts.append(comparison())
        return parts[0] if len(parts) == 1 else BoolOp("and", tuple(parts))

    parts = [conjunction()]
    while pos < len(tokens) and tokens[pos] == ("bool", "or"):
        pos += 1
        parts.append(conjunction())
    if pos != len(tokens):
        raise ConditionError(f"{text!r}: unexpected {tokens[pos][1]!r}")
    return parts[0] if len(parts) == 1 else BoolOp("or", tuple(parts))


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

SHOWN, SKIPPED, PENDING = "shown", "skipped", "pending"

Answer = str | list[str] | None


def is_blank(answer: Answer) -> bool:
    if answer is None:
        return True
    if isinstance(answer, list):
        return not [a for a in answer if str(a).strip()]
    return not str(answer).strip()


def _as_set(answer: Answer) -> set[str]:
    items = answer if isinstance(answer, list) else [answer]
    return {str(a).strip().casefold() for a in items if a is not None and str(a).strip()}


def _compare(node: Comparison, state: dict[str, str], answers: dict[str, Answer]) -> bool | None:
    parent_state = state.get(node.qid)
    if parent_state == SKIPPED:
        return False
    if parent_state == PENDING or is_blank(answers.get(node.qid)):
        return None
    selected = _as_set(answers.get(node.qid))
    wanted = {v.casefold() for v in node.values}
    if node.op in ("=", "in"):
        return bool(selected & wanted)
    return bool(selected - wanted)          # "!=" / "not in": something outside the list


def _evaluate(node: Node, state: dict[str, str], answers: dict[str, Answer]) -> bool | None:
    if isinstance(node, Comparison):
        return _compare(node, state, answers)
    results = [_evaluate(child, state, answers) for child in node.children]
    if node.op == "and":
        if False in results:
            return False
        return None if None in results else True
    if True in results:
        return True
    return None if None in results else False


# ──────────────────────────────────────────────────────────────────────────────
# Label pools
# ──────────────────────────────────────────────────────────────────────────────

LABEL_LOG_COLUMNS = ["pool", "label", "added_by", "added_at", "paper_id", "renamed_to"]


@dataclass
class LabelRegistry:
    """
    Labels added or renamed while reading, layered over the xlsx labels.

    Built from the append-only log in `labels_added.csv`. A row with an empty
    `renamed_to` adds `label` to `pool`; a row with `renamed_to` set renames `label`
    to it. Renames are replayed in log order into one flat table, old name -> current
    name: renaming X to Y re-points everything that pointed at X, and Y itself becomes
    current. So undoing a rename (A->B, then B->A) leaves A as the one label, and chains
    (A->B->C) resolve in a single lookup, with no cycles possible.
    """

    added: dict[str, list[str]] = field(default_factory=dict)
    renames: dict[str, dict[str, str]] = field(default_factory=dict)

    @classmethod
    def from_log(cls, log: pd.DataFrame | None) -> "LabelRegistry":
        registry = cls()
        if log is None or log.empty:
            return registry
        for _, row in log.iterrows():
            pool = str(row["pool"]).strip()
            label = str(row["label"]).strip()
            renamed_to = str(row.get("renamed_to") or "").strip()
            if not pool or not label:
                continue
            if renamed_to:
                registry._rename(pool, label, renamed_to)
                registry._add(pool, renamed_to)
            else:
                registry._add(pool, label)
        return registry

    def _add(self, pool: str, label: str) -> None:
        labels = self.added.setdefault(pool, [])
        if label.casefold() not in {existing.casefold() for existing in labels}:
            labels.append(label)

    def _rename(self, pool: str, old: str, new: str) -> None:
        mapping = self.renames.setdefault(pool, {})
        for name, current in list(mapping.items()):
            if current == old:
                mapping[name] = new
        mapping[old] = new
        mapping.pop(new, None)             # the new name is current, not an alias
        for name in [n for n, current in mapping.items() if n == current]:
            del mapping[name]

    def resolve(self, pool: str, label: str) -> str:
        """The current name of a label (itself, unless it was renamed away)."""
        mapping = self.renames.get(pool, {})
        if label in mapping:
            return mapping[label]
        folded = label.casefold()
        for name, current in mapping.items():
            if name.casefold() == folded:
                return current
        return label

    def renamed_away(self, pool: str, label: str) -> bool:
        return self.resolve(pool, label) != label


# ──────────────────────────────────────────────────────────────────────────────
# The codebook
# ──────────────────────────────────────────────────────────────────────────────

def _clean(cell) -> str:
    if cell is None or (not isinstance(cell, str) and pd.isna(cell)):
        return ""
    return str(cell).strip()


def _parse_labels(raw: str) -> list[Label]:
    labels = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        if match := _CODED.match(line):
            labels.append(Label(text=line, code=match.group(1)))
        else:
            labels.append(Label(text=line))
    return labels


def _parse_kind(raw_type: str) -> tuple[str | None, bool]:
    low = raw_type.strip().lower()
    updatable = low.startswith("updatable ")
    if updatable:
        low = low[len("updatable "):]
    return KINDS.get(low), updatable


@dataclass
class Codebook:
    questions: list[Question]
    sections: dict[str, str]                  # key -> section name, in order
    parse_errors: list[tuple[str, str]]       # (question id, message)
    registry: LabelRegistry = field(default_factory=LabelRegistry)

    def __post_init__(self) -> None:
        self.by_id = {q.id: q for q in self.questions}
        self.by_display = {q.display_id: q for q in self.questions if q.display_id}

    # ── versions ─────────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, dict]:
        """fixID -> fingerprint of everything the reviewer sees or the logic uses."""
        return {q.id: fingerprint(q) for q in self.questions}

    @property
    def version(self) -> str:
        return snapshot_version(self.snapshot())

    # ── structure ────────────────────────────────────────────────────────────

    def level(self, level: str) -> list[Question]:
        return [q for q in self.questions if q.level == level]

    def section_questions(self, key: str, level: str | None = None) -> list[Question]:
        return [q for q in self.questions
                if q.section_key == key and (level is None or q.level == level)]

    def children(self, qid: str) -> list[Question]:
        return [q for q in self.questions if qid in q.parents]

    # ── labels ───────────────────────────────────────────────────────────────

    def pool_owner(self, question: Question) -> Question:
        return self.by_id.get(question.pool, question)

    def labels(self, question: Question) -> list[Label]:
        """Current labels: the pool's xlsx labels plus additions, renames applied."""
        owner = self.pool_owner(question)
        pool = owner.id
        out: list[Label] = []
        seen: set[str] = set()
        candidates = list(owner.labels)
        if question.updatable or owner.updatable:
            candidates += [Label(text=t) for t in self.registry.added.get(pool, [])]
        for label in candidates:
            if label.code is None:
                current = self.registry.resolve(pool, label.text)
                label = Label(text=current)
            key = label.value.casefold()
            if key not in seen:
                seen.add(key)
                out.append(label)
        return out

    def resolve(self, question: Question, value: str) -> str:
        """Current name of a stored label value (renames applied)."""
        return self.registry.resolve(self.pool_owner(question).id, value)

    # ── branching ────────────────────────────────────────────────────────────

    def visibility(self, answers: dict[str, Answer],
                   questions: list[Question] | None = None,
                   context: dict[str, str] | None = None) -> dict[str, str]:
        """
        shown / skipped / pending for each question.

        `context` carries the visibility already computed for questions outside
        `questions` -- the paper-level state when evaluating one dataset.
        Questions are evaluated in dependency order, so a parent's state is always
        known before its children's.
        """
        questions = self.questions if questions is None else questions
        state: dict[str, str] = dict(context or {})
        for question in self.dependency_order(questions):
            if question.condition is None:
                state[question.id] = SHOWN
                continue
            result = _evaluate(question.condition, state, answers)
            state[question.id] = {True: SHOWN, False: SKIPPED, None: PENDING}[result]
        return state

    def dependency_order(self, questions: list[Question]) -> list[Question]:
        """Questions reordered so every parent comes before its children."""
        ids = {q.id for q in questions}
        done: set[str] = set()
        ordered: list[Question] = []

        def visit(question: Question, stack: tuple[str, ...]) -> None:
            if question.id in done or question.id in stack:
                return
            for parent in question.parents:
                if parent in ids:
                    visit(self.by_id[parent], stack + (question.id,))
            done.add(question.id)
            ordered.append(question)

        for question in questions:
            visit(question, ())
        return ordered

    def datasets_applicable(self, paper_answers: dict[str, Answer]) -> bool | None:
        """
        Whether the paper has datasets to describe: True, False, or None (undecided).

        Decided by the paper-level questions that the dataset-level conditions refer
        to (DA.1 in the current codebook), not by a hard-coded id.
        """
        paper_state = self.visibility(paper_answers, self.level("paper"))
        gated = [q for q in self.level("dataset")
                 if q.condition and all(p in paper_state for p in q.parents)]
        if not gated:
            return True
        results = {_evaluate(q.condition, paper_state, paper_answers) for q in gated}
        if True in results:
            return True
        return None if None in results else False


def load_codebook(path: Path = CODEBOOK_PATH,
                  label_log: pd.DataFrame | None = None) -> Codebook:
    """Read the xlsx into a `Codebook`. Parse problems are collected, not raised."""
    sheets = pd.read_excel(path, sheet_name=None)
    names = list(sheets)
    df = sheets[names[0]]
    sections: dict[str, str] = {}
    if len(names) > 1 and {"Key", "Value"} <= set(sheets[names[1]].columns):
        for _, row in sheets[names[1]].iterrows():
            sections[_clean(row["Key"])] = _clean(row["Value"])

    questions: list[Question] = []
    errors: list[tuple[str, str]] = []
    fix_column = next((c for c in df.columns
                       if str(c).replace(" ", "").replace("_", "").lower() == "fixid"), None)
    if fix_column is None:
        errors.append(("", "the xlsx has no fixID column; add one (e.g. MIS.def) and write "
                           "conditions in terms of it"))
    for _, row in df.iterrows():
        display_id = _clean(row.get("ID"))
        qid = _clean(row.get(fix_column)) if fix_column else ""
        if not qid and not display_id:
            continue
        if not qid:
            if fix_column:
                errors.append((display_id, f"row {display_id} has no fixID"))
            qid = display_id
        raw_type = _clean(row.get("Type"))
        raw_labels = _clean(row.get("Labels"))
        kind, updatable = _parse_kind(raw_type)
        if kind is None:
            errors.append((qid, f"unknown Type {raw_type!r}"))
            kind = "free"

        pool = qid
        labels: list[Label] = []
        if match := _POOL_REF.match(raw_labels):
            pool = match.group(1)
        else:
            labels = _parse_labels(raw_labels)

        condition_text = _clean(row.get("Condition")) or None
        condition = None
        if condition_text:
            try:
                condition = parse_condition(condition_text)
            except ConditionError as exc:
                errors.append((qid, str(exc)))

        counter = row.get("Section counter")
        section_key = _clean(row.get("Section key"))
        questions.append(Question(
            id=qid,
            section_key=section_key,
            section=_clean(row.get("Section")) or sections.get(section_key, section_key),
            counter=int(counter) if not pd.isna(counter) else 0,
            text=_clean(row.get("Question")),
            level=_clean(row.get("Level")).lower(),
            kind=kind,
            updatable=updatable,
            labels=labels,
            condition_text=condition_text,
            condition=condition,
            pool=pool,
            raw_type=raw_type,
            raw_labels=raw_labels,
            display_id=display_id,
        ))

    if not sections:
        for q in questions:
            sections.setdefault(q.section_key, q.section)
    return Codebook(questions=questions, sections=sections, parse_errors=errors,
                    registry=LabelRegistry.from_log(label_log))


# ──────────────────────────────────────────────────────────────────────────────
# Versions: what changed between two codebooks
# ──────────────────────────────────────────────────────────────────────────────

FINGERPRINT_FIELDS = {          # field -> how a change is described to the reviewer
    "text": "wording",
    "kind": "type",
    "updatable": "type",
    "level": "level",
    "labels": "labels",
    "pool": "labels",
    "condition": "condition",
    "section": "section",
}


def fingerprint(q: Question) -> dict:
    """The parts of a question whose change a reviewer should look at again.

    Labels added in the tool are left out on purpose: they grow while reading and
    are not a change to the codebook.
    """
    return {
        "display_id": q.display_id,
        "section": q.section_key,
        "text": " ".join(q.text.split()),
        "kind": q.kind,
        "updatable": q.updatable,
        "level": q.level,
        "labels": [label.text for label in q.labels],
        "pool": q.pool,
        "condition": " ".join((q.condition_text or "").split()),
    }


def snapshot_version(snapshot: dict[str, dict]) -> str:
    blob = json.dumps(snapshot, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


@dataclass
class CodebookDiff:
    added: list[str] = field(default_factory=list)          # fixIDs new in `new`
    removed: list[str] = field(default_factory=list)        # fixIDs gone from `new`
    changed: dict[str, list[str]] = field(default_factory=dict)   # fixID -> ["wording", ...]

    def __bool__(self) -> bool:
        return bool(self.added or self.removed or self.changed)


def diff_snapshots(old: dict[str, dict], new: dict[str, dict]) -> CodebookDiff:
    """Questions added, removed, and changed (with what changed) from `old` to `new`."""
    diff = CodebookDiff(added=[q for q in new if q not in old],
                        removed=[q for q in old if q not in new])
    for qid, fp in new.items():
        if qid not in old:
            continue
        what: list[str] = []
        for field_name, description in FINGERPRINT_FIELDS.items():
            if old[qid].get(field_name) != fp.get(field_name) and description not in what:
                what.append(description)
        if what:
            diff.changed[qid] = what
    return diff
