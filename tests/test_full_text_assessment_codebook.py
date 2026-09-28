"""Branching logic and label pools of the full-text assessment codebook."""

import pandas as pd
import pytest

from src.full_text_assessment.audit import ERROR, audit
from src.full_text_assessment.codebook import (
    PENDING, SHOWN, SKIPPED, ConditionError, LABEL_LOG_COLUMNS, load_codebook,
    parse_condition,
)


@pytest.fixture(scope="module")
def cb():
    return load_codebook()


def paper_state(cb, answers):
    return cb.visibility(answers, cb.level("paper"))


# ── the real xlsx ─────────────────────────────────────────────────────────────

def test_codebook_has_no_audit_errors(cb):
    errors = [i for i in audit(cb) if i.severity == ERROR]
    assert errors == []


def test_every_condition_parses_and_evaluates(cb):
    for q in cb.questions:
        if q.condition_text:
            assert q.condition is not None, q.id
            # Unanswered parents -> pending, never an exception.
            assert cb.visibility({}, cb.questions)[q.id] in (PENDING, SKIPPED, SHOWN)


def test_misinformation_no_skips_the_whole_branch(cb):
    state = paper_state(cb, {"MIS.is_present": "no", "POL.is_present": "yes, direct"})
    for q in cb.section_questions("MIS"):
        if q.id != "MIS.is_present":
            assert state[q.id] == SKIPPED, q.id
    # MIS.7 and MIS.9 sit two levels down; they are skipped through their parent.
    assert state["MP.rel_measured"] == SKIPPED and state["MP.rel_evidence"] == SKIPPED


def test_misinformation_yes_opens_first_level_only(cb):
    state = paper_state(cb, {"MIS.is_present": "yes, indirect"})
    assert state["MIS.has_def"] == SHOWN
    assert state["MIS.def"] == PENDING          # waits for MIS.2
    assert state["MIS.agent_identification"] == PENDING          # waits for MIS.6


def test_multiselect_not_in(cb):
    base = {"MIS.is_present": "yes, direct"}
    assert paper_state(cb, {**base, "MIS.agents": ["not reported"]})["MIS.agent_identification"] == SKIPPED
    assert paper_state(cb, {**base, "MIS.agents": ["unclear"]})["MIS.agent_identification"] == SKIPPED
    assert paper_state(cb, {**base, "MIS.agents": ["influencers"]})["MIS.agent_identification"] == SHOWN


def test_multiselect_in_any_overlap(cb):
    ds = cb.level("dataset")
    ctx = paper_state(cb, {"DA.is_data": "yes"})
    answers = {"DA.is_data": "yes", "DA.annot": "yes",
               "DA.annot_who": ["paper's authors", "LLM or generative model"]}
    assert cb.visibility(answers, ds, ctx)["DA.annot_model"] == SHOWN
    answers["DA.annot_who"] = ["paper's authors"]
    assert cb.visibility(answers, ds, ctx)["DA.annot_model"] == SKIPPED


def test_numeric_codes(cb):
    assert paper_state(cb, {"MOT.theory": "0"})["MOT.theory_why"] == SKIPPED
    assert paper_state(cb, {"MOT.theory": "0"})["MOT.validation"] == SKIPPED
    assert paper_state(cb, {"MOT.theory": "4"})["MOT.theory_why"] == SHOWN
    assert paper_state(cb, {"MOT.theory": "4"})["MOT.validation"] == SHOWN
    assert paper_state(cb, {"STAK.are_stakeholders": "0"})["STAK.who"] == SKIPPED


def test_motivation_not_reported(cb):
    state = paper_state(cb, {"MOT.motiv_cite": "not reported", "CON.concl": "they conclude x"})
    for qid in ("MOT.motiv_cat", "MOT.motiv_clear", "MOT.motiv_clear_cite", "CON.motiv_match"):
        assert state[qid] == SKIPPED, qid
    # Free text is "another value", so != 'not reported' holds.
    state = paper_state(cb, {"MOT.motiv_cite": "we want to study x", "CON.concl": "they conclude x"})
    assert state["MOT.motiv_cat"] == SHOWN and state["CON.motiv_match"] == SHOWN


def test_and_is_false_as_soon_as_one_side_is(cb):
    # MP.1 needs both constructs; POL.1 = no decides it even with MIS.1 unanswered.
    assert paper_state(cb, {"POL.is_present": "no"})["MP.rel_measured"] == SKIPPED
    assert paper_state(cb, {"POL.is_present": "yes, direct"})["MP.rel_measured"] == PENDING


def test_dataset_gate(cb):
    assert cb.datasets_applicable({"DA.is_data": "yes"}) is True
    assert cb.datasets_applicable({"DA.is_data": "no"}) is False
    assert cb.datasets_applicable({"DA.is_data": "not reported"}) is False
    assert cb.datasets_applicable({}) is None


def test_annotation_branch(cb):
    ds = cb.level("dataset")
    ctx = paper_state(cb, {"DA.is_data": "yes"})
    state = cb.visibility({"DA.is_data": "yes", "DA.annot": "no"}, ds, ctx)
    assert state["DA.annot_who"] == SKIPPED and state["DA.annot_model"] == SKIPPED


def test_skipped_parent_skips_not_equal_child(cb):
    # DA.11 is `DA.10 != 'not reported'`; with DA.10 skipped it must not show.
    ds = cb.level("dataset")
    ctx = paper_state(cb, {"DA.is_data": "no"})
    assert cb.visibility({"DA.is_data": "no"}, ds, ctx)["DA.size_unit"] == SKIPPED


# ── the parser ────────────────────────────────────────────────────────────────

def test_parser_accepts_the_grammar():
    node = parse_condition("A.1 in ('x', 'y, z') AND B.2 != 0 OR C.3 not in (\"q\")")
    assert node.op == "or"
    assert parse_condition("MIS.2='yes'").values == ("yes",)
    assert parse_condition("DA.1= 'yes'").op == "="


@pytest.mark.parametrize("bad", [
    "MIS.2'=yes'",                 # the old malformed quoting
    "MIS.1 in 'yes'",              # list without parentheses
    "MIS.1 = ",                    # missing value
    "MIS.1 = 'yes' AND",           # dangling operator
    "mis.1 = 'yes'",               # lower-case id
])
def test_parser_rejects_malformed(bad):
    with pytest.raises(ConditionError):
        parse_condition(bad)


# ── label pools ───────────────────────────────────────────────────────────────

def _log(*rows):
    return pd.DataFrame([dict(zip(LABEL_LOG_COLUMNS, r)) for r in rows],
                        columns=LABEL_LOG_COLUMNS)


def test_shared_pool_is_bidirectional():
    cb = load_codebook(label_log=_log(
        ("ML.task", "summarization", "a", "t", "p1", ""),
    ))
    ml1 = [l.text for l in cb.labels(cb.by_id["ML.task"])]
    da17 = [l.text for l in cb.labels(cb.by_id["DA.task"])]
    assert "summarization" in ml1 and ml1 == da17
    assert cb.pool_owner(cb.by_id["DA.task"]).id == "ML.task"


def test_rename_applies_to_both_and_resolves_old_answers():
    cb = load_codebook(label_log=_log(
        ("ML.task", "sentiment", "a", "t", "p1", "sentiment analysis"),
    ))
    for qid in ("ML.task", "DA.task"):
        texts = [l.text for l in cb.labels(cb.by_id[qid])]
        assert "sentiment analysis" in texts and "sentiment" not in texts
        assert cb.resolve(cb.by_id[qid], "sentiment") == "sentiment analysis"


def test_rename_into_existing_label_merges():
    cb = load_codebook(label_log=_log(
        ("CLI.domain", "wildfire", "a", "t", "p1", ""),
        ("CLI.domain", "wildfires", "a", "t", "p2", ""),
        ("CLI.domain", "wildfire", "b", "t", "p3", "wildfires"),
    ))
    texts = [l.text for l in cb.labels(cb.by_id["CLI.domain"])]
    assert texts.count("wildfires") == 1 and "wildfire" not in texts


def test_additions_do_not_leak_to_other_questions():
    cb = load_codebook(label_log=_log(("CLI.domain", "floods", "a", "t", "p1", "")))
    assert "floods" not in [l.text for l in cb.labels(cb.by_id["MIS.type"])]
