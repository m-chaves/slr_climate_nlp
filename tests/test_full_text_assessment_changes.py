"""fixIDs, codebook versions, renames that rewrite the CSVs, reviewers, and time."""

import pandas as pd
import pytest

from src.full_text_assessment import store, timing
from src.full_text_assessment.audit import ERROR, audit
from src.full_text_assessment.codebook import (
    CODEBOOK_PATH, LABEL_LOG_COLUMNS, NA, diff_snapshots, load_codebook,
)


# ── helpers ───────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    for name, filename in [("PAPERS_PATH", "papers.csv"), ("DATASETS_PATH", "datasets.csv"),
                           ("LABELS_PATH", "labels_added.csv"),
                           ("VERSIONS_PATH", "codebook_versions.json"),
                           ("PAPERS_ARCHIVE_PATH", "papers_archive.csv"),
                           ("DATASETS_ARCHIVE_PATH", "datasets_archive.csv"),
                           ("TIME_PATH", "time_log.csv")]:
        monkeypatch.setattr(store, name, tmp_path / filename)
    return tmp_path


def sheets():
    return pd.read_excel(CODEBOOK_PATH, sheet_name=None)


def write_codebook(tmp_path, edit, name="codebook.xlsx"):
    """A copy of the real codebook with `edit(questions_df)` applied."""
    all_sheets = sheets()
    first = list(all_sheets)[0]
    all_sheets[first] = edit(all_sheets[first].copy())
    path = tmp_path / name
    with pd.ExcelWriter(path) as writer:
        for sheet, df in all_sheets.items():
            df.to_excel(writer, sheet_name=sheet, index=False)
    return path


def errors(cb):
    return [i for i in audit(cb) if i.severity == ERROR]


def row_of(df, fix):
    return df.index[df["fixID"] == fix][0]


# ── fixIDs in the audit ───────────────────────────────────────────────────────

def test_real_codebook_uses_fixids():
    cb = load_codebook()
    assert errors(cb) == []
    assert cb.by_id["MIS.def"].display_id == "MIS.3"
    assert cb.by_id["MIS.def"].parents == ["MIS.has_def"]
    assert cb.by_id["DA.task"].pool == "ML.task"


def test_duplicate_fixid_is_an_error(tmp_path):
    def edit(df):
        df.loc[row_of(df, "MIS.def_src"), "fixID"] = "MIS.def"
        return df
    cb = load_codebook(write_codebook(tmp_path, edit))
    assert any(i.check == "unique fixID" and "MIS.def" in i.message for i in errors(cb))


def test_case_only_duplicate_fixid_is_an_error(tmp_path):
    def edit(df):
        df.loc[row_of(df, "MIS.def_src"), "fixID"] = "MIS.Def"
        return df
    cb = load_codebook(write_codebook(tmp_path, edit))
    assert any("differ only in case" in i.message for i in errors(cb))


def test_condition_with_display_id_names_the_fixid(tmp_path):
    def edit(df):
        df.loc[row_of(df, "MIS.def"), "Condition"] = "MIS.2='yes'"
        return df
    cb = load_codebook(write_codebook(tmp_path, edit))
    assert any("uses display ID MIS.2" in i.message and "MIS.has_def" in i.message
               for i in errors(cb))


def test_pool_with_display_id_names_the_fixid(tmp_path):
    def edit(df):
        df.loc[row_of(df, "DA.task"), "Labels"] = "same labels as ML.1"
        return df
    cb = load_codebook(write_codebook(tmp_path, edit))
    assert any("ML.1 is a display ID, use ML.task" in i.message for i in errors(cb))


def test_missing_fixid_column_is_an_error(tmp_path):
    cb = load_codebook(write_codebook(tmp_path, lambda df: df.drop(columns="fixID")))
    assert any("no fixID column" in i.message for i in errors(cb))


def test_bad_fixid_pattern_is_an_error(tmp_path):
    def edit(df):
        df.loc[row_of(df, "CON.subfield"), "fixID"] = "CON sub field"
        return df
    cb = load_codebook(write_codebook(tmp_path, edit))
    assert any(i.check == "fixID" for i in errors(cb))


# ── versions and diffs ────────────────────────────────────────────────────────

def modified(df):
    """Reword one question, drop one, add one, change one condition."""
    df.loc[row_of(df, "MIS.def"), "Question"] = "Quote the definition of misinformation."
    df.loc[row_of(df, "MIS.def_src"), "Condition"] = "MIS.is_present='yes, indirect'"
    df = df.drop(index=row_of(df, "MIS.findings"))
    new = df.loc[row_of(df, "CLI.focus")].copy()
    new["fixID"], new["ID"], new["Question"] = "CLI.scale", "CLI.3", "Which spatial scale?"
    return pd.concat([df, new.to_frame().T], ignore_index=True)


def test_version_is_stable_and_diff_names_the_changes(tmp_path):
    cb1 = load_codebook()
    assert cb1.version == load_codebook().version            # deterministic
    cb2 = load_codebook(write_codebook(tmp_path, modified))
    assert cb2.version != cb1.version
    diff = diff_snapshots(cb1.snapshot(), cb2.snapshot())
    assert diff.added == ["CLI.scale"]
    assert diff.removed == ["MIS.findings"]
    assert diff.changed["MIS.def"] == ["wording"]
    assert diff.changed["MIS.def_src"] == ["condition"]
    assert "CLI.focus" not in diff.changed


def test_labels_added_in_the_tool_do_not_change_the_version():
    log = pd.DataFrame([{"pool": "CLI.domain", "label": "floods", "added_by": "a",
                         "added_at": "t", "paper_id": "p", "renamed_to": ""}],
                       columns=LABEL_LOG_COLUMNS)
    assert load_codebook(label_log=log).version == load_codebook().version


def test_record_version_and_snapshot(tmp_store):
    cb = load_codebook()
    assert store.record_version(cb) == cb.version
    store.record_version(cb)                                    # idempotent
    assert list(store.load_versions()) == [cb.version]
    assert store.snapshot_of(cb.version)["MIS.def"]["display_id"] == "MIS.3"


def test_archive_once_per_version_and_retired_answers(tmp_store, tmp_path):
    cb1 = load_codebook()
    answers = {"MIS.is_present": "yes, direct", "MIS.has_def": "yes",
               "MIS.findings": "misinformation spreads", "MIS.def_src": "not reported"}
    store.save_paper(cb1, "p1", "Ada Lovelace", answers, {"MIS.findings": True})
    store.save_paper(cb1, "p1", "Ada Lovelace", answers, {"MIS.findings": True})
    assert not store.PAPERS_ARCHIVE_PATH.exists()               # same version: no archive

    cb2 = load_codebook(write_codebook(tmp_path, modified))
    paper = store.get_paper(cb2, "p1", "Ada Lovelace")
    assert paper["codebook_version"] == cb1.version             # still the reviewed one
    assert paper["retired"] == {"MIS.findings": "misinformation spreads"}

    for _ in range(2):
        store.save_paper(cb2, "p1", "Ada Lovelace", paper["answers"], paper["free"],
                         codebook_version=paper["codebook_version"])
    archive = pd.read_csv(store.PAPERS_ARCHIVE_PATH, dtype=str, keep_default_na=False)
    assert len(archive) == 1 and archive.iloc[0]["saved_with"] == cb1.version
    assert archive.iloc[0]["MIS.def_src"] == "not reported"

    stored = pd.read_csv(store.PAPERS_PATH, dtype=str, keep_default_na=False).iloc[0]
    assert stored["MIS.findings"] == "misinformation spreads"   # removed column kept
    assert stored["MIS.def_src"] == NA                          # new condition: skipped
    assert stored["codebook_version"] == cb1.version and stored["saved_with"] == cb2.version


# ── renames rewrite the CSVs ──────────────────────────────────────────────────

def test_rename_rewrites_both_files_across_a_shared_pool(tmp_store):
    cb = load_codebook()
    store.save_paper(cb, "p1", "Ada Lovelace",
                     {"ML.task": ["sentiment", "topic modeling"], "MOT.motiv_cite": "sentiment"},
                     {"MOT.motiv_cite": True})
    store.save_paper(cb, "p2", "Alan Turing", {"ML.task": ["sentiment"]}, {})
    store.save_dataset(cb, "p1", "Ada Lovelace", 1, "A", {"DA.task": ["sentiment"]}, {},
                       {"DA.is_data": "yes"})

    n = store.rename_label(cb, "ML.task", "sentiment", "sentiment analysis", "Ada Lovelace")
    assert n == 3
    papers = pd.read_csv(store.PAPERS_PATH, dtype=str, keep_default_na=False)
    datasets = pd.read_csv(store.DATASETS_PATH, dtype=str, keep_default_na=False)
    assert papers["ML.task"].tolist() == ["sentiment analysis | topic modeling",
                                          "sentiment analysis"]
    assert datasets.iloc[0]["DA.task"] == "sentiment analysis"
    assert papers.iloc[0]["MOT.motiv_cite"] == "sentiment"      # typed free text untouched
    log = store.load_label_log()
    assert log.iloc[-1][["pool", "label", "renamed_to"]].tolist() == [
        "ML.task", "sentiment", "sentiment analysis"]


def test_rename_onto_existing_label_is_refused(tmp_store):
    cb = load_codebook()
    store.save_paper(cb, "p1", "Ada Lovelace", {"ML.task": ["sentiment", "retrieval"]}, {})
    for new in ("retrieval", "Retrieval", "sentiment", " "):
        assert store.rename_error(cb, "ML.task", "sentiment", new)
        with pytest.raises(ValueError):
            store.rename_label(cb, "ML.task", "sentiment", new, "Ada Lovelace")
    assert pd.read_csv(store.PAPERS_PATH, dtype=str).iloc[0]["ML.task"] == "sentiment | retrieval"
    assert not store.LABELS_PATH.exists()                    # nothing logged


def log_of(*renames, pool="ML.task"):
    return pd.DataFrame([{"pool": pool, "label": old, "added_by": "a", "added_at": "t",
                          "paper_id": "p", "renamed_to": new} for old, new in renames],
                        columns=LABEL_LOG_COLUMNS)


def labels_of(cb, qid="ML.task"):
    return [l.text for l in cb.labels(cb.by_id[qid])]


def test_undoing_a_rename_leaves_one_label():
    cb = load_codebook(label_log=log_of(("sentiment", "sentiment analysis"),
                                        ("sentiment analysis", "sentiment")))
    for qid in ("ML.task", "DA.task"):
        assert labels_of(cb, qid).count("sentiment") == 1
        assert "sentiment analysis" not in labels_of(cb, qid)
        assert cb.resolve(cb.by_id[qid], "sentiment analysis") == "sentiment"
        assert cb.resolve(cb.by_id[qid], "sentiment") == "sentiment"


def test_rename_chain_resolves_in_one_step():
    cb = load_codebook(label_log=log_of(("sentiment", "polarity"), ("polarity", "valence")))
    q = cb.by_id["ML.task"]
    assert cb.resolve(q, "sentiment") == "valence" and cb.resolve(q, "polarity") == "valence"
    assert "valence" in labels_of(cb)
    assert not {"sentiment", "polarity"} & set(labels_of(cb))


def test_old_name_is_renamed_away_and_not_a_new_label(tmp_store):
    cb = load_codebook()
    store.rename_label(cb, "ML.task", "sentiment", "sentiment analysis", "Ada Lovelace")
    cb = load_codebook(label_log=store.load_label_log())
    assert cb.registry.renamed_away("ML.task", "sentiment")
    assert cb.registry.renamed_away("ML.task", "Sentiment")  # typed with other case
    assert cb.resolve(cb.by_id["DA.task"], "sentiment") == "sentiment analysis"
    # renaming the current name back is allowed: the old name is no longer a label
    assert store.rename_error(cb, "ML.task", "sentiment analysis", "sentiment") is None


def test_stale_session_cannot_write_an_old_name_back(tmp_store):
    cb = load_codebook()
    store.rename_label(cb, "ML.task", "sentiment", "sentiment analysis", "Ada Lovelace")
    cb_after = load_codebook(label_log=store.load_label_log())
    # an open tab still holding the old name saves it
    store.save_paper(cb_after, "p1", "Alan Turing", {"ML.task": ["sentiment"]}, {})
    assert pd.read_csv(store.PAPERS_PATH, dtype=str).iloc[0]["ML.task"] == "sentiment analysis"


# ── reviewers ─────────────────────────────────────────────────────────────────

def test_canonical_reviewer():
    known = ["Ada Lovelace"]
    assert store.canonical_reviewer("  ada   lovelace ", known) == "Ada Lovelace"
    assert store.canonical_reviewer("Alan  Turing", known) == "Alan Turing"


# ── time ──────────────────────────────────────────────────────────────────────

def test_accrue_counts_open_time():
    assert timing.accrue(100, 130, last_interaction=100, paused=False) == 30


def test_accrue_stops_after_idle_limit():
    t0 = 1000
    idle_end = t0 + timing.IDLE_LIMIT
    assert timing.accrue(idle_end - 10, idle_end + 20, t0, False) == 10
    assert timing.accrue(idle_end + 20, idle_end + 50, t0, False) == 0


def test_accrue_pause_and_gap_cap():
    assert timing.accrue(100, 130, 100, paused=True) == 0
    assert timing.accrue(100, 100 + 3600, 100 + 3600, paused=False) == timing.GAP_CAP
    assert timing.accrue(None, 130, 100, False) == 0


def test_time_log_accumulates_per_reviewer_and_paper(tmp_store):
    store.add_time("Ada Lovelace", "p1", 30)
    store.add_time("Ada Lovelace", "p1", 12.5)
    store.add_time("Alan Turing", "p1", 60)
    assert store.seconds_on("Ada Lovelace", "p1") == 42.5
    table = store.time_per_paper()
    assert table.loc["p1", "Alan Turing"] == 1.0
    assert "Ada Lovelace" in store.reviewers()
