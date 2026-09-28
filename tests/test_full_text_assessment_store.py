"""Round trips through the full-text assessment CSVs."""

import pandas as pd
import pytest

from src.full_text_assessment import store
from src.full_text_assessment.codebook import NA, load_codebook


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "PAPERS_PATH", tmp_path / "papers.csv")
    monkeypatch.setattr(store, "DATASETS_PATH", tmp_path / "datasets.csv")
    monkeypatch.setattr(store, "LABELS_PATH", tmp_path / "labels_added.csv")
    return tmp_path


@pytest.fixture(scope="module")
def cb():
    return load_codebook()


def test_paper_round_trip_and_na(tmp_store, cb):
    answers = {"MIS.is_present": "no", "MIS.has_def": "yes",           # MIS.2 is stale: MIS.1 = no
               "CLI.domain": ["wildfires", "heat waves"],
               "MOT.motiv_cite": "we study x", "MIS.def_src": "not reported"}
    free = {"MOT.motiv_cite": True}
    store.save_paper(cb, "p1", "alice", answers, free)
    raw = pd.read_csv(store.PAPERS_PATH, dtype=str, keep_default_na=False)
    row = raw.iloc[0]
    assert row["MIS.has_def"] == NA and row["MIS.findings"] == NA     # skipped -> literal NA
    assert row["CLI.domain"] == "wildfires | heat waves"
    assert row["MOT.motiv_cite__free"] == "True"
    assert row["POL.is_present"] == ""                             # unanswered stays blank

    back = store.get_paper(cb, "p1", "alice")
    assert back["answers"]["CLI.domain"] == ["wildfires", "heat waves"]
    assert back["answers"]["MIS.has_def"] == NA
    assert back["free"]["MOT.motiv_cite"] is True


def test_reviewers_are_separate_rows_sorted(tmp_store, cb):
    store.save_paper(cb, "p2", "bob", {"MIS.is_present": "no"}, {})
    store.save_paper(cb, "p1", "bob", {"MIS.is_present": "yes, direct"}, {})
    store.save_paper(cb, "p1", "alice", {"MIS.is_present": "no"}, {})
    store.save_paper(cb, "p1", "bob", {"MIS.is_present": "no"}, {})    # overwrite, not append
    raw = pd.read_csv(store.PAPERS_PATH, dtype=str, keep_default_na=False)
    assert list(zip(raw["paper_id"], raw["reviewer"])) == [
        ("p1", "alice"), ("p1", "bob"), ("p2", "bob")]
    assert store.get_paper(cb, "p1", "bob")["answers"]["MIS.is_present"] == "no"


def test_datasets_round_trip(tmp_store, cb):
    paper = {"DA.is_data": "yes"}
    store.save_dataset(cb, "p1", "alice", 2, "B", {"DA.annot": "no", "DA.annot_who": ["crowdworkers"]},
                       {}, paper)
    store.save_dataset(cb, "p1", "alice", 1, "A", {"DA.modality": ["text"], "DA.size": "1200"},
                       {}, paper)
    got = store.get_datasets(cb, "p1", "alice")
    assert [d["idx"] for d in got] == [1, 2]
    assert got[0]["answers"]["DA.modality"] == ["text"] and got[0]["answers"]["DA.size"] == "1200"
    assert got[1]["answers"]["DA.annot_who"] == NA               # DA.14 = no
    assert store.delete_datasets("p1", "alice", 2) == 1
    assert len(store.get_datasets(cb, "p1", "alice")) == 1


def test_missing_lists_shown_blanks_only(cb):
    answers = {"MIS.is_present": "yes, direct"}
    gaps = store.missing(cb, answers, "paper")
    assert "MIS.has_def" in gaps and "MIS.def" not in gaps        # MIS.3 pending on MIS.2
    assert "POL.is_present" in gaps


def test_label_log_and_renames_on_read(tmp_store, cb):
    store.save_paper(cb, "p1", "alice", {"CLI.domain": ["wildfire"]}, {})
    store.log_label("CLI.domain", "wildfire", "alice", "p1")
    store.log_label("CLI.domain", "wildfire", "bob", "p2", renamed_to="wildfires")
    cb2 = load_codebook(label_log=store.load_label_log())
    assert store.get_paper(cb2, "p1", "alice")["answers"]["CLI.domain"] == ["wildfires"]
    assert store.load_resolved(cb2, "paper").iloc[0]["CLI.domain"] == "wildfires"
    # the stored row is untouched until alice saves again
    raw = pd.read_csv(store.PAPERS_PATH, dtype=str, keep_default_na=False)
    assert raw.iloc[0]["CLI.domain"] == "wildfire"
