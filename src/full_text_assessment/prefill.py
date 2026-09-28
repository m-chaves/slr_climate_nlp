"""
The pre-filled "Paper info" section: what we already know about each included paper.

Joins the screening output (via `load_included`, which explodes the `ids` column)
with the retrieval tracking sheet and the PDF manifest. Read-only for both inputs.
The answer CSVs carry only `paper_id` and join to this file, so bibliographic data
lives in one place.
"""

from __future__ import annotations

import pandas as pd

from src.full_text_assessment.codebook import PROJECT_ROOT
from src.full_text_assessment.pdfs import MANIFEST_PATH, TRACKING
from src.full_text_assessment.store import DATA_DIR, read_csv, write_csv_atomic
from src.zotero_retrieval import SCREENING_PATH, load_included, value

PAPER_INFO_PATH = DATA_DIR / "paper_info.csv"

ID_COLUMNS = ["doi", "arxiv_id", "acl_id", "scopus_eid", "wos_id"]
COLUMNS = (["paper_id", "title", "year", "published_in", "paper_type"]
           + ID_COLUMNS + ["research_area", "pdf_path"])


def build_paper_info() -> pd.DataFrame:
    included = load_included(PROJECT_ROOT / SCREENING_PATH)
    tracking = pd.read_excel(TRACKING, dtype={"arxiv_id": str})
    manifest = read_csv(MANIFEST_PATH, ["paper_id", "pdf_path"])

    # Only papers that made it to retrieval; notebook 05 is the authority on that set.
    papers = included[included["paper_id"].isin(tracking["paper_id"])].copy()
    pdf_by_id = dict(zip(manifest["paper_id"], manifest["pdf_path"]))

    rows = []
    for _, row in papers.iterrows():
        pdf_path = pdf_by_id.get(row["paper_id"], "")
        if pdf_path and not (PROJECT_ROOT / pdf_path).exists():
            pdf_path = ""
        rows.append({
            "paper_id": row["paper_id"],
            "title": value(row, "title") or "",
            "year": str(row["year"]),
            "published_in": value(row, "published_in") or "",
            "paper_type": value(row, "paper_type") or "",
            **{c: value(row, c) or "" for c in ID_COLUMNS},
            "research_area": value(row, "research_area") or "",
            "pdf_path": pdf_path,
        })
    df = pd.DataFrame(rows, columns=COLUMNS).sort_values(["year", "title"], kind="stable")
    return df.reset_index(drop=True)


def write_paper_info() -> pd.DataFrame:
    df = build_paper_info()
    write_csv_atomic(df, PAPER_INFO_PATH)
    return df


def load_paper_info() -> pd.DataFrame:
    if not PAPER_INFO_PATH.exists():
        return write_paper_info()
    return read_csv(PAPER_INFO_PATH, COLUMNS)


if __name__ == "__main__":
    info = write_paper_info()
    print(f"{len(info)} papers -> {PAPER_INFO_PATH.relative_to(PROJECT_ROOT)}")
    print(f"with a PDF path: {(info['pdf_path'] != '').sum()}")
