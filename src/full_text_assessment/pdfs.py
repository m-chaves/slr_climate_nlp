"""
Copy the included papers' PDFs out of Zotero's storage into `data/pdfs/<paper_id>.pdf`.

Independent of notebook 05 by construction: it only *reads* the retrieval tracking
sheet (for `paper_id -> zotero_key`) and the Zotero library (read-only API key, via
`ZoteroLibrary` from `src/zotero_retrieval.py`). It writes nothing under
`data/processed/retrieval/` and never writes to Zotero.

Zotero keeps each attachment at `<storage>/<attachment key>/<filename>`. The storage
directory comes from `ZOTERO_STORAGE` in `.env`, e.g.

    ZOTERO_STORAGE=/mnt/c/Users/<you>/Zotero/storage

`data/pdfs/` is gitignored (copyright, size). Idempotent: a copy is redone only when
the source file's size differs.

    python -m src.full_text_assessment.pdfs
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd

from src.full_text_assessment.codebook import PROJECT_ROOT
from src.full_text_assessment.store import DATA_DIR, write_csv_atomic
from src.zotero_retrieval import TRACKING_PATH, ZoteroLibrary, _is_pdf, get_credentials

TRACKING = PROJECT_ROOT / TRACKING_PATH
PDF_DIR = PROJECT_ROOT / "data" / "pdfs"
MANIFEST_PATH = DATA_DIR / "pdf_manifest.csv"


def storage_dir() -> Path:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    raw = os.getenv("ZOTERO_STORAGE")
    if not raw:
        raise RuntimeError("Set ZOTERO_STORAGE in .env to Zotero's storage directory, "
                           "e.g. /mnt/c/Users/<you>/Zotero/storage")
    path = Path(raw).expanduser()
    if not path.is_dir():
        raise RuntimeError(f"ZOTERO_STORAGE={path} is not a directory")
    return path


def build_manifest(storage: Path, library: ZoteroLibrary | None = None) -> pd.DataFrame:
    """
    One row per included paper: which Zotero attachment holds its PDF.

    When a paper has several PDFs attached, the most recently added one wins -- the
    usual reason for a second PDF is swapping a preprint for the published version.
    `n_pdfs` stays in the manifest so those cases can be checked by eye.
    """
    if library is None:
        library_id, api_key = get_credentials()
        library = ZoteroLibrary(library_id=library_id, api_key=api_key)
    attachments = library.attachments_by_parent()
    tracking = pd.read_excel(TRACKING)

    rows = []
    for _, paper in tracking.iterrows():
        pdfs = [a for a in attachments.get(paper["zotero_key"], []) if _is_pdf(a)]
        pdfs.sort(key=lambda a: a["data"].get("dateAdded", ""), reverse=True)
        chosen = pdfs[0]["data"] if pdfs else {}
        source = (storage / chosen["key"] / chosen["filename"]
                  if chosen.get("key") and chosen.get("filename") else None)
        rows.append({
            "paper_id": paper["paper_id"],
            "zotero_key": paper["zotero_key"],
            "n_pdfs": len(pdfs),
            "attachment_key": chosen.get("key", ""),
            "filename": chosen.get("filename", ""),
            "source_path": str(source) if source else "",
            "pdf_path": str((PDF_DIR / f"{paper['paper_id']}.pdf").relative_to(PROJECT_ROOT)),
        })
    return pd.DataFrame(rows)


def copy_pdfs(manifest: pd.DataFrame) -> pd.DataFrame:
    """Copy every PDF in the manifest; adds a `copy_status` column and saves it."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    statuses = []
    for _, row in manifest.iterrows():
        source = Path(row["source_path"]) if row["source_path"] else None
        target = PROJECT_ROOT / row["pdf_path"]
        if source is None:
            statuses.append("no PDF attachment in Zotero")
        elif not source.exists():
            statuses.append("missing in Zotero storage (not synced to this machine?)")
        elif target.exists() and target.stat().st_size == source.stat().st_size:
            statuses.append("up to date")
        else:
            shutil.copy2(source, target)
            statuses.append("copied")
    manifest = manifest.assign(copy_status=statuses)
    write_csv_atomic(manifest, MANIFEST_PATH)
    return manifest


def main() -> None:
    manifest = copy_pdfs(build_manifest(storage_dir()))
    print(manifest["copy_status"].value_counts().to_string())
    print(f"\nManifest : {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")
    print(f"PDFs     : {PDF_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
