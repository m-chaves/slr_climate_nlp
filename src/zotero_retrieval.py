"""
Full-text retrieval tracking for the papers that passed screening.

The workflow is deliberately split between Zotero (which does the fetching) and
this module (which does the bookkeeping):

  1. `build_identifier_plan`  -> the paste-ready identifier list for Zotero's
                                 "Add Item by Identifier", plus the manual queue
                                 for papers Zotero cannot resolve.
  2. Zotero, by hand          -> Add Item by Identifier, then Find Available PDFs,
                                 then the browser connector for whatever is left.
  3. `reconcile`              -> read the Zotero collection back, match items to
                                 `paper_id`, and report which papers still have
                                 no PDF.

Step 3 is stateless and idempotent: it never writes to Zotero, only reads. Run it
as often as you like -- after every manual sweep, or after swapping an arXiv
preprint for the published version -- and the numbers refresh.

Reading the library needs a *read-only* Zotero API key. Put it in `.env`:

    ZOTERO_API_KEY=...
    ZOTERO_LIBRARY_ID=...      # the numeric user ID from zotero.org/settings/keys

Command line:

    python -m src.zotero_retrieval plan
    python -m src.zotero_retrieval reconcile --collection "SLR climate NLP"
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import requests
from rapidfuzz import fuzz

from src.doi_utils import normalize_doi

# ──────────────────────────────────────────────────────────────────────────────
# Paths and constants
# ──────────────────────────────────────────────────────────────────────────────

SCREENING_PATH = Path("data/processed/screening/screening_consolidated.xlsx")
RETRIEVAL_DIR = Path("data/processed/retrieval")

IDENTIFIERS_PATH = RETRIEVAL_DIR / "zotero_identifiers.txt"
TRACKING_PATH = RETRIEVAL_DIR / "retrieval_tracking.xlsx"
WORKLIST_PATH = RETRIEVAL_DIR / "manual_worklist.html"
AUDIT_PATH = RETRIEVAL_DIR / "collection_audit.xlsx"

ZOTERO_API = "https://api.zotero.org"
ZOTERO_API_VERSION = "3"

# Identifier types Zotero's "Add Item by Identifier" can resolve, in the order we
# prefer them. DOI first so we get the *published* version rather than a preprint;
# ACL Anthology IDs are absent on purpose -- Zotero cannot look them up.
ZOTERO_RESOLVABLE = ["doi", "arxiv_id"]

# Fallback title matching. Zotero's metadata comes from CrossRef, ours from Scopus
# and WoS, so titles differ in case, punctuation, and the occasional subtitle.
TITLE_MATCH_THRESHOLD = 90

PDF_LINK_MODES = {"imported_file", "imported_url"}

# Columns `_describe_item` produces. Pinned so that an empty audit still yields a frame
# with the right schema -- `pd.DataFrame([])` has no columns at all, so callers indexing
# a column would hit a KeyError exactly when there is nothing wrong to report.
UNKNOWN_COLUMNS = ["zotero_key", "item_type", "title", "year", "doi", "arxiv_id",
                   "url", "has_pdf", "duplicate_of"]


# ──────────────────────────────────────────────────────────────────────────────
# Reading the screening output
# ──────────────────────────────────────────────────────────────────────────────

def parse_ids(ids_cell: str | None) -> dict[str, str]:
    """
    Explode the `ids` column back into a dict.

    The column is written by 02_data_merge.ipynb as "doi: 10.x, scopus_eid: 2-s2.0-y",
    i.e. "<type>: <value>" pairs joined by ", " -- see the notebook for the exact
    construction. Returns {} for a blank cell.
    """
    if ids_cell is None or (not isinstance(ids_cell, str) and pd.isna(ids_cell)):
        return {}
    out: dict[str, str] = {}
    for part in str(ids_cell).split(", "):
        if ": " in part:
            key, value = part.split(": ", 1)
            key, value = key.strip(), value.strip()
            if key and value:
                out[key] = value
    return out


def load_screened(screening_path: Path = SCREENING_PATH,
                  include: bool | None = True) -> pd.DataFrame:
    """
    Screened papers with the `ids` column exploded into one column per id type.

    `include=True` keeps the papers that passed screening, `False` the ones that were
    excluded, `None` all of them. One row per paper, indexed positionally, carrying
    `paper_id`, `title`, `year`, and a blank wherever a paper lacks that id.
    """
    df = pd.read_excel(screening_path)
    if include is not None:
        df = df[df["include"] == include]
    papers = df.copy()

    exploded = papers["ids"].map(parse_ids)
    for id_type in ["doi", "arxiv_id", "wos_id", "acl_id", "scopus_eid", "semantic_scholar_id"]:
        papers[id_type] = exploded.map(lambda d, t=id_type: d.get(t)).astype(object)

    # pandas turns an all-missing column into floats, so guard before normalizing.
    papers["doi"] = papers["doi"].map(
        lambda v: normalize_doi(v) if isinstance(v, str) else None
    )
    return papers.reset_index(drop=True)


def load_included(screening_path: Path = SCREENING_PATH) -> pd.DataFrame:
    """The papers that passed screening. Thin wrapper over `load_screened`."""
    return load_screened(screening_path, include=True)


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 -- the identifier plan
# ──────────────────────────────────────────────────────────────────────────────

def value(row, key: str) -> str | None:
    """
    A row field as a clean string, or None.

    Guards against the NaN that pandas leaves in missing object cells -- note that
    `bool(float("nan"))` is True, so a bare truthiness test is not enough.
    """
    raw = row.get(key)
    if raw is None or (not isinstance(raw, str) and pd.isna(raw)):
        return None
    text = str(raw).strip()
    return text or None


def _manual_url(row: pd.Series) -> str | None:
    """Best landing page for a paper Zotero cannot resolve by identifier."""
    if acl_id := value(row, "acl_id"):
        return f"https://aclanthology.org/{acl_id}/"
    if eid := value(row, "scopus_eid"):
        return f"https://www.scopus.com/record/display.uri?eid={eid}&origin=resultslist"
    if wos_id := value(row, "wos_id"):
        return f"https://www.webofscience.com/wos/woscc/full-record/{wos_id}"
    return None


def _scholar_url(title: str) -> str:
    return "https://scholar.google.com/scholar?q=" + requests.utils.quote(f'"{title}"')


def build_identifier_plan(included: pd.DataFrame) -> pd.DataFrame:
    """
    Decide, per paper, how it should reach Zotero.

    `route` is either "identifier" (paste it into Add Item by Identifier) or
    "manual" (no identifier Zotero can resolve -- use the browser connector on the
    landing page). Exactly one identifier per paper, so Zotero never creates
    duplicate items for a paper we happen to hold two ids for.
    """
    rows = []
    for _, row in included.iterrows():
        identifier, id_type = None, None
        for candidate in ZOTERO_RESOLVABLE:
            if found := value(row, candidate):
                identifier, id_type = found, candidate
                break

        rows.append({
            "paper_id": row["paper_id"],
            "title": row["title"],
            "year": row["year"],
            "route": "identifier" if identifier else "manual",
            "identifier_type": id_type,
            # Zotero wants arXiv ids prefixed; DOIs it takes bare.
            "identifier": (f"arXiv:{identifier}" if id_type == "arxiv_id" else identifier),
            "doi": value(row, "doi"),
            "arxiv_id": value(row, "arxiv_id"),
            "acl_id": value(row, "acl_id"),
            "scopus_eid": value(row, "scopus_eid"),
            "wos_id": value(row, "wos_id"),
            "manual_url": _manual_url(row),
            "scholar_url": _scholar_url(str(row["title"])),
            "research_area": value(row, "research_area"),
        })
    return pd.DataFrame(rows)


def write_identifier_file(plan: pd.DataFrame, path: Path = IDENTIFIERS_PATH) -> int:
    """Write the paste-ready identifier list, one per line. Returns the line count."""
    identifiers = plan.loc[plan["route"] == "identifier", "identifier"].tolist()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(identifiers) + "\n", encoding="utf-8")
    return len(identifiers)


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 -- reading the Zotero library
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ZoteroLibrary:
    """Thin read-only client over the Zotero Web API."""

    library_id: str
    api_key: str
    library_type: str = "users"
    session: requests.Session = field(default_factory=requests.Session)

    @property
    def _base(self) -> str:
        return f"{ZOTERO_API}/{self.library_type}/{self.library_id}"

    def _get(self, path: str, **params) -> requests.Response:
        """One GET, honouring Zotero's Backoff / Retry-After throttling headers."""
        for attempt in range(1, 6):
            response = self.session.get(
                f"{self._base}{path}",
                params=params,
                headers={"Zotero-API-Key": self.api_key,
                         "Zotero-API-Version": ZOTERO_API_VERSION},
                timeout=60,
            )
            if response.status_code == 429 or response.status_code >= 500:
                wait = int(response.headers.get("Retry-After", 2 ** attempt))
                time.sleep(wait)
                continue
            response.raise_for_status()
            if backoff := response.headers.get("Backoff"):
                time.sleep(int(backoff))
            return response
        raise RuntimeError(f"Zotero API kept failing for {path}")

    def _paginate(self, path: str, **params) -> list[dict]:
        """All results for an endpoint, 100 at a time."""
        out: list[dict] = []
        start = 0
        while True:
            response = self._get(path, limit=100, start=start, **params)
            batch = response.json()
            out.extend(batch)
            if len(batch) < 100:
                return out
            start += 100

    def find_collection(self, name: str) -> dict:
        """Look up a collection by its display name (case-insensitive)."""
        collections = self._paginate("/collections")
        matches = [c for c in collections
                   if c["data"]["name"].strip().lower() == name.strip().lower()]
        if not matches:
            available = sorted(c["data"]["name"] for c in collections)
            raise LookupError(
                f"No collection named {name!r}. Available: {available}"
            )
        if len(matches) > 1:
            raise LookupError(f"{len(matches)} collections named {name!r}; rename one")
        return matches[0]

    def collection_items(self, collection_key: str) -> list[dict]:
        """Top-level items in a collection (attachments and notes excluded)."""
        return self._paginate(f"/collections/{collection_key}/items/top")

    def attachments_by_parent(self) -> dict[str, list[dict]]:
        """
        Every attachment in the library, grouped by parent item key.

        Fetched library-wide rather than per item: it is two or three requests
        instead of one per paper, and child items are not themselves collection
        members so a collection query would not reliably return them.
        """
        grouped: dict[str, list[dict]] = {}
        for item in self._paginate("/items", itemType="attachment"):
            parent = item["data"].get("parentItem")
            if parent:
                grouped.setdefault(parent, []).append(item)
        return grouped


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 -- matching Zotero items back to paper_id
# ──────────────────────────────────────────────────────────────────────────────

def _item_doi(data: dict) -> str | None:
    """
    DOI of a Zotero item.

    Journal articles have a real `DOI` field; conference papers and preprints
    often carry it as a "DOI: 10.x" line in `Extra` instead.
    """
    if doi := normalize_doi(data.get("DOI")):
        return doi
    for line in str(data.get("extra", "")).splitlines():
        if line.lower().startswith("doi:"):
            if doi := normalize_doi(line.split(":", 1)[1]):
                return doi
    return None


def _item_arxiv_id(data: dict) -> str | None:
    """arXiv id of a Zotero item, from `archiveID`, `Extra`, or an arxiv.org URL."""
    import re

    haystacks = [str(data.get("archiveID", "")), str(data.get("extra", "")),
                 str(data.get("url", ""))]
    for text in haystacks:
        if match := re.search(r"(\d{4}\.\d{4,5})(v\d+)?", text):
            if "arxiv" in text.lower():
                return match.group(1)
    return None


def _is_pdf(attachment: dict) -> bool:
    data = attachment["data"]
    return (data.get("contentType") == "application/pdf"
            and data.get("linkMode") in PDF_LINK_MODES)


def _looks_like_preprint(item_data: dict, attachments: list[dict]) -> bool:
    """
    Flag items whose full text is a preprint rather than the published version.

    Feeds the manual QC pass: these are the ones worth swapping for the version
    of record before the review is written up.
    """
    if item_data.get("itemType") == "preprint":
        return True
    if (item_data.get("DOI") or "").lower().startswith("10.48550/"):
        return True
    sources = [str(item_data.get("url", ""))]
    sources += [str(a["data"].get("url", "")) for a in attachments]
    return any("arxiv.org" in s.lower() or "biorxiv" in s.lower() for s in sources)


def match_items(plan: pd.DataFrame, items: list[dict],
                attachments: dict[str, list[dict]]) -> pd.DataFrame:
    """
    Join Zotero items onto the retrieval plan.

    Matching is by normalized DOI, then arXiv id, then fuzzy title. Adds the
    columns the tracking sheet reports on: whether an item exists, whether it has
    a PDF attached, how it was matched, and whether the PDF looks like a preprint.
    """
    by_doi: dict[str, dict] = {}
    by_arxiv: dict[str, dict] = {}
    for item in items:
        data = item["data"]
        if doi := _item_doi(data):
            by_doi.setdefault(doi, item)
        if arxiv := _item_arxiv_id(data):
            by_arxiv.setdefault(arxiv, item)

    matched_keys: set[str] = set()
    results = []
    for _, row in plan.iterrows():
        item, how = None, None

        if row.get("doi") and row["doi"] in by_doi:
            item, how = by_doi[row["doi"]], "doi"
        elif row.get("arxiv_id") and row["arxiv_id"] in by_arxiv:
            item, how = by_arxiv[row["arxiv_id"]], "arxiv_id"
        else:
            best, best_score = None, 0
            for candidate in items:
                if candidate["key"] in matched_keys:
                    continue
                score = fuzz.token_set_ratio(str(row["title"]).lower(),
                                             str(candidate["data"].get("title", "")).lower())
                if score > best_score:
                    best, best_score = candidate, score
            if best is not None and best_score >= TITLE_MATCH_THRESHOLD:
                item, how = best, f"title ({best_score:.0f})"

        if item is None:
            results.append({"in_zotero": False, "has_pdf": False, "matched_by": None,
                            "zotero_key": None, "zotero_title": None,
                            "n_pdfs": 0, "is_preprint": False})
            continue

        matched_keys.add(item["key"])
        item_attachments = attachments.get(item["key"], [])
        pdfs = [a for a in item_attachments if _is_pdf(a)]
        results.append({
            "in_zotero": True,
            "has_pdf": bool(pdfs),
            "matched_by": how,
            "zotero_key": item["key"],
            "zotero_title": item["data"].get("title"),
            "n_pdfs": len(pdfs),
            "is_preprint": bool(pdfs) and _looks_like_preprint(item["data"], pdfs),
        })

    return pd.concat([plan.reset_index(drop=True), pd.DataFrame(results)], axis=1)


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 -- reporting
# ──────────────────────────────────────────────────────────────────────────────

def summarise(tracking: pd.DataFrame) -> str:
    """Human-readable progress block, printed by the CLI and the notebook."""
    total = len(tracking)
    in_zotero = int(tracking["in_zotero"].sum())
    with_pdf = int(tracking["has_pdf"].sum())
    preprints = int(tracking["is_preprint"].sum())
    lines = [
        f"Included papers               : {total}",
        f"  present in Zotero           : {in_zotero:>3} ({in_zotero / total * 100:.0f}%)",
        f"  with a PDF attached         : {with_pdf:>3} ({with_pdf / total * 100:.0f}%)",
        f"  PDF looks like a preprint   : {preprints:>3}",
        f"  missing from Zotero         : {total - in_zotero:>3}",
        f"  in Zotero but no PDF        : {in_zotero - with_pdf:>3}",
    ]
    return "\n".join(lines)


def write_worklist(tracking: pd.DataFrame, path: Path = WORKLIST_PATH) -> int:
    """
    Clickable HTML checklist of everything still missing a PDF.

    Open it next to Zotero while on the VPN and work top to bottom: each row links
    to the publisher page (via DOI), the fallback landing page, and Google Scholar.
    """
    gaps = tracking[~tracking["has_pdf"]].copy()

    rows = []
    for _, row in gaps.iterrows():
        links = []
        if row.get("doi"):
            links.append(f'<a href="https://doi.org/{row["doi"]}">doi.org</a>')
        if row.get("arxiv_id"):
            links.append(f'<a href="https://arxiv.org/abs/{row["arxiv_id"]}">arXiv</a>')
        if row.get("manual_url"):
            links.append(f'<a href="{row["manual_url"]}">landing page</a>')
        links.append(f'<a href="{row["scholar_url"]}">Scholar</a>')
        status = "not in Zotero" if not row["in_zotero"] else "no PDF attached"
        rows.append(
            f'<tr><td><input type="checkbox"></td>'
            f'<td>{row["year"]}</td>'
            f'<td>{row["title"]}</td>'
            f'<td class="status">{status}</td>'
            f'<td>{" &middot; ".join(links)}</td>'
            f'<td class="pid">{row["paper_id"]}</td></tr>'
        )

    html = f"""<!doctype html>
<meta charset="utf-8"><title>PDF retrieval worklist</title>
<style>
 body {{ font: 14px/1.5 system-ui, sans-serif; margin: 2rem; max-width: 70rem; }}
 table {{ border-collapse: collapse; width: 100%; }}
 td, th {{ border-bottom: 1px solid #ddd; padding: .4rem .6rem; vertical-align: top; }}
 th {{ text-align: left; background: #f6f6f6; }}
 .status {{ white-space: nowrap; color: #a33; }}
 .pid {{ font-family: monospace; font-size: .85em; color: #888; }}
</style>
<h1>PDF retrieval worklist</h1>
<p>{len(gaps)} of {len(tracking)} included papers still have no PDF in Zotero.
Work through these on the VPN with the Zotero browser connector, saving into the
same collection, then re-run <code>python -m src.zotero_retrieval reconcile</code>.</p>
<table>
<tr><th></th><th>Year</th><th>Title</th><th>Status</th><th>Links</th><th>paper_id</th></tr>
{chr(10).join(rows)}
</table>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return len(gaps)


# ──────────────────────────────────────────────────────────────────────────────
# The reverse check -- what is in Zotero that should not be
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CollectionAudit:
    """
    Both directions of the comparison between the collection and the screening table.

    `tracking` is the forward check (one row per included paper). The other two are
    the reverse: items sitting in Zotero that no longer correspond to an included
    paper, split by why.
    """

    tracking: pd.DataFrame          # one row per included paper
    excluded_present: pd.DataFrame  # screening says exclude, but it is in the collection
    unknown: pd.DataFrame           # matches no screened paper at all
    n_items: int                    # top-level items in the collection


def _describe_item(item: dict, attachments: dict[str, list[dict]],
                   included: pd.DataFrame) -> dict:
    """
    Flatten a Zotero item for the audit report.

    `duplicate_of` is filled when the item looks like a second copy of a paper that
    already matched: the first item wins the DOI lookup in `match_items`, so any
    further copy falls through to the unknown pile and would otherwise look like a
    stray.
    """
    data = item["data"]
    doi = _item_doi(data)
    arxiv = _item_arxiv_id(data)
    title = str(data.get("title", ""))

    duplicate_of = None
    if doi is not None:
        hit = included[included["doi"] == doi]
        if not hit.empty:
            duplicate_of = hit.iloc[0]["paper_id"]
    if duplicate_of is None and title:
        scores = included["title"].astype(str).map(
            lambda t: fuzz.token_set_ratio(title.lower(), t.lower())
        )
        if not scores.empty and scores.max() >= TITLE_MATCH_THRESHOLD:
            duplicate_of = included.loc[scores.idxmax(), "paper_id"]

    pdfs = [a for a in attachments.get(item["key"], []) if _is_pdf(a)]
    return {
        "zotero_key": item["key"],
        "item_type": data.get("itemType"),
        "title": title,
        "year": str(data.get("date", ""))[:4],
        "doi": doi,
        "arxiv_id": arxiv,
        "url": data.get("url"),
        "has_pdf": bool(pdfs),
        "duplicate_of": duplicate_of,
    }


def audit_collection(collection_name: str,
                     screening_path: Path = SCREENING_PATH,
                     library_type: str = "users") -> CollectionAudit:
    """
    Compare the Zotero collection against the screening table in both directions.

    The forward direction is `reconcile`: which included papers are missing. This
    adds the reverse, which matters once screening decisions change after the
    collection was populated -- a paper flipped to exclude stays in Zotero unless
    somebody removes it, and would otherwise silently reach the synthesis.

    Every item is accounted for exactly once: matched to an included paper, matched
    to an excluded one, or listed as unknown. Read-only, like everything here.
    """
    library_id, api_key = get_credentials()
    library = ZoteroLibrary(library_id=library_id, api_key=api_key,
                            library_type=library_type)

    collection = library.find_collection(collection_name)
    items = library.collection_items(collection["key"])
    attachments = library.attachments_by_parent()

    included = load_screened(screening_path, include=True)
    tracking = match_items(build_identifier_plan(included), items, attachments)
    claimed = set(tracking["zotero_key"].dropna())

    # Whatever the included papers did not claim, offer to the excluded ones.
    leftover = [i for i in items if i["key"] not in claimed]
    excluded_plan = build_identifier_plan(load_screened(screening_path, include=False))
    excluded_tracking = match_items(excluded_plan, leftover, attachments)
    excluded_present = excluded_tracking[excluded_tracking["in_zotero"]].copy()
    claimed |= set(excluded_present["zotero_key"].dropna())

    unknown = pd.DataFrame(
        [_describe_item(i, attachments, included) for i in items if i["key"] not in claimed],
        columns=UNKNOWN_COLUMNS,
    )

    return CollectionAudit(tracking=tracking, excluded_present=excluded_present,
                           unknown=unknown, n_items=len(items))


def summarise_audit(audit: CollectionAudit) -> str:
    """Human-readable block for the reverse check."""
    matched = int(audit.tracking["in_zotero"].sum())
    n_excluded = len(audit.excluded_present)
    n_unknown = len(audit.unknown)
    n_dupes = 0 if audit.unknown.empty else int(audit.unknown["duplicate_of"].notna().sum())
    lines = [
        f"Items in the Zotero collection    : {audit.n_items}",
        f"  matching an included paper      : {matched:>3}",
        f"  matching an EXCLUDED paper      : {n_excluded:>3}  <- remove from Zotero",
        f"  matching no screened paper      : {n_unknown:>3}",
        f"    of which look like duplicates : {n_dupes:>3}",
        "",
        f"Included papers not in Zotero     : {int((~audit.tracking['in_zotero']).sum())}",
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────────────────────

def get_credentials() -> tuple[str, str]:
    """Read the Zotero credentials from the environment (loaded from `.env`)."""
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("ZOTERO_API_KEY")
    library_id = os.getenv("ZOTERO_LIBRARY_ID")
    if not api_key or not library_id:
        raise RuntimeError(
            "Set ZOTERO_API_KEY and ZOTERO_LIBRARY_ID in .env. Create a read-only "
            "key at https://www.zotero.org/settings/keys -- the numeric user ID is "
            "shown on the same page."
        )
    return library_id, api_key


def reconcile(collection_name: str,
              screening_path: Path = SCREENING_PATH,
              library_type: str = "users") -> pd.DataFrame:
    """
    Read the Zotero collection and rebuild the tracking sheet and worklist.

    Read-only and idempotent -- safe to run after every manual sweep.
    """
    library_id, api_key = get_credentials()
    library = ZoteroLibrary(library_id=library_id, api_key=api_key,
                            library_type=library_type)

    collection = library.find_collection(collection_name)
    items = library.collection_items(collection["key"])
    attachments = library.attachments_by_parent()

    plan = build_identifier_plan(load_included(screening_path))
    tracking = match_items(plan, items, attachments)

    TRACKING_PATH.parent.mkdir(parents=True, exist_ok=True)
    tracking.to_excel(TRACKING_PATH, index=False)
    write_worklist(tracking)
    return tracking


def write_audit(audit: CollectionAudit, path: Path = AUDIT_PATH) -> None:
    """Persist the reverse check: one sheet per category of stray item."""
    path.parent.mkdir(parents=True, exist_ok=True)
    excluded_cols = ["paper_id", "title", "year", "doi", "arxiv_id",
                     "zotero_key", "zotero_title", "matched_by", "has_pdf"]
    with pd.ExcelWriter(path) as writer:
        (audit.excluded_present.reindex(columns=excluded_cols)
         .to_excel(writer, sheet_name="excluded_in_zotero", index=False))
        audit.unknown.to_excel(writer, sheet_name="unknown_items", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("plan", help="write the paste-ready identifier list")

    for name, help_text in [
        ("reconcile", "read Zotero back and refresh the tracking sheet"),
        ("audit", "also check for items in Zotero that are no longer included"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--collection", required=True, help="Zotero collection name")
        p.add_argument("--group", action="store_true",
                       help="the library is a group library, not your personal one")

    args = parser.parse_args()

    if args.command == "plan":
        plan = build_identifier_plan(load_included())
        n = write_identifier_file(plan)
        n_manual = int((plan["route"] == "manual").sum())
        print(f"Wrote {n} identifiers to {IDENTIFIERS_PATH}")
        print(f"{n_manual} papers have no Zotero-resolvable identifier (manual queue)")
        return

    library_type = "groups" if args.group else "users"

    if args.command == "reconcile":
        tracking = reconcile(args.collection, library_type=library_type)
        print(summarise(tracking))
        print(f"\nTracking sheet : {TRACKING_PATH}")
        print(f"Worklist       : {WORKLIST_PATH}")
    else:
        audit = audit_collection(args.collection, library_type=library_type)
        print(summarise(audit.tracking))
        print()
        print(summarise_audit(audit))
        write_audit(audit)
        print(f"\nAudit sheet    : {AUDIT_PATH}")


if __name__ == "__main__":
    main()
