"""
Source-specific loaders that produce NormalizedRecord dicts.

Each loader reads one raw data file and returns a list of NormalizedRecord
dicts.  The schema is the same regardless of source — differences in field
names, types, and formats are resolved here, not in the merge layer.

NormalizedRecord schema
-----------------------
{
    # Source IDs — exactly one non-None per record
    "wos_id":              str | None,   # WoS UT field
    "acl_id":              str | None,   # ACL full_id
    "scopus_eid":          str | None,   # Scopus eid
    "semantic_scholar_id": str | None,   # S2 paperId

    # Merge keys (set by extract_* functions from doi_utils)
    "doi":             str | None,   # non-arXiv DOI only
    "acl_anthology_id": str | None,  # ACL Anthology paper ID (e.g. "2024.climatenlp-1.18")
                                     # present for acl records and some semantic_scholar records
    "arxiv_id":        str | None,   # canonical arXiv ID (e.g. "2604.16935")

    # Compatible attributes (coalesced at merge time)
    "title":    str | None,
    "abstract": str | None,
    "year":     int | None,   # always int

    # Source metadata (not merged)
    "_source":  str,    # "scopus"|"acl"|"semantic_scholar"|"wos"|"acm"
    "_node_id": str,    # unique graph key: "{_source}:{source_id}"
    "_raw":     dict,   # complete original record
}
"""

import json
import logging
from pathlib import Path

import wosfile

from src.doi_utils import extract_doi, extract_acl_id, extract_arxiv_id

logger = logging.getLogger(__name__)

# Resolved relative to this file so notebooks work regardless of cwd
DEFAULT_DATA_DIR: Path = Path(__file__).parent.parent / "data"


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _record(
    source: str,
    source_id: str,
    raw: dict,
    *,
    title: str | None = None,
    abstract: str | None = None,
    year: int | None = None,
    wos_id: str | None = None,
    acl_id: str | None = None,
    scopus_eid: str | None = None,
    semantic_scholar_id: str | None = None,
) -> dict:
    return {
        "wos_id":              wos_id,
        "acl_id":              acl_id,
        "scopus_eid":          scopus_eid,
        "semantic_scholar_id": semantic_scholar_id,
        "doi":              extract_doi(raw, source),
        "acl_anthology_id": extract_acl_id(raw, source),
        "arxiv_id":         extract_arxiv_id(raw, source),
        "title":    title,
        "abstract": abstract,
        "year":     year,
        "_source":  source,
        "_node_id": f"{source}:{source_id}",
        "_raw":     raw,
    }


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ──────────────────────────────────────────────────────────────────────────────
# Source loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_scopus(path: Path) -> list[dict]:
    """Load Scopus JSONL export."""
    out = []
    for raw in _read_jsonl(path):
        eid = raw.get("eid") or ""
        year = None
        cover = raw.get("coverDate") or ""
        if len(cover) >= 4:
            try:
                year = int(cover[:4])
            except ValueError:
                pass
        out.append(_record(
            "scopus", eid, raw,
            title=raw.get("title"),
            abstract=raw.get("description"),   # Scopus calls abstract "description"
            year=year,
            scopus_eid=eid or None,
        ))
    logger.info("Scopus: loaded %d records from %s", len(out), path)
    return out


def load_acl(path: Path) -> list[dict]:
    """Load ACL Anthology JSONL export."""
    out = []
    for raw in _read_jsonl(path):
        full_id = raw.get("full_id") or ""
        year = None
        raw_year = raw.get("year")
        if raw_year is not None:
            try:
                year = int(raw_year)
            except (ValueError, TypeError):
                pass
        out.append(_record(
            "acl", full_id, raw,
            title=raw.get("title"),
            abstract=raw.get("abstract"),
            year=year,
            acl_id=full_id or None,
        ))
    logger.info("ACL: loaded %d records from %s", len(out), path)
    return out


def load_semantic_scholar(path: Path) -> list[dict]:
    """Load Semantic Scholar JSONL export."""
    out = []
    for raw in _read_jsonl(path):
        paper_id = raw.get("paperId") or ""
        year = raw.get("year")
        if year is None:
            pub_date = raw.get("publicationDate") or ""
            if len(pub_date) >= 4:
                try:
                    year = int(pub_date[:4])
                except ValueError:
                    pass
        if year is not None:
            try:
                year = int(year)
            except (ValueError, TypeError):
                year = None
        out.append(_record(
            "semantic_scholar", paper_id, raw,
            title=raw.get("title"),
            abstract=raw.get("abstract"),
            year=year,
            semantic_scholar_id=paper_id or None,
        ))
    logger.info("Semantic Scholar: loaded %d records from %s", len(out), path)
    return out


def load_wos(path: Path, *, strip_cr: bool = True) -> list[dict]:
    """
    Load Web of Science tagged-text export.

    Uses wosfile.records_from() which returns multi-value fields (AU, CR, DE)
    as lists.  strip_cr=True removes the cited-references list from _raw to
    keep file size manageable; the merge pipeline does not need CR.
    """
    out = []
    for rec in wosfile.records_from(str(path)):
        raw = dict(rec)
        if strip_cr:
            raw.pop("CR", None)

        ut = raw.get("UT") or ""
        year = None
        raw_py = raw.get("PY")
        if raw_py is not None:
            try:
                year = int(raw_py)
            except (ValueError, TypeError):
                pass
        out.append(_record(
            "wos", ut, raw,
            title=raw.get("TI"),
            abstract=raw.get("AB"),
            year=year,
            wos_id=ut or None,
        ))
    logger.info("WoS: loaded %d records from %s", len(out), path)
    return out


def load_acm(path: Path) -> list[dict]:
    """
    Load ACM Digital Library JSONL export.

    Stub — returns an empty list when the file is absent (ACM returned 0
    results for this query's keyword combination).  The function signature
    is in place so the pipeline can incorporate ACM data without changes
    to load_all_sources() when data becomes available.
    """
    if not path.exists():
        logger.debug("ACM: file not found at %s, skipping", path)
        return []
    out = []
    for raw in _read_jsonl(path):
        doc_id = raw.get("doi") or raw.get("id") or ""
        year = None
        raw_year = raw.get("year") or raw.get("publicationYear")
        if raw_year is not None:
            try:
                year = int(raw_year)
            except (ValueError, TypeError):
                pass
        out.append(_record(
            "acm", str(doc_id), raw,
            title=raw.get("title"),
            abstract=raw.get("abstract"),
            year=year,
        ))
    logger.info("ACM: loaded %d records from %s", len(out), path)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Aggregate loader
# ──────────────────────────────────────────────────────────────────────────────

# Canonical source order (determines processing order in load_all_sources)
_SOURCE_ORDER: list[str] = ["scopus", "acl", "semantic_scholar", "wos", "acm"]

_LOADERS = {
    "scopus":           load_scopus,
    "acl":              load_acl,
    "semantic_scholar": load_semantic_scholar,
    "wos":              load_wos,
    "acm":              load_acm,
}


def default_source_paths(data_dir: Path = DEFAULT_DATA_DIR) -> dict[str, Path]:
    """
    Return the standard file path for every source under `data_dir`.

    Useful as a starting point when you want to load most sources from the
    default location but override or drop one:

        paths = default_source_paths()
        paths["scopus"] = Path("/custom/export.jsonl")
        del paths["acm"]                           # skip ACM
        records = load_all_sources(paths)
    """
    raw_dir = Path(data_dir) / "raw"
    return {
        "scopus":           raw_dir / "scopus"           / "scopus_data.jsonl",
        "acl":              raw_dir / "acl"              / "acl_data.jsonl",
        "semantic_scholar": raw_dir / "semantic_scholar" / "semantic_scholar_data.jsonl",
        "wos":              raw_dir / "wos"              / "wos_data.txt",
        "acm":              raw_dir / "acm"              / "acm_data.jsonl",
    }


def load_all_sources(
    source_paths: dict[str, Path | str] | None = None,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> list[dict]:
    """
    Load sources and return a flat list of NormalizedRecord dicts.

    Args:
        source_paths: explicit mapping of source name → file path.
            Only the listed sources are loaded, in canonical order
            (scopus → acl → semantic_scholar → wos → acm).
            If None, all sources are loaded from `data_dir` using the
            standard naming convention (equivalent to passing
            ``default_source_paths(data_dir)``).
        data_dir: used only when `source_paths` is None to resolve
            paths via the default naming convention.

    Examples::

        # Load everything from the default location
        records = load_all_sources()

        # Load only two sources
        records = load_all_sources({"scopus": "data/raw/scopus/scopus_data.jsonl",
                                    "acl":    "data/raw/acl/acl_data.jsonl"})

        # Load from defaults but override one path
        paths = default_source_paths()
        paths["wos"] = Path("/mnt/shared/wos_export.txt")
        records = load_all_sources(paths)
    """
    if source_paths is None:
        source_paths = default_source_paths(data_dir)

    unknown = set(source_paths) - set(_LOADERS)
    if unknown:
        raise ValueError(f"Unknown source name(s): {unknown}. Expected one of {set(_LOADERS)}.")

    all_records: list[dict] = []
    for source in _SOURCE_ORDER:
        path = source_paths.get(source)
        if path is None:
            continue
        records = _LOADERS[source](Path(path))
        all_records.extend(records)

    logger.info("Total records loaded: %d from %d source(s)", len(all_records), len(source_paths))
    return all_records
