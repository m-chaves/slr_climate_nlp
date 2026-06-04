"""
DOI, ACL Anthology ID, and arXiv ID extraction and normalization utilities.

Shared by data_loading.py, data_merge.py, and citation_network.py.
Kept separate from utils.py to avoid a transitive matplotlib dependency.
"""

import re


# ──────────────────────────────────────────────────────────────────────────────
# Low-level string normalization
# ──────────────────────────────────────────────────────────────────────────────

def normalize_doi(raw_doi: str | None) -> str | None:
    """
    Return a canonical lowercase DOI string.

    Strips common URL and label prefixes (case-insensitive):
      https://doi.org/  http://doi.org/  doi:  doi.org/
    Lowercases and strips surrounding whitespace.
    Does NOT classify arXiv vs non-arXiv — use is_arxiv_doi() for that.
    Returns None for empty or None input.
    """
    if not raw_doi:
        return None
    doi = raw_doi.strip()
    doi = re.sub(r'^https?://doi\.org/', '', doi, flags=re.IGNORECASE)
    doi = re.sub(r'^doi\.org/', '', doi, flags=re.IGNORECASE)
    doi = re.sub(r'^doi:\s*', '', doi, flags=re.IGNORECASE)
    doi = doi.strip().lower()
    return doi or None


def is_arxiv_doi(doi: str | None) -> bool:
    """
    Return True if the (already normalized) DOI is an arXiv DOI.

    arXiv DOIs use the prefix 10.48550, e.g. 10.48550/arxiv.2604.16935.
    These are NOT journal/conference DOIs and should be treated as arXiv
    identifiers rather than definitive publication DOIs.
    """
    if not doi:
        return False
    return doi.startswith("10.48550/")


# ──────────────────────────────────────────────────────────────────────────────
# Record-level extractors  (dispatch on source name)
# ──────────────────────────────────────────────────────────────────────────────

_DOI_FIELD: dict[str, str | list] = {
    "scopus":           "doi",
    "acl":              "doi",
    "semantic_scholar": ["externalIds", "DOI"],   # nested path
    "wos":              "DI",
    "acm":              "doi",
}


def extract_doi(raw: dict, source: str) -> str | None:
    """
    Extract the canonical non-arXiv DOI from a source-specific raw record.

    Dispatches on `source` to read the correct field, then applies
    normalize_doi().  If the normalized result is an arXiv DOI (10.48550/...),
    returns None — arXiv DOIs are captured by extract_arxiv_id() instead.

    Returns None when the record has no DOI or only an arXiv DOI.
    """
    field = _DOI_FIELD.get(source)
    if field is None:
        return None

    if isinstance(field, list):
        # nested path, e.g. ["externalIds", "DOI"]
        val = raw
        for key in field:
            if not isinstance(val, dict):
                return None
            val = val.get(key)
        raw_doi = val
    else:
        raw_doi = raw.get(field)

    doi = normalize_doi(raw_doi)
    if doi is None or is_arxiv_doi(doi):
        return None
    return doi


def extract_arxiv_id(raw: dict, source: str) -> str | None:
    """
    Extract the canonical arXiv ID from a source-specific raw record.

    Checks in priority order:
      1. Source-dedicated arXiv field (most reliable — already bare ID):
           Semantic Scholar: raw["externalIds"]["ArXiv"]
      2. DOI field: if normalize_doi() yields an arXiv DOI (10.48550/...),
         extract the ID portion after the "10.48550/arxiv." prefix.

    Normalization applied to all extracted values:
      - Lowercase
      - Strip any leading "arxiv." prefix from bare IDs
      - New-style IDs: "2604.16935"  (keep as-is after lowercase)
      - Old-style IDs: "cs/0601001", "hep-th/9901001" (keep slash, lowercase)

    Returns None if no arXiv identifier is found.
    """
    # 1. Source-dedicated arXiv field
    if source == "semantic_scholar":
        bare = raw.get("externalIds", {}) if isinstance(raw.get("externalIds"), dict) else {}
        arxiv_val = bare.get("ArXiv")
        if arxiv_val:
            return _normalize_arxiv_id(arxiv_val)

    # 2. DOI field fallback
    field = _DOI_FIELD.get(source)
    if field is None:
        return None

    if isinstance(field, list):
        val = raw
        for key in field:
            if not isinstance(val, dict):
                return None
            val = val.get(key)
        raw_doi = val
    else:
        raw_doi = raw.get(field)

    doi = normalize_doi(raw_doi)
    if doi and is_arxiv_doi(doi):
        # e.g. "10.48550/arxiv.2604.16935" → "2604.16935"
        # strip the "10.48550/" prefix, then strip any "arxiv." prefix
        remainder = doi[len("10.48550/"):]
        return _normalize_arxiv_id(remainder)

    return None


def extract_acl_id(raw: dict, source: str) -> str | None:
    """
    Extract the canonical ACL Anthology paper ID from a source-specific raw record.

    Only two sources carry ACL IDs:
      - ACL Anthology: raw["full_id"]  (e.g. "2024.climatenlp-1.18")
      - Semantic Scholar: raw["externalIds"]["ACL"]  (same format)

    Normalization: strip whitespace and lowercase.
    ACL IDs use a fixed dotted format (year.venue-track.number) that is
    already lowercase in practice, but lowercasing is applied for safety.

    Returns None for all other sources or when the field is absent/empty.
    """
    raw_id: str | None = None
    if source == "acl":
        raw_id = raw.get("full_id")
    elif source == "semantic_scholar":
        ext = raw.get("externalIds")
        if isinstance(ext, dict):
            raw_id = ext.get("ACL")
    if not raw_id:
        return None
    return raw_id.strip().lower() or None


def _normalize_arxiv_id(raw_id: str) -> str | None:
    """
    Normalize a raw arXiv ID to canonical form.

    Strips "arxiv:" or "arxiv." label prefix (case-insensitive), lowercases.
    Returns None for empty input.
    """
    if not raw_id:
        return None
    s = raw_id.strip().lower()
    s = re.sub(r'^arxiv[.:]', '', s)
    return s or None


# ──────────────────────────────────────────────────────────────────────────────
# WoS cited-reference DOI parser  (moved from citation_network.py)
# ──────────────────────────────────────────────────────────────────────────────

def parse_cr_doi(cr_entry: str) -> str | None:
    """
    Extract a canonical lowercase DOI from one WoS CR line.

    Handles all bracket and non-bracket formats observed in the data:
      - DOI 10.1002/aaai.12188
      - DOI DOI 10.1145/3589335.3651910       (double prefix)
      - DOI arXiv:2404.12065
      - DOI [10.1609/..., DOI 10.1609/...]    (bracket, duplicates)
      - DOI [arXiv:..., 10.18653/...]         (bracket, prefer real DOI)
      - DOI [arXiv:..., 10.48550/arXiv....]   (bracket, all arXiv)

    Returns the DOI string (lowercased) or None if no DOI is found.
    Trailing punctuation (.,;)][) is stripped from every extracted token.
    """
    bracket_match = re.search(r'DOI\s+\[([^\]]+)\]', cr_entry, re.IGNORECASE)
    if bracket_match:
        tokens = [t.strip() for t in bracket_match.group(1).split(',')]
        cleaned = []
        for t in tokens:
            t = re.sub(r'^DOI\s+', '', t, flags=re.IGNORECASE).strip()
            t = t.rstrip('.,;)][').lower()
            cleaned.append(t)

        # Prefer non-arXiv journal/conference DOIs (not 10.48550)
        real = [t for t in cleaned if re.match(r'^10\.(?!48550)', t)]
        if real:
            return real[0]
        # Fall back to arXiv DOIs (10.48550/...)
        arxiv_doi = [t for t in cleaned if t.startswith('10.48550')]
        if arxiv_doi:
            return arxiv_doi[0]
        # Fall back to arXiv: identifiers
        arxiv_id = [t for t in cleaned if t.startswith('arxiv:')]
        if arxiv_id:
            return arxiv_id[0]
        return None

    # Non-bracket: "DOI [DOI] token" — optional double prefix
    match = re.search(r'\bDOI\s+(?:DOI\s+)?(\S+)', cr_entry, re.IGNORECASE)
    if match:
        doi = match.group(1).rstrip('.,;)][').lower()
        return doi or None

    return None
