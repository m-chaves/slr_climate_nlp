import json
import time
import requests
import os
import yaml
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# CONFIGURATION LOADING
# ─────────────────────────────────────────────

def load_config(config_path="../config.yml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
# BASE EXTRACTION FUNCTION
# ─────────────────────────────────────────────

def collect_semantic_scholar_data(
    query,
    fields,
    output_path,
    date_start,
    date_end,
    api_key="",
    max_retries=5,
    backoff_factor=4,
):
    """
    Fetch Semantic Scholar bulk search results and stream to JSONL.

    Args:
        query (str): string indicating the group topics. Ex: "climate AND nlp"
        fields (str): List of fields to retrieve for each paper. Ex: "paperId,title,year"
        output_path (str): string. Path to output .jsonl file. Ex: "../data/raw/semantic_scholar/sets/climate_nlp.jsonl"
        date_start (datetime): datetime in format YYYY-MM-DD.
        date_end (datetime): datetime in format YYYY-MM-DD.
        api_key (str): Semantic Scholar API key (optional).
        max_retries (int): Number of times to retry a failed request before giving up.
        backoff_factor (int): Exponential backoff factor for retries (e.g., 3 means wait 3s, then 9s, then 27s between retries).
    Returns:
        total_downloaded (int): Total number of papers successfully downloaded.
        grand_total (int): Total number of papers matching the query according to the API

    """

    base_url = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"

    headers = {"x-api-key": api_key} if api_key else {}
    sleep_time = 1.1 if api_key else 3.5

    query_params = {
        "query": query,
        "publicationDateOrYear": f"{date_start}:{date_end}",
        "fields": ",".join(fields),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Query: {query}")
    print(f"Output: {output_path}")
    print(f"API key: {'yes' if api_key else 'no'} | Sleep: {sleep_time}s")

    total_downloaded = 0
    grand_total = None
    has_more = True

    # Better structured error info
    error_info = None

    # Write results to file as we go. Rewrite the file if append=False, otherwise add to it.
    with open(output_path, "w", encoding="utf-8") as f:

        while has_more:

            success = False

            for attempt in range(1, max_retries + 1):

                try:

                    response = requests.get(
                        base_url,
                        params=query_params,
                        headers=headers,
                        timeout=60,
                    )

                    response.raise_for_status()

                    payload = response.json()

                    # Check HTTP-level errors
                    if response.status_code != 200:
                        print(f"Status code: {response.status_code}")

                    # Check for API-level error fields
                    if "error" in payload:
                        print(f"API Error: {payload['error']}")
                    if "message" in payload:
                        print(f"API Message: {payload['message']}")
                    if "warnings" in payload:
                        print(f"Warnings: {payload['warnings']}")
                    
                    # First batch only
                    if total_downloaded == 0:
                        grand_total = payload.get("total", "unknown")
                        print(
                            f"Total papers to retrieve: {grand_total}"
                        )

                    # Process papers in current batch
                    papers_batch = payload.get("data") or []

                    for paper in papers_batch:
                        f.write(json.dumps(paper) + "\n")

                    total_downloaded += len(papers_batch)

                    # pagination
                    next_token = payload.get("token")
                    # print(f"  Next token: {next_token!r}")

                    if next_token:
                        query_params["token"] = next_token
                    else:
                        has_more = False

                    success = True

                    # normal pacing sleep
                    time.sleep(sleep_time)

                    break

                except requests.exceptions.HTTPError as err:

                    status_code = (
                        err.response.status_code
                        if err.response is not None
                        else None
                    )

                    # Retry transient server-side errors
                    # 429, Too Many Requests. You've hit the rate limit, slow down your requests.
                    # 500, Internal Server Error. Something went wrong on the server’s side.
                    retryable = status_code == 429 or status_code >= 500

                    print(
                        f"HTTP error ({status_code}) "
                        f"[attempt {attempt}/{max_retries}]"
                    )

                    if retryable and attempt < max_retries:

                        if status_code == 429:
                            print("Rate limit hit — sleeping 90s")
                            time.sleep(90)
                            continue
                        elif status_code >= 500:
                            wait = backoff_factor ** (attempt - 1)
                            print(f"Retrying in {wait} seconds...")
                            time.sleep(wait)
                            continue

                    # Final failure
                    error_info = {
                        "type": "HTTPError",
                        "status_code": status_code,
                        "message": str(err),
                        "attempt": attempt,
                        "query_params": query_params.copy(),
                        "timestamp": datetime.now().isoformat(),
                    }

                    has_more = False
                    break

                except (
                    requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError,
                ) as err:

                    print(f"Network error [attempt {attempt}/{max_retries}]: {err}")
                    print(f"Error: {type(err).__name__}: {err}")

                    if attempt < max_retries:
                        wait = backoff_factor ** (attempt - 1)
                        print(f"Retrying in {wait} seconds...")
                        time.sleep(wait)

                        continue

                    error_info = {
                        "type": type(err).__name__,
                        "message": str(err),
                        "attempt": attempt,
                        "query_params": query_params.copy(),
                        "timestamp": datetime.now().isoformat(),
                    }

                    has_more = False
                    break

                except Exception as err:

                    print(f"Unexpected error: {err}")

                    error_info = {
                        "type": type(err).__name__,
                        "message": str(err),
                        "query_params": query_params.copy(),
                        "timestamp": datetime.now().isoformat(),
                    }

                    has_more = False
                    break

    # metadata sidecar
    meta_path = output_path.with_name(
        output_path.stem + "_meta.json"
    )

    meta = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "fields": fields,
        "date_start": str(date_start),
        "date_end": str(date_end),
        "api_key_used": bool(api_key),
        "total_downloaded": total_downloaded,
        "grand_total": grand_total,
        "error": error_info,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if grand_total and total_downloaded < grand_total:
        print(
            f"WARNING: downloaded {total_downloaded} of {grand_total} expected papers. "
            f"API stopped pagination early (no token returned). Check {meta_path}.\n"
        )
    else:
        print(
            f"Done — {total_downloaded} papers saved to {output_path}\n"
        )

    return total_downloaded, grand_total


# ─────────────────────────────────────────────
# ACL ANTHOLOGY DATA COLLECTION
# ─────────────────────────────────────────────

from acl_anthology.collections.types import VolumeType as _VolumeType

# Check for other fields of interest at 
# https://acl-anthology.readthedocs.io/py-v1.1.0/api/collections.paper/
# https://acl-anthology.readthedocs.io/py-v1.1.0/api/collections.volume/ 
# and other collection.* documentation
_ACL_FIELD_EXTRACTORS = {
    "full_id":        lambda p: p.full_id,
    "title":         lambda p: p.title.as_text(),
    "abstract":      lambda p: p.abstract.as_text() if p.abstract else None,
    "year":          lambda p: p.year,
    "authors":       lambda p: [{"first": ns.first, "last": ns.last, "affiliation": ns.affiliation} for ns in p.authors],
    "doi":           lambda p: p.doi,
    "language":      lambda p: p.language,
    "journal_title":       lambda p: p.journal_title,
    "paper_type":    lambda p: p.type.value,
    "volume_type":   lambda p: p.parent.type.value,
    "volume_title":   lambda p: p.parent.title.as_text(),
    "venue_ids":     lambda p: p.venue_ids,
    "venue_acronym": lambda p: p.parent.venue_acronym,
    "venue_full_id": lambda p: p.parent.full_id,
}


def paper_to_dict(paper, fields):
    """
    Extract requested fields from an ACL Paper object into a JSON-serialisable dict.
    Valid field names: see _ACL_FIELD_EXTRACTORS keys.

    Raises:
        ValueError: If an unrecognised field name is requested.
    """
    unknown = set(fields) - _ACL_FIELD_EXTRACTORS.keys()
    if unknown:
        raise ValueError(
            f"Unknown field(s) {sorted(unknown)}. "
            f"Valid fields: {sorted(_ACL_FIELD_EXTRACTORS)}"
        )
    return {f: _ACL_FIELD_EXTRACTORS[f](paper) for f in fields}


def collect_acl_data(
    logic_string,
    keywords_path,
    output_path,
    date_start,
    date_end,
    fields,
    anthology_dir=None,
):
    """
    Scan the ACL Anthology for papers matching a boolean keyword query and write
    results to a JSONL file (one JSON object per line).

    Matching is performed against each paper's title and abstract using
    case-insensitive regex. Wildcards in keywords (e.g. "transformer*") are
    expanded to word-boundary prefix patterns.

    Args:
        logic_string (str): Boolean logic over keyword group names,
            e.g. "climate AND (nlp OR misinformation)".
        keywords_path (str): Path to the YAML file defining keyword groups.
        output_path (str): Path to the output .jsonl file.
        date_start: Start of year range. Accepts datetime.date or "YYYY-MM-DD" str.
        date_end: End of year range (inclusive). Accepts datetime.date or "YYYY-MM-DD" str.
        fields (list[str]): Fields to include for each paper.
        anthology_dir (str | None): Path to the Anthology data/ directory.
            If None, the library downloads the data from GitHub on first run
            (~1 GB) and caches it in the user data directory.

    Returns:
        total_matched (int): Number of papers written to the output file.
        total_scanned (int): Total number of papers iterated.
    """
    from acl_anthology import Anthology
    from acl_anthology.collections.types import PaperType

    try:
        from src.query_builder import build_acl_text_matcher, build_query, load_keywords
    except ImportError:
        from query_builder import build_acl_text_matcher, build_query, load_keywords

    groups = load_keywords(keywords_path)
    matcher = build_acl_text_matcher(logic_string, groups)
    human_readable_query = build_query(logic_string, groups)

    if anthology_dir is not None:
        anthology = Anthology(anthology_dir)
    else:
        anthology = Anthology.from_repo()

    year_start = int(str(date_start)[:4])
    year_end = int(str(date_end)[:4])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Logic: {logic_string}")
    print(f"Year range: {year_start}–{year_end}")
    print(f"Output: {output_path}")

    total_matched = 0
    total_scanned = 0
    error_info = None

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for paper in anthology.papers():
                total_scanned += 1
                if total_scanned % 10_000 == 0:
                    print(f"Scanned {total_scanned:,} | matched {total_matched:,}")

                if paper.type != PaperType.PAPER or paper.is_deleted:
                    continue

                try:
                    year = int(paper.year)
                except (ValueError, TypeError):
                    continue

                if not (year_start <= year <= year_end):
                    continue

                if paper.language is not None and paper.language != "eng":
                    continue

                text = paper.title.as_text()
                if paper.abstract:
                    text += " " + paper.abstract.as_text()

                if matcher(text):
                    f.write(json.dumps(paper_to_dict(paper, fields)) + "\n")
                    total_matched += 1

    except Exception as err:
        error_info = {
            "type": type(err).__name__,
            "message": str(err),
            "timestamp": datetime.now().isoformat(),
        }

    meta_path = output_path.with_name(output_path.stem + "_meta.json")
    meta = {
        "timestamp": datetime.now().isoformat(),
        "logic_string": logic_string,
        "human_readable_query": human_readable_query,
        "fields": fields,
        "date_start": str(date_start),
        "date_end": str(date_end),
        "anthology_dir": str(anthology_dir) if anthology_dir else None,
        "total_matched": total_matched,
        "total_scanned": total_scanned,
        "error": error_info,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if error_info:
        print(f"ERROR during scan — check {meta_path}")
    else:
        print(
            f"Done — {total_matched:,} papers matched out of {total_scanned:,} scanned. "
            f"Saved to {output_path}\n"
        )

    return total_matched, total_scanned


# ─────────────────────────────────────────────
# SCOPUS DATA COLLECTION
# ─────────────────────────────────────────────

_SCOPUS_FIELD_EXTRACTORS = {
    "eid":                lambda d: d.eid,
    "doi":                lambda d: d.doi,
    "title":              lambda d: d.title,
    "description":        lambda d: d.description,
    "coverDate":          lambda d: d.coverDate,
    "publicationName":    lambda d: d.publicationName,
    "aggregationType":    lambda d: d.aggregationType,
    "subtypeDescription": lambda d: d.subtypeDescription,
    "subtype":            lambda d: d.subtype,
    "authkeywords":       lambda d: d.authkeywords,
    "author_names":       lambda d: d.author_names,
    "author_ids":         lambda d: d.author_ids,
    "citedby_count":      lambda d: d.citedby_count,
    "openaccess":         lambda d: d.openaccess,
    "afid":               lambda d: d.afid,
    "affilname":          lambda d: d.affilname,
    "author_afids":       lambda d: d.author_afids,
    "pii":                lambda d: d.pii,
    "pubmed_id":          lambda d: d.pubmed_id,
    "issn":               lambda d: d.issn,
    "source_id":          lambda d: d.source_id,
}

# These fields are only populated when view="COMPLETE"; silently None with view="STANDARD".
_SCOPUS_COMPLETE_ONLY_FIELDS = {"description", "authkeywords"}


def scopus_doc_to_dict(doc, fields):
    """
    Extract requested fields from a pybliometrics Document namedtuple into a dict.
    Valid field names: see _SCOPUS_FIELD_EXTRACTORS keys.

    Raises:
        ValueError: If an unrecognised field name is requested.
    """
    unknown = set(fields) - _SCOPUS_FIELD_EXTRACTORS.keys()
    if unknown:
        raise ValueError(
            f"Unknown Scopus field(s) {sorted(unknown)}. "
            f"Valid fields: {sorted(_SCOPUS_FIELD_EXTRACTORS)}"
        )
    return {f: _SCOPUS_FIELD_EXTRACTORS[f](doc) for f in fields}


def collect_scopus_data(
    query,
    fields,
    output_path,
    date_start,
    date_end,
    api_key="",
    batch_threshold=4000,
    subscriber=True,
    view="COMPLETE",
    refresh=False,
    max_retries=5,
    backoff_factor=4,
):
    """
    Fetch Scopus papers via pybliometrics ScopusSearch and stream results to JSONL.

    Args:
        query (str): Scopus query string without the TITLE-ABS-KEY() wrapper — i.e.,
            the direct output of build_scopus_query(). The wrapper is added internally.
        fields (list[str]): Fields to retrieve per paper. Must be keys of
            _SCOPUS_FIELD_EXTRACTORS (e.g. config["scopus"]["fields_full"]).
            Note: "description" and "authkeywords" require view="COMPLETE".
        output_path (str): Path to the output .jsonl file.
        date_start: Start of date range. Accepts datetime.date or "YYYY-MM-DD" str.
        date_end: End of date range (inclusive). Accepts datetime.date or "YYYY-MM-DD" str.
        api_key (str): Scopus API key. Overrides the key in pybliometrics.cfg for this
            process if provided. If omitted, the configured key is used.
        batch_threshold (int): If the total result count exceeds this value, the
            function switches from a single query to year-by-year batching to limit
            peak RAM usage. Default 4000 (conservative buffer below the 5000-result
            hard cap for non-subscriber keys).
        subscriber (bool): Pass True if the API key has subscriber privileges. Enables
            cursor-based pagination, removing the 5000-result cap per query.
        view (str): ScopusSearch view — "COMPLETE" (default) or "STANDARD".
            "STANDARD" does not return description or authkeywords.
        refresh (bool | int): Passed directly to ScopusSearch. True forces a cache
            bypass; an integer n refreshes if the cached file is older than n days.
            Useful when re-running after a partial failure.
        max_retries (int): Retry attempts per batch on transient errors.
        backoff_factor (int): Exponential backoff base for server errors (e.g. 4 gives
            waits of 1s, 4s, 16s, 64s between retries).

    Returns:
        total_downloaded (int): Papers written to the output file (date-matched +
            unknown-date papers that could not be filtered).
        grand_total (int): Total papers reported by the API for the full query range.

    Notes:
        - Papers with coverDate=None cannot be date-filtered and are included in the
          output; their count is recorded in the metadata sidecar as unknown_date_count.
        - pybliometrics caches responses locally. Pass refresh=True to force fresh data.
        - Scopus only supports year-level filtering (PUBYEAR). Exact date boundaries
          are enforced by post-filtering on coverDate after download, so total_downloaded
          may legitimately be less than grand_total when date_start or date_end falls
          mid-year.
    """
    from pybliometrics.scopus import ScopusSearch
    from pybliometrics.utils import init
    from pybliometrics.exception import (
        Scopus429Error,
        Scopus401Error,
        Scopus403Error,
        ScopusQueryError,
        ScopusServerError,
    )

    # Initialise pybliometrics — must always be called, even if using the config-file key.
    if api_key:
        init(keys=[api_key])
    else:
        init()

    # Warn about COMPLETE-only fields requested with STANDARD view.
    incomplete_fields = _SCOPUS_COMPLETE_ONLY_FIELDS & set(fields)
    if view == "STANDARD" and incomplete_fields:
        print(
            f"WARNING: fields {sorted(incomplete_fields)} require view='COMPLETE' "
            f"but view='STANDARD' was requested — these fields will be None."
        )

    date_start_str = str(date_start)[:10]
    date_end_str = str(date_end)[:10]
    year_start = int(date_start_str[:4])
    year_end = int(date_end_str[:4])

    full_query = (
        f"TITLE-ABS-KEY({query}) "
        f"AND PUBYEAR > {year_start - 1} AND PUBYEAR < {year_end + 1}"
    )

    print(f"Query: {full_query}")
    print(f"Date range: {date_start_str} → {date_end_str}")

    # ── Pre-count ───────────────────────────────────────────────────────────
    counter = ScopusSearch(full_query, download=False, subscriber=subscriber)
    grand_total = counter.get_results_size()
    use_batches = grand_total > batch_threshold

    print(f"Total papers to retrieve: {grand_total}")
    print(f"Batching strategy: {'year-by-year' if use_batches else 'single query'}")

    # ── Build batch list ─────────────────────────────────────────────────────
    if not use_batches:
        batch_queries = [(None, full_query)]
    else:
        year_range = range(year_start, year_end + 1)
        year_queries = [
            (y, f"TITLE-ABS-KEY({query}) AND PUBYEAR IS {y}")
            for y in year_range
        ]

        # Preflight: check all years before downloading any data.
        print("Preflight: checking per-year counts...")
        offending = []
        for year, yq in year_queries:
            yc = ScopusSearch(yq, download=False, subscriber=subscriber).get_results_size()
            print(f"  {year}: {yc} papers")
            if yc > batch_threshold:
                offending.append({"year": year, "count": yc})

        if offending:
            error_info = {
                "type": "BatchThresholdExceeded",
                "message": (
                    f"The following years exceed batch_threshold={batch_threshold} "
                    f"and cannot be downloaded in a single year query. "
                    f"Consider increasing batch_threshold or narrowing the query."
                ),
                "offending_years": offending,
                "timestamp": datetime.now().isoformat(),
            }
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path = output_path.with_name(output_path.stem + "_meta.json")
            meta = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "fields": fields,
                "date_start": str(date_start),
                "date_end": str(date_end),
                "grand_total": grand_total,
                "total_retrieved": 0,
                "total_downloaded": 0,
                "unknown_date_count": 0,
                "use_batches": True,
                "batch_threshold": batch_threshold,
                "view": view,
                "subscriber": subscriber,
                "error": error_info,
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            print(f"ERROR: {error_info['message']} — see {meta_path}")
            return 0, grand_total

        batch_queries = year_queries

    # ── Download ─────────────────────────────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_retrieved = 0
    total_downloaded = 0
    unknown_date_count = 0
    error_info = None

    with open(output_path, "w", encoding="utf-8") as f:
        for year_label, batch_query in batch_queries:
            if year_label is not None:
                print(f"\nYear {year_label}")

            success = False
            for attempt in range(1, max_retries + 1):
                try:
                    print(f"  Fetching | total so far: {total_downloaded}")
                    search = ScopusSearch(
                        batch_query,
                        download=True,
                        view=view,
                        subscriber=subscriber,
                        verbose=True,
                        refresh=refresh,
                    )

                    docs = search.results or []
                    total_retrieved += len(docs)

                    for doc in docs:
                        if doc.coverDate is None:
                            f.write(json.dumps(scopus_doc_to_dict(doc, fields)) + "\n")
                            unknown_date_count += 1
                            total_downloaded += 1
                            continue
                        if not (date_start_str <= doc.coverDate <= date_end_str):
                            continue
                        f.write(json.dumps(scopus_doc_to_dict(doc, fields)) + "\n")
                        total_downloaded += 1

                    success = True
                    break

                except Scopus429Error:
                    print(f"  Rate limit hit [attempt {attempt}/{max_retries}] — sleeping 90s")
                    time.sleep(90)

                except ScopusServerError as err:
                    wait = backoff_factor ** (attempt - 1)
                    print(
                        f"  Server error [attempt {attempt}/{max_retries}]: {err} "
                        f"— retrying in {wait}s"
                    )
                    time.sleep(wait)

                except (Scopus401Error, Scopus403Error, ScopusQueryError) as err:
                    print(f"  Fatal error: {type(err).__name__}: {err}")
                    error_info = {
                        "type": type(err).__name__,
                        "message": str(err),
                        "attempt": attempt,
                        "batch_query": batch_query,
                        "timestamp": datetime.now().isoformat(),
                    }
                    break

                except Exception as err:
                    print(f"  Unexpected error: {err}")
                    error_info = {
                        "type": type(err).__name__,
                        "message": str(err),
                        "batch_query": batch_query,
                        "timestamp": datetime.now().isoformat(),
                    }
                    break

            if not success and error_info:
                break

    # ── Mismatch check ───────────────────────────────────────────────────────
    if grand_total and total_retrieved < grand_total:
        print(
            f"WARNING: API returned {total_retrieved} papers but reported {grand_total}. "
            f"Check the metadata sidecar for details."
        )

    # ── Metadata sidecar ─────────────────────────────────────────────────────
    meta_path = output_path.with_name(output_path.stem + "_meta.json")
    meta = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "fields": fields,
        "date_start": str(date_start),
        "date_end": str(date_end),
        "grand_total": grand_total,
        "total_retrieved": total_retrieved,
        "total_downloaded": total_downloaded,
        "unknown_date_count": unknown_date_count,
        "use_batches": use_batches,
        "batch_threshold": batch_threshold,
        "view": view,
        "subscriber": subscriber,
        "error": error_info,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if error_info:
        print(f"ERROR during collection — check {meta_path}")
    else:
        print(
            f"Done — {total_downloaded} papers saved to {output_path} "
            f"({unknown_date_count} with unknown date included)\n"
        )

    return total_downloaded, grand_total


# ─────────────────────────────────────────────
# WEB OF SCIENCE METADATA
# ─────────────────────────────────────────────

def create_wos_metadata(
    data_path,
    wos_query,
    date_start=None,
    date_end=None,
):
    """
    Create a metadata sidecar JSON for a manually downloaded WoS export file.

    WoS data collection is manual: the researcher runs the query in the WoS
    Advanced Search interface, exports the results as a tagged .txt file, and
    saves it locally. 

    Args:
        data_path (str): Path to the WoS tagged text export file (.txt).
        wos_query (str): The query string used in the WoS Advanced Search interface to generate the export file.
        date_start: Start of the search date range (optional). Accepts
            datetime.date or "YYYY-MM-DD" str. Used for documentation only.
        date_end: End of the search date range (optional). Same format.

    Returns:
        Path: Path to the written metadata JSON file.
    """
    import wosfile

    data_path = Path(data_path)

    n_papers = sum(1 for _ in wosfile.read(data_path))
    file_mtime = datetime.fromtimestamp(data_path.stat().st_mtime).isoformat()

    meta_path = data_path.with_name(data_path.stem + "_meta.json")
    meta = {
        "timestamp": file_mtime,
        "query": wos_query,
        "date_start": str(date_start) if date_start is not None else None,
        "date_end": str(date_end) if date_end is not None else None,
        "total_downloaded": n_papers,
        "data_path": str(data_path),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Done — {n_papers} papers found in {data_path}")
    print(f"Metadata saved to {meta_path}\n")

    return meta_path
