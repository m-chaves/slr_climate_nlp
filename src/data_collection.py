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
                    print(
                        f"Fetching batch | total so far: {total_downloaded}"
                    )

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
