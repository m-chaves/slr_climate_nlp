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

    print("Function version 5")

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

                    # First batch only
                    if total_downloaded == 0:
                        grand_total = payload.get("total", "unknown")
                        print(
                            f"Total papers to retrieve: {grand_total}"
                        )

                    papers_batch = payload.get("data", [])

                    for paper in papers_batch:
                        f.write(json.dumps(paper) + "\n")

                    total_downloaded += len(papers_batch)

                    # pagination
                    next_token = payload.get("token")

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
                    retryable = status_code in [429, 500]

                    print(
                        f"HTTP error ({status_code}) "
                        f"[attempt {attempt}/{max_retries}]"
                    )

                    if retryable and attempt < max_retries:

                        if status_code == 429:
                            print("Rate limit hit — sleeping 60s")
                            time.sleep(60)
                            continue
                        elif status_code == 500:
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

                    print(
                        f"Network error [attempt {attempt}/{max_retries}]: {err}"
                    )

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

            # stop outer loop if all retries failed
            if not success and error_info:
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

    print(
        f"Done — {total_downloaded} papers saved to {output_path}\n"
    )

    return total_downloaded, grand_total