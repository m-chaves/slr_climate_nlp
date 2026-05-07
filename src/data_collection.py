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


# def load_api_key(env_path="../.env"):
#     load_dotenv(env_path)
#     return os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")

# ─────────────────────────────────────────────
# BASE EXTRACTION FUNCTION
# ─────────────────────────────────────────────

def collect_semantic_scholar_data(query, fields, output_path, date_start, date_end, api_key="", base_url=None):
    """
    Base extraction function. Fetches all pages for a given query
    and writes results directly to a JSONL file to preserve RAM.

    Args:
        query:       Boolean query string (SS syntax)
        fields:      List of fields to retrieve
        output_path: Path to output .jsonl file
        date_start:  Start date (datetime)
        date_end:    End date (datetime)
        api_key:     Semantic Scholar API key (optional)
        base_url:    API endpoint URL
    
    Returns:
        total_downloaded (int)
    """
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"

    headers = {"x-api-key": api_key} if api_key else {}
    sleep_time = 1.1 if api_key else 3.5

    query_params = {
        "query": query,
        "publicationDateOrYear": f"{date_start}:{date_end}",
        "fields": ",".join(fields),
    }

    print(query_params)

    # Open a local file to safely stream data onto your hard drive line-by-line
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Query: {query}")  
    print(f"Output: {output_path}")
    print(f"API key: {'yes' if api_key else 'no'} | Sleep: {sleep_time}s")

    total_downloaded = 0
    has_more = True

    with open(output_path, "w", encoding="utf-8") as f:
        while has_more:
            try:
                print(f"  Fetching batch | total so far: {total_downloaded}")

                response = requests.get(
                    base_url,
                    params=query_params,
                    headers=headers
                )

                # Handle rate limiting dynamically if the server pushes back
                if response.status_code == 429:
                    print("  Rate limit hit — sleeping 60s")
                    time.sleep(60)
                    continue

                response.raise_for_status()
                payload = response.json()

                papers_batch = payload.get("data", [])

                # write batch to disk immediately, then free memory
                for paper in papers_batch:
                    f.write(json.dumps(paper) + "\n")
                
                total_downloaded += len(papers_batch)
                papers_batch.clear()  # free memory immediately

                # Check for the token to see if there is another page
                next_token = payload.get("token")
                if next_token:
                    query_params["token"] = next_token
                else:
                    has_more = False

                # Rest before the next loop iteration
                time.sleep(sleep_time)

            except requests.exceptions.HTTPError as err:
                print(f"  HTTP error: {err}")
                break
            except Exception as e:
                print(f"  Unexpected error: {e}")
                break
    
    # save metadata sidecar file with query info and stats about the download
    if total_downloaded > 0:
        meta_path = output_path.with_name(output_path.stem + "_meta.json")
        meta = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "fields": fields,
            "date_start": str(date_start),
            "date_end": str(date_end),
            "api_key_used": bool(api_key),
            "total_downloaded": total_downloaded,
            "base_url": base_url
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    
    print(f"  Done — {total_downloaded} papers saved to {output_path}\n")
    return total_downloaded