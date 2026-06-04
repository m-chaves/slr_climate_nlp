"""
Graph-based deduplication and harmonization of multi-source paper records.

Algorithm overview
------------------
Each NormalizedRecord (one paper from one source) becomes a node in an
undirected graph.  An edge between two nodes means "same paper".  Four
stages add edges in decreasing confidence order:

  Stage 1 — definitive DOI match        (exact, non-arXiv DOIs)
  Stage 2 — ACL Anthology ID match      (exact; ACL and Semantic Scholar only)
  Stage 3 — arXiv ID match              (exact, canonical arXiv IDs)
  Stage 4 — title + year fuzzy          (normalized title, ±year_tolerance)

Bijection constraint (enforced at component level before every edge):
  Each connected component may contain at most one node per source type.
  This is checked via nx.node_connected_component(), not just direct
  neighbors, so transitive violations are also prevented.

After all edges are added, connected components become MergedRecord dicts.

Public API
----------
  build_merge_graph(records)   → nx.Graph
  minimal_merge(...)           → (list[MergedRecord], nx.Graph)
"""

import hashlib
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
from rapidfuzz import fuzz

from src.data_loading import DEFAULT_DATA_DIR, load_all_sources
from src.data_normalization import normalize_title_for_matching, normalize_published_in, normalize_paper_type, clean_abstract

logger = logging.getLogger(__name__)

# Source priority used for deterministic coalescing and sorting.
# Lower index = higher priority.
_SOURCE_PRIORITY: list[str] = ["scopus", "wos", "semantic_scholar", "acl", "acm"]


def _source_rank(source: str) -> int:
    try:
        return _SOURCE_PRIORITY.index(source)
    except ValueError:
        return len(_SOURCE_PRIORITY)


# ──────────────────────────────────────────────────────────────────────────────
# Bijection constraint
# ──────────────────────────────────────────────────────────────────────────────

def _can_add_edge(G: nx.Graph, u: str, v: str) -> bool:
    """
    Return True if edge (u, v) satisfies the component-level bijection constraint.

    Checks that:
      - u and v are from different sources (same-source edges are never added)
      - The component of u contains no node of v's source type
      - The component of v contains no node of u's source type

    This prevents both direct and transitive source-type conflicts within
    any connected component.
    """
    u_source = G.nodes[u]["_source"]
    v_source = G.nodes[v]["_source"]
    if u_source == v_source:
        return False
    comp_u = nx.node_connected_component(G, u)
    comp_v = nx.node_connected_component(G, v)
    if any(G.nodes[n]["_source"] == v_source for n in comp_u):
        return False
    if any(G.nodes[n]["_source"] == u_source for n in comp_v):
        return False
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Stage functions
# ──────────────────────────────────────────────────────────────────────────────

def _try_add_edge(G: nx.Graph, u: str, v: str, stage: int, key, label: str) -> bool:
    """
    Attempt to add edge (u, v) with the given stage and match key.

    Returns True if the edge was added. Silent skip if:
      - same source
      - already directly connected
      - already in the same connected component (redundant edge)
    Logs a WARNING only for a true bijection conflict (different components,
    different sources, but component already contains the other's source type).
    """
    if G.nodes[u]["_source"] == G.nodes[v]["_source"]:
        return False
    if G.has_edge(u, v):
        return False
    if v in nx.node_connected_component(G, u):
        return False  # already merged via another path — redundant, skip silently
    if _can_add_edge(G, u, v):
        G.add_edge(u, v, match_stage=stage, match_key=key)
        return True
    logger.warning(
        "%s: bijection conflict — cannot link %s and %s (component source clash)",
        label, u, v,
    )
    return False


def _stage1_doi(G: nx.Graph) -> int:
    """Add edges for nodes sharing the same non-arXiv DOI. Returns edge count."""
    doi_index: dict[str, list[str]] = defaultdict(list)
    for node_id, attrs in G.nodes(data=True):
        doi = attrs.get("doi")
        if doi:
            doi_index[doi].append(node_id)

    added = 0
    for doi, nodes in doi_index.items():
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if _try_add_edge(G, nodes[i], nodes[j], 1, doi, "Stage 1"):
                    added += 1
    return added


def _stage2_acl(G: nx.Graph) -> int:
    """
    Add edges for nodes sharing the same ACL Anthology ID. Returns edge count.

    Only ACL and Semantic Scholar records carry an acl_anthology_id; all
    other sources will simply have no entry in the index.
    """
    acl_index: dict[str, list[str]] = defaultdict(list)
    for node_id, attrs in G.nodes(data=True):
        acl_id = attrs.get("acl_anthology_id")
        if acl_id:
            acl_index[acl_id].append(node_id)

    added = 0
    for acl_id, nodes in acl_index.items():
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if _try_add_edge(G, nodes[i], nodes[j], 2, acl_id, "Stage 2"):
                    added += 1
    return added


def _stage3_arxiv(G: nx.Graph) -> int:
    """Add edges for nodes sharing the same arXiv ID. Returns edge count."""
    arxiv_index: dict[str, list[str]] = defaultdict(list)
    for node_id, attrs in G.nodes(data=True):
        arxiv_id = attrs.get("arxiv_id")
        if arxiv_id:
            arxiv_index[arxiv_id].append(node_id)

    added = 0
    for arxiv_id, nodes in arxiv_index.items():
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if _try_add_edge(G, nodes[i], nodes[j], 3, arxiv_id, "Stage 3"):
                    added += 1
    return added


def _stage4_title_year(
    G: nx.Graph,
    fuzzy_threshold: int = 95,
    year_tolerance: int = 1,
) -> int:
    """
    Add edges for nodes sharing a similar title within a year window.

    Guards:
      - Skip if both nodes have non-None doi values that differ
        (conflicting high-confidence identifiers → different papers)
      - Skip if normalized title length < 10 chars (too short for reliable match)
      - year=None nodes: exact title match only

    Pairs are enumerated sorted by fuzzy score DESC for determinism.
    The component-level bijection check is applied before each edge addition.
    Returns edge count added.
    """
    # Collect candidate nodes with their normalized titles
    nodes_data: list[tuple[str, dict]] = list(G.nodes(data=True))

    # Build year buckets with ±tolerance expansion
    # Map: year → list of node_ids in that year window
    year_buckets: dict[int | None, list[str]] = defaultdict(list)
    for node_id, attrs in nodes_data:
        year_buckets[attrs.get("year")].append(node_id)

    # Collect all candidate pairs within year tolerance
    candidate_pairs: list[tuple[float, str, str]] = []  # (score, u, v)

    all_years = sorted((y for y in year_buckets if y is not None))
    year_none_nodes = year_buckets.get(None, [])

    # Pairs within year tolerance window (including None-year exact-only)
    checked: set[frozenset] = set()

    def _add_candidates(nodes_a: list[str], nodes_b: list[str], exact_only: bool) -> None:
        for i, u in enumerate(nodes_a):
            u_attrs = G.nodes[u]
            u_title = normalize_title_for_matching(u_attrs.get("title"))
            if not u_title:
                continue
            start_j = i + 1 if nodes_a is nodes_b else 0
            for v in (nodes_b[start_j:] if nodes_a is nodes_b else nodes_b):
                key = frozenset([u, v])
                if key in checked:
                    continue
                checked.add(key)
                if G.has_edge(u, v):
                    continue
                if u_attrs["_source"] == G.nodes[v]["_source"]:
                    continue
                v_attrs = G.nodes[v]
                # Guard: conflicting definitive DOIs → skip
                u_doi = u_attrs.get("doi")
                v_doi = v_attrs.get("doi")
                if u_doi and v_doi and u_doi != v_doi:
                    continue
                v_title = normalize_title_for_matching(v_attrs.get("title"))
                if not v_title:
                    continue
                if len(u_title) < 10 or len(v_title) < 10:
                    if u_title == v_title:
                        candidate_pairs.append((100.0, u, v))
                    continue
                if exact_only:
                    if u_title == v_title:
                        candidate_pairs.append((100.0, u, v))
                else:
                    score = fuzz.ratio(u_title, v_title)
                    if score >= fuzzy_threshold:
                        candidate_pairs.append((score, u, v))

    # Within-year matches
    for year in all_years:
        nodes_in_year = year_buckets[year]
        _add_candidates(nodes_in_year, nodes_in_year, exact_only=False)

    # Cross-year matches within tolerance window
    for idx, year in enumerate(all_years):
        for offset in range(1, year_tolerance + 1):
            neighbor_year = year + offset
            if neighbor_year in year_buckets:
                _add_candidates(year_buckets[year], year_buckets[neighbor_year], exact_only=False)

    # year=None: exact only
    _add_candidates(year_none_nodes, year_none_nodes, exact_only=True)
    for year in all_years:
        _add_candidates(year_none_nodes, year_buckets[year], exact_only=True)

    # Sort by score DESC, then source priorities for determinism
    candidate_pairs.sort(
        key=lambda t: (
            -t[0],
            _source_rank(G.nodes[t[1]]["_source"]),
            _source_rank(G.nodes[t[2]]["_source"]),
        )
    )

    added = 0
    for score, u, v in candidate_pairs:
        if _try_add_edge(G, u, v, 4, score, "Stage 4"):
            logger.info(
                "Stage 4 match (score=%.1f, years=%s/%s): '%s' | '%s'",
                score,
                G.nodes[u].get("year"),
                G.nodes[v].get("year"),
                G.nodes[u].get("title", "")[:60],
                G.nodes[v].get("title", "")[:60],
            )
            added += 1
    return added


# ──────────────────────────────────────────────────────────────────────────────
# Graph construction
# ──────────────────────────────────────────────────────────────────────────────

def build_merge_graph(
    records: list[dict],
    fuzzy_threshold: int = 95,
    year_tolerance: int = 1,
) -> nx.Graph:
    """
    Build the merge graph from a flat list of NormalizedRecord dicts.

    Nodes are NormalizedRecords; edges assert "same paper".
    Runs Stage 1 (DOI) → Stage 2 (arXiv ID) → Stage 3 (title+year).
    All stages operate on the full graph — nodes matched in Stage 1 can
    still receive Stage 2 or Stage 3 edges to nodes of other source types.
    """
    G = nx.Graph()

    for rec in records:
        node_id = rec["_node_id"]
        G.add_node(node_id, **rec)

    n1 = _stage1_doi(G)
    n2 = _stage2_acl(G)
    n3 = _stage3_arxiv(G)
    n4 = _stage4_title_year(G, fuzzy_threshold=fuzzy_threshold, year_tolerance=year_tolerance)

    logger.info(
        "Merge graph: %d nodes, %d edges (doi=%d, acl=%d, arxiv=%d, title=%d)",
        G.number_of_nodes(), G.number_of_edges(), n1, n2, n3, n4,
    )
    return G


# ──────────────────────────────────────────────────────────────────────────────
# Component → MergedRecord
# ──────────────────────────────────────────────────────────────────────────────

# Field coalescing priorities (source with highest priority wins)
_FIELD_PRIORITY: dict[str, list[str]] = {
    "doi":        ["wos", "scopus", "semantic_scholar", "acl", "acm"],
    "arxiv_id":   ["semantic_scholar", "scopus", "wos", "acl", "acm"],
    "title":      ["scopus", "wos", "semantic_scholar", "acl", "acm"],
    "abstract":   ["scopus", "semantic_scholar", "wos", "acl", "acm"],
    "year":       ["wos", "scopus", "semantic_scholar", "acl", "acm"],
    "paper_type": ["scopus", "wos", "semantic_scholar", "acl", "acm"],
}

# Regex to detect Scopus structured-abstract headers
_STRUCTURED_ABSTRACT_RE = r"\b(Purpose|Findings|Design/methodology|Originality|Practical implications)\s*[:/]"

import re as _re


def _coalesce(nodes_by_source: dict[str, dict], field: str) -> tuple:
    """
    Return (selected_value, conflict_dict | None) for `field`.

    Iterates sources in field priority order, picks first non-None value.
    Records a conflict when ≥2 sources have different non-None values.
    """
    import unicodedata as _ud

    priority = _FIELD_PRIORITY.get(field, _SOURCE_PRIORITY)
    values: list[dict] = []
    for src in priority:
        node = nodes_by_source.get(src)
        if node is None:
            continue
        raw_val = node.get(field)
        if raw_val is None:
            continue
        values.append({"source": src, "value": raw_val})

    if not values:
        return None, None

    selected = values[0]["value"]
    if len(values) == 1:
        return selected, None

    # Classify conflict
    def _norm(v):
        if not isinstance(v, str):
            return v
        s = _ud.normalize("NFKD", v).encode("ascii", "ignore").decode().lower().strip()
        return _re.sub(r"\s+", " ", s)

    normed = [_norm(e["value"]) for e in values]
    if len(set(normed)) == 1:
        conflict_type = "encoding"   # same after normalization
    elif len(set(str(e["value"]).lower() for e in values)) == 1:
        conflict_type = "case"
    elif field == "abstract" and any(
        _re.search(_STRUCTURED_ABSTRACT_RE, str(e["value"]), _re.IGNORECASE)
        for e in values
    ):
        conflict_type = "formatting"
    else:
        conflict_type = "content"

    if conflict_type == "encoding":
        return selected, None  # encoding variants are not real conflicts

    conflict = {
        "values": values,
        "selected": selected,
        "conflict_type": conflict_type,
    }
    return selected, conflict


def _published_in_from_raw(raw: dict, source: str) -> str | None:
    if source == "scopus":
        return raw.get("publicationName")
    if source == "wos":
        return raw.get("SO")
    if source == "semantic_scholar":
        pv = raw.get("publicationVenue") or {}
        pv_name = pv.get("name") if isinstance(pv, dict) else None
        if pv_name:
            return pv_name
        journal = raw.get("journal") or {}
        return journal.get("name") if isinstance(journal, dict) else None
    if source == "acl":
        return raw.get("volume_title") or raw.get("journal_title")
    return None


def _paper_type_from_raw(raw: dict, source: str) -> str | None:
    if source == "scopus":
        return raw.get("subtypeDescription")
    if source == "wos":
        return raw.get("DT")
    if source == "semantic_scholar":
        pt = raw.get("publicationTypes")
        if isinstance(pt, list) and pt:
            return pt[0]
        return None
    if source == "acl":
        return raw.get("paper_type")
    return None


def _generate_paper_id(nodes_by_source: dict[str, dict]) -> str:
    """Deterministic 16-char SHA-256 id from the highest-priority identifier."""
    for src in ["wos", "scopus", "semantic_scholar", "acl", "acm"]:
        node = nodes_by_source.get(src)
        if node is None:
            continue
        doi = node.get("doi")
        if doi:
            key = f"doi:{doi}"
            break
        wos_id = node.get("wos_id")
        if wos_id:
            key = f"wos:{wos_id}"
            break
        eid = node.get("scopus_eid")
        if eid:
            key = f"scopus:{eid}"
            break
        s2id = node.get("semantic_scholar_id")
        if s2id:
            key = f"s2:{s2id}"
            break
        acl_id = node.get("acl_id")
        if acl_id:
            key = f"acl:{acl_id}"
            break
    else:
        # fallback: hash all node ids together
        key = "|".join(sorted(nodes_by_source.keys()))
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _merge_component(component: set[str], G: nx.Graph) -> dict:
    """Merge one connected component into a MergedRecord dict."""
    nodes = sorted(component, key=lambda n: _source_rank(G.nodes[n]["_source"]))
    nodes_by_source: dict[str, dict] = {}
    for node_id in nodes:
        attrs = G.nodes[node_id]
        src = attrs["_source"]
        if src in nodes_by_source:
            logger.warning(
                "Intra-source duplicate in component: two '%s' nodes (%s, %s). "
                "Using first encountered.",
                src, nodes_by_source[src]["_node_id"], node_id,
            )
        else:
            nodes_by_source[src] = attrs

    conflicts: dict = {}
    merged: dict = {}
    for field in ("doi", "arxiv_id", "title", "abstract", "year"):
        val, conflict = _coalesce(nodes_by_source, field)
        if field == "abstract":
            val = clean_abstract(val)
        merged[field] = val
        if conflict:
            conflicts[field] = conflict
            logger.debug("Conflict on '%s': %s", field, conflict)

    # Published-in and paper type (unnormalized stubs)
    published_in_candidates = {
        src: normalize_published_in(_published_in_from_raw(node["_raw"], src), src)
        for src, node in nodes_by_source.items()
    }
    paper_type_candidates = {
        src: normalize_paper_type(_paper_type_from_raw(node["_raw"], src), src)
        for src, node in nodes_by_source.items()
    }

    return {
        "paper_id":             _generate_paper_id(nodes_by_source),
        "wos_id":               next((n.get("wos_id") for n in nodes_by_source.values() if n.get("wos_id")), None),
        "acl_id":               next((n.get("acl_id") for n in nodes_by_source.values() if n.get("acl_id")), None),
        "scopus_eid":           next((n.get("scopus_eid") for n in nodes_by_source.values() if n.get("scopus_eid")), None),
        "semantic_scholar_id":  next((n.get("semantic_scholar_id") for n in nodes_by_source.values() if n.get("semantic_scholar_id")), None),
        "doi":       merged["doi"],
        "arxiv_id":  merged["arxiv_id"],
        "title":     merged["title"],
        "abstract":  merged["abstract"],
        "year":      merged["year"],
        "published_in": {
            src: val
            for src, val in sorted(published_in_candidates.items(), key=lambda x: _source_rank(x[0]))
            if val
        },
        "paper_type": {
            src: val
            for src, val in sorted(paper_type_candidates.items(), key=lambda x: _source_rank(x[0]))
            if val
        },
        "sources":   sorted(nodes_by_source.keys(), key=_source_rank),
        "_conflicts": conflicts,
        "_raws":     {src: node["_raw"] for src, node in nodes_by_source.items()},
    }


# ──────────────────────────────────────────────────────────────────────────────
# Top-level entry point
# ──────────────────────────────────────────────────────────────────────────────

def minimal_merge(
    records: list[dict] | None = None,
    data_dir: Path = DEFAULT_DATA_DIR,
    output_path: Path | None = None,
    fuzzy_threshold: int = 95,
    year_tolerance: int = 1,
) -> tuple[list[dict], nx.Graph]:
    """
    Deduplicate and harmonize records from all sources.

    If `records` is None, loads all sources from `data_dir`.
    Writes the merged output to `output_path` (JSONL) and a sidecar
    `_meta.json` with deduplication statistics and input file hashes.

    Returns (merged_records, merge_graph).  The graph is returned for
    visualization and inspection in the notebook.
    """
    if output_path is None:
        output_path = Path(data_dir) / "processed" / "merged_minimal.jsonl"
    output_path = Path(output_path)

    if records is None:
        records = load_all_sources(data_dir=data_dir)

    G = build_merge_graph(records, fuzzy_threshold=fuzzy_threshold, year_tolerance=year_tolerance)

    # Extract connected components and merge each
    components = list(nx.connected_components(G))
    merged = [_merge_component(comp, G) for comp in components]
    merged.sort(key=lambda r: r["paper_id"])

    # Statistics
    stage_edges = Counter(G.edges[u, v]["match_stage"] for u, v in G.edges()) if G.edges() else Counter()
    singletons = sum(1 for comp in components if len(comp) == 1)
    conflict_counts: Counter = Counter()
    content_conflicts = 0
    for rec in merged:
        for field, c in rec.get("_conflicts", {}).items():
            conflict_counts[field] += 1
            if c.get("conflict_type") == "content":
                content_conflicts += 1

    # Input file hashes for reproducibility
    import hashlib as _hl
    raw_dir = Path(data_dir) / "raw"
    source_files = {
        "scopus":           raw_dir / "scopus" / "scopus_data.jsonl",
        "acl":              raw_dir / "acl" / "acl_data.jsonl",
        "semantic_scholar": raw_dir / "semantic_scholar" / "semantic_scholar_data.jsonl",
        "wos":              raw_dir / "wos" / "wos_data.txt",
    }
    source_hashes = {}
    for name, path in source_files.items():
        if path.exists():
            h = _hl.md5(path.read_bytes()).hexdigest()
            source_hashes[name] = h

    meta = {
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "n_input_records":  len(records),
        "n_output_records": len(merged),
        "n_doi_edges":        stage_edges.get(1, 0),
        "n_acl_edges":        stage_edges.get(2, 0),
        "n_arxiv_edges":      stage_edges.get(3, 0),
        "n_title_year_edges": stage_edges.get(4, 0),
        "n_singletons":     singletons,
        "fuzzy_threshold":  fuzzy_threshold,
        "year_tolerance":   year_tolerance,
        "n_conflicts_by_field": dict(conflict_counts),
        "n_content_conflicts":  content_conflicts,
        "source_hashes":    source_hashes,
    }

    # Write outputs
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for rec in merged:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    meta_path = output_path.with_name(output_path.stem + "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

    logger.info(
        "minimal_merge: %d input → %d output records. "
        "Edges: doi=%d acl=%d arxiv=%d title=%d. Singletons=%d. Content conflicts=%d.",
        len(records), len(merged),
        meta["n_doi_edges"], meta["n_acl_edges"], meta["n_arxiv_edges"], meta["n_title_year_edges"],
        singletons, content_conflicts,
    )
    return merged, G
