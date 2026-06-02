import re
from collections import Counter

import networkx as nx
import matplotlib.pyplot as plt
import wosfile


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


def build_citation_graph(
    data_path: str,
    include_external: bool = True,
) -> tuple[nx.DiGraph, set[str], dict]:
    """
    Build a directed citation graph from a WoS tagged-text export.

    Each corpus record becomes a node.  Directed edges run from citing
    paper to cited paper.  External papers (cited but not in the corpus)
    are added as nodes when include_external=True.

    CR entries with no extractable DOI are skipped — they cannot be
    reliably identified and would create misleading anonymous nodes.

    NOTE: uses wosfile.records_from(), which returns CR as list[str].
    Do not substitute wosfile.read(), which returns CR as a raw str.

    Parameters
    ----------
    data_path : str
        Path to the WoS .txt export file.
    include_external : bool
        If True (default), cited papers not in the corpus are also added
        as nodes (node_type="external"), enabling cross-corpus in-degree
        analysis.  If False, only corpus-to-corpus edges are kept.

    Returns
    -------
    G : nx.DiGraph
    corpus_node_ids : set[str]
    coverage : dict
        Keys: corpus_with_doi, corpus_without_doi, corpus_without_doi_ids,
              cr_total, cr_skipped_no_doi, cr_edges_added.
    """
    G = nx.DiGraph()
    records = list(wosfile.records_from(data_path))

    # ------------------------------------------------------------------ #
    # First pass: add all corpus nodes                                    #
    # ------------------------------------------------------------------ #
    doi_to_corpus_id: dict[str, str] = {}
    corpus_node_ids: set[str] = set()
    corpus_without_doi_ids: list[str] = []

    for rec in records:
        di = rec.get('DI')
        ut = rec.get('UT')
        if di:
            node_id = di.lower().strip()
            id_type = 'doi'
        else:
            node_id = ut.lower().strip() if ut else f'unknown_{id(rec)}'
            id_type = 'ut'
            corpus_without_doi_ids.append(ut or node_id)

        G.add_node(node_id, node_type='corpus', id_type=id_type,
                   ut=ut, py=rec.get('PY'))
        corpus_node_ids.add(node_id)
        if id_type == 'doi':
            doi_to_corpus_id[node_id] = node_id

    n_with_doi = len(corpus_node_ids) - len(corpus_without_doi_ids)
    if corpus_without_doi_ids:
        print(
            f"[citation_network] Corpus nodes: {n_with_doi} with DOI, "
            f"{len(corpus_without_doi_ids)} without DOI (using UT as node id): "
            f"{corpus_without_doi_ids}"
        )
    else:
        print(f"[citation_network] Corpus nodes: {n_with_doi} with DOI, 0 without DOI")

    # ------------------------------------------------------------------ #
    # Second pass: build edges from CR fields                             #
    # ------------------------------------------------------------------ #
    cr_total = 0
    cr_skipped = 0
    cr_edges_added = 0

    for rec in records:
        di = rec.get('DI')
        ut = rec.get('UT')
        citing_id = di.lower().strip() if di else (ut.lower().strip() if ut else None)
        if citing_id is None:
            continue

        for cr_entry in rec.get('CR') or []:
            cr_total += 1
            cited_id = parse_cr_doi(cr_entry)
            if cited_id is None:
                cr_skipped += 1
                continue

            if cited_id not in corpus_node_ids and include_external:
                G.add_node(cited_id, node_type='external')

            if cited_id in G:
                G.add_edge(citing_id, cited_id)
                cr_edges_added += 1

    pct_skipped = cr_skipped / cr_total * 100 if cr_total else 0
    print(
        f"[citation_network] CR entries processed: {cr_total}. "
        f"Skipped (no DOI): {cr_skipped} ({pct_skipped:.1f}%). "
        f"Edges added: {cr_edges_added}."
    )

    coverage = {
        'corpus_with_doi': n_with_doi,
        'corpus_without_doi': len(corpus_without_doi_ids),
        'corpus_without_doi_ids': corpus_without_doi_ids,
        'cr_total': cr_total,
        'cr_skipped_no_doi': cr_skipped,
        'cr_edges_added': cr_edges_added,
    }
    return G, corpus_node_ids, coverage


def compute_network_metrics(
    G: nx.DiGraph,
    corpus_node_ids: set[str],
    top_n: int = 10,
) -> dict:
    """
    Compute summary statistics for the citation graph.

    Parameters
    ----------
    G : nx.DiGraph
    corpus_node_ids : set[str]
    top_n : int
        Number of top-cited papers to include in ranked lists.

    Returns
    -------
    dict with keys:
        n_nodes, n_corpus_nodes, n_external_nodes, n_edges, density,
        in_degree_distribution, out_degree_distribution,
        top_cited, top_cited_corpus,
        n_weakly_connected_components, largest_wcc_size.

    Note: top_cited_corpus lists corpus nodes ranked by in-degree.
    With sparse internal citations (e.g. 10 corpus-corpus edges across 83
    papers), most corpus nodes will have in-degree 0 — this metric is most
    useful when the corpus has strong internal citation density.
    """
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())

    top_cited = sorted(in_deg.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_cited_corpus = sorted(
        ((n, d) for n, d in in_deg.items() if n in corpus_node_ids),
        key=lambda x: x[1],
        reverse=True,
    )[:top_n]

    wccs = list(nx.weakly_connected_components(G))

    return {
        'n_nodes': G.number_of_nodes(),
        'n_corpus_nodes': len(corpus_node_ids),
        'n_external_nodes': G.number_of_nodes() - len(corpus_node_ids),
        'n_edges': G.number_of_edges(),
        'density': nx.density(G),
        'in_degree_distribution': Counter(in_deg),
        'out_degree_distribution': Counter(out_deg),
        'top_cited': top_cited,
        'top_cited_corpus': top_cited_corpus,
        'n_weakly_connected_components': len(wccs),
        'largest_wcc_size': max(len(c) for c in wccs) if wccs else 0,
    }


def plot_citation_network(
    G: nx.DiGraph,
    corpus_node_ids: set[str],
    min_indegree: int = 2,
    top_n_labels: int = 10,
    figsize: tuple[int, int] = (14, 10),
    ax=None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Draw the citation network, filtered to nodes with in-degree >= min_indegree
    plus all corpus nodes.

    A full spring layout on thousands of nodes produces an illegible hairball.
    The filtered view (typically 50-200 nodes) is interpretable.

    Corpus nodes: blue.  External nodes: light gray.
    Node size proportional to in-degree (floor 50).
    Labels shown for the top_n_labels highest in-degree nodes only.
    """
    in_deg = dict(G.in_degree())
    keep = {n for n, d in in_deg.items() if d >= min_indegree} | corpus_node_ids
    sub = G.subgraph(keep).copy()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    pos = nx.spring_layout(sub, seed=42)

    sub_in_deg = dict(sub.in_degree())
    node_colors = ['#4C72B0' if n in corpus_node_ids else '#D3D3D3' for n in sub.nodes()]
    node_sizes = [50 + 20 * sub_in_deg.get(n, 0) for n in sub.nodes()]

    nx.draw_networkx_edges(sub, pos, ax=ax, alpha=0.3, arrows=True,
                           arrowsize=8, edge_color='#888888')
    nx.draw_networkx_nodes(sub, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.85)

    top_label_nodes = {n for n, _ in sorted(sub_in_deg.items(),
                                             key=lambda x: x[1], reverse=True)[:top_n_labels]}
    labels = {n: n[:30] for n in top_label_nodes}
    nx.draw_networkx_labels(sub, pos, labels=labels, ax=ax, font_size=7)

    ax.set_title(
        f"Citation network (in-degree ≥ {min_indegree} or corpus; "
        f"{sub.number_of_nodes()} nodes, {sub.number_of_edges()} edges)",
        fontsize=12,
    )
    ax.axis('off')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4C72B0', label='Corpus'),
        Patch(facecolor='#D3D3D3', label='External'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    plt.tight_layout()
    return fig, ax



