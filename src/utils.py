from matplotlib_venn import venn3
import matplotlib.pyplot as plt


def venn_from_sizes(
    a: float,
    b: float,
    c: float,
    ab: float,
    ac: float,
    bc: float,
    abc: float,
    set_labels: tuple = ("A", "B", "C"),
    ax=None,
    **venn3_kwargs,
):
    """
    Draw a three-set Venn diagram from natural set-size inputs.

    Parameters
    ----------
    a, b, c : float
        Total sizes of the three sets |A|, |B|, |C|.
    ab : float
        Size of the pairwise intersection |A ∩ B|.
    ac : float
        Size of the pairwise intersection |A ∩ C|.
    bc : float
        Size of the pairwise intersection |B ∩ C|.
    abc : float
        Size of the triple intersection |A ∩ B ∩ C|.
    set_labels : tuple of three strings, optional
        Labels for the three sets. Default: ("A", "B", "C").
    ax : matplotlib Axes, optional
        Axes to draw on. If None, a new figure + axes is created.
    **venn3_kwargs
        Extra keyword arguments forwarded to venn3()
        (e.g. alpha=0.5, normalize_to=1.0, ...).

    Returns
    -------
    diagram : matplotlib_venn._venn3.Venn3
        The venn3 object — use .get_patch_by_id() / .get_label_by_id()
        to customise individual regions afterwards.
    ax : matplotlib Axes

    Raises
    ------
    ValueError
        If any derived exclusive-region count is negative, meaning the
        supplied sizes are logically inconsistent.

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> venn_from_sizes(
    ...     a=60, b=50, c=45,
    ...     ab=20, ac=15, bc=18, abc=8,
    ...     set_labels=("Python", "R", "Julia"),
    ...     ax=ax, alpha=0.6,
    ... )
    >>> plt.show()
    """
    # ------------------------------------------------------------------ #
    # Compute exclusive region sizes via inclusion-exclusion              #
    # ------------------------------------------------------------------ #
    region_ABC = abc
    region_ABc = ab  - abc          # A∩B but not C
    region_AbC = ac  - abc          # A∩C but not B
    region_aBC = bc  - abc          # B∩C but not A
    region_Abc = a   - ab - ac + abc   # only A
    region_aBc = b   - ab - bc + abc   # only B
    region_abC = c   - ac - bc + abc   # only C

    # Sanity check: no region may be negative
    regions = {
        "Abc (only A)"  : region_Abc,
        "aBc (only B)"  : region_aBc,
        "abC (only C)"  : region_abC,
        "ABc (A∩B excl C)": region_ABc,
        "AbC (A∩C excl B)": region_AbC,
        "aBC (B∩C excl A)": region_aBC,
        "ABC (A∩B∩C)"   : region_ABC,
    }
    negative = {k: v for k, v in regions.items() if v < 0}
    if negative:
        raise ValueError(
            "Logically inconsistent inputs — the following exclusive "
            f"region counts came out negative:\n  {negative}\n"
            "Verify that abc ≤ ab, ac, bc and that each pairwise "
            "intersection is ≤ the smaller of its two parent sets."
        )

    # venn3 expects exactly this tuple order:
    #   (Abc, aBc, ABc, abC, AbC, aBC, ABC)
    subsets = (
        region_Abc,
        region_aBc,
        region_ABc,
        region_abC,
        region_AbC,
        region_aBC,
        region_ABC,
    )

    # ------------------------------------------------------------------ #
    # Draw                                                                #
    # ------------------------------------------------------------------ #
    if ax is None:
        _, ax = plt.subplots()

    diagram = venn3(subsets=subsets, set_labels=set_labels, ax=ax, **venn3_kwargs)
    return diagram, ax


def venn_from_topics(topics, colors, sizes):
    """
    Draw a 3-set Venn diagram for a given triple of topic names.

    Parameters
    ----------
    topics : tuple of 3 strings
        The three set names, e.g. ("climate", "nlp", "misinformation").
    colors : dict
        Maps topic name -> matplotlib color string.
        for instance, {"climate": "green", "nlp": "orange", "misinformation": "red"}.
    sizes : dict
        Maps topic name or "X AND Y" / "X AND Y AND Z" -> integer count.
        Keys use " AND " as separator; order within a key does not matter.
        For instance, {"climate": 100, "nlp": 80, "misinformation": 50, "climate AND nlp": 30, ...}.

    Returns
    -------
    diagram, ax
    """
    if len(topics) != 3:
        raise ValueError("topics must contain exactly 3 names.")

    a_name, b_name, c_name = topics

    def get_size(key_topics):
        """Look up a single or multi-topic size, trying all orderings."""
        if len(key_topics) == 1:
            return sizes[key_topics[0]]
        # Build all permutations of the AND-joined key
        from itertools import permutations
        for perm in permutations(key_topics):
            key = " AND ".join(perm)
            if key in sizes:
                return sizes[key]
        raise KeyError(
            f"Could not find size for topics {key_topics} in any order. "
            f"Tried keys like '{' AND '.join(key_topics)}'."
        )

    a   = get_size([a_name])
    b   = get_size([b_name])
    c   = get_size([c_name])
    ab  = get_size([a_name, b_name])
    ac  = get_size([a_name, c_name])
    bc  = get_size([b_name, c_name])
    abc = get_size([a_name, b_name, c_name])

    set_colors = (
        colors.get(a_name, "blue"),
        colors.get(b_name, "blue"),
        colors.get(c_name, "blue"),
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    diagram, ax = venn_from_sizes(
        a=a, b=b, c=c,
        ab=ab, ac=ac, bc=bc, abc=abc,
        set_labels=topics,
        ax=ax,
        set_colors=set_colors,
        alpha=0.6,
    )
    ax.set_title(" ∩ ".join(topics), fontsize=13, pad=12)
    plt.tight_layout()
    return diagram, ax