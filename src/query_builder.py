# This module provides functions to build boolean search queries from a logic expression and keyword groups defined in a YAML file. 
# 
# For example, given a logic string like "climate AND nlp", and a YAML file defining groups like:
# climate:
#   - "climate change"
#   - "global warming"
# nlp:
#   - "natural language processing"
#   - "NLP"
# 
# The module will produce a query source-agnostic string like:
# ("climate change" OR "global warming") AND ("natural language processing" OR NLP
# 
# Then we can convert this generic query to a source-specific syntax, such as Semantic Scholar's, ACL Anthology's, etc.


import yaml
import re
from pathlib import Path


# ─────────────────────────────────────────────
# KEYWORD LOADING
# ─────────────────────────────────────────────

def load_keywords(keywords_path):
    """Load all keyword groups from a single YAML file."""
    with open(keywords_path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
# KEYWORD FORMATTING (source-agnostic)
# ─────────────────────────────────────────────

def format_keyword(keyword):
    """
    Wrap multi-word keywords or those with wildcards in quotes.
    Single words without wildcards are left bare.
    Example:
        "climate change" -> '"climate change"'
        "climate" -> 'climate'
        "climat*" -> 'climat*'
    """
    if " " in keyword or "*" in keyword or "?" in keyword:
        return f'"{keyword}"'
    return keyword


def group_to_or_clause(keywords):
    """
    Format a list of keywords as an OR clause.
    Example: ("climate change" OR "global warming" OR "climate denial")
    """
    formatted = [format_keyword(kw) for kw in keywords]
    if len(formatted) == 1:
        return formatted[0]
    inner = " OR ".join(formatted)
    return f"({inner})"


# ─────────────────────────────────────────────
# LOGIC PARSING AND VALIDATION (source-agnostic)
# ─────────────────────────────────────────────

def tokenize_logic_string(logic_string):
    """
    Split logic string into tokens: AND, OR, (, ), group names.
    Example: "climate AND (misinformation OR polarization)" -> ["climate", "AND", "(", "misinformation", "OR", "polarization", ")"]
    """
    # split on spaces and parentheses, keeping parentheses as tokens
    tokens = re.findall(r'\(|\)|[^\s()]+', logic_string)
    return tokens


def validate_and_expand(tokens, groups):
    """
    Validate each token and replace group names with their formatted keyword clauses.
    Raises ValueError if an unrecognized token is found. For instance, if the logic string contains a group name that is not defined in the groups dictionary. 
    """
    valid_operators = {"AND", "OR"}
    valid_punctuation = {"(", ")"}
    expanded = []

    for token in tokens:
        if token in valid_operators or token in valid_punctuation:
            expanded.append(token)
        elif token in groups:
            expanded.append(group_to_or_clause(groups[token]))
        else:
            raise ValueError(
                f"Unrecognized token '{token}'. "
                f"Expected AND, OR, parentheses, or one of: {list(groups.keys())}"
            )

    return expanded


def build_query(logic_string, groups):
    """
    Build a source-agnostic query string from a logic expression and keyword groups.
    Uses plain AND/OR operators.

    Example:
        logic_string = "climate AND nlp" 
        where the group "climate" has keywords ["climate change", "global warming"] and "nlp" has ["natural language processing", "NLP"]
        then the output is: ("climate change" OR "global warming") AND ("natural language processing" OR NLP)
    """
    tokens = tokenize_logic_string(logic_string)
    expanded = validate_and_expand(tokens, groups)
    return " ".join(expanded)


# ─────────────────────────────────────────────
# SOURCE-SPECIFIC CONVERSION
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# Scopus QUERY BUILDING
# ─────────────────────────────────────────────

def to_scopus(query_string):
    """
    Convert a source-agnostic AND/OR query string to Scopus syntax.
    Scopus uses AND/OR natively so no operator conversion is needed.
    Two characters require special handling:
    - / is a reserved proximity operator (e.g. bare Twitter/X is parsed as Twitter W/X).
    - - (hyphen) inside quoted phrases is treated as a space ("alt-tech" matches "alt tech").
    Any token or quoted phrase containing / or - is wrapped in curly braces,
    which forces Scopus to match the string exactly as written.

    Examples:
        Twitter/X    -> {Twitter/X}
        alt-tech     -> {alt-tech}
        "fake-news"  -> {fake-news}
        "fake news"  -> "fake news"  (unchanged)
        {Twitter/X}  -> {Twitter/X}  (already braced, unchanged)
    """
    def maybe_wrap(m):
        token = m.group(0)
        if '-' not in token and '/' not in token:
            return token
        if token.startswith('"') and token.endswith('"'):
            return '{' + token[1:-1] + '}'
        if not token.startswith('{'):
            return '{' + token + '}'
        return token

    # Match quoted phrases, already-braced tokens, or bare tokens — in that priority order
    return re.sub(r'"[^"]*"|{[^}]*}|[^\s()]+', maybe_wrap, query_string)

def build_scopus_query(logic_string, keywords_path):
    """
    Full pipeline: load keywords, build generic query, convert to Scopus syntax.
    Parameters:
    - logic_string: a string like "climate AND nlp AND (misinformation OR polarization)"
    - keywords_path: path to YAML file defining keyword groups

    Note: wrap the result in TITLE-ABS-KEY(...) when calling the Scopus API.
    """
    groups = load_keywords(keywords_path)
    generic_query = build_query(logic_string, groups)
    return to_scopus(generic_query)

# ─────────────────────────────────────────────
# Semanctic Scholar QUERY BUILDING
# ─────────────────────────────────────────────

def to_semantic_scholar(query_string):
    """
    Convert a source-agnostic AND/OR query string to Semantic Scholar syntax.
    AND -> +
    OR  -> |

    Semantic Scholar format available here: https://api.semanticscholar.org/api-docs/#tag/Paper-Data/operation/get_graph_paper_bulk_search

    """
    result = re.sub(r'\bAND\b', '+', query_string)
    result = re.sub(r'\bOR\b', '|', result)
    return result

def build_semantic_scholar_query(logic_string, keywords_path):
    """
    Full pipeline: load keywords, build generic query, convert to SS syntax.
    Parameters:
    - logic_string: a string like "climate AND nlp AND (misinformation OR polarization)"
    - keywords_path: path to YAML file defining keyword groups
    """
    groups = load_keywords(keywords_path)
    generic_query = build_query(logic_string, groups)
    return to_semantic_scholar(generic_query)

# ─────────────────────────────────────────────
# Web of Science QUERY BUILDING
# ─────────────────────────────────────────────

def to_wos(query_string):
    """
    Convert a source-agnostic AND/OR query string to Web of Science (WoS) syntax.

    WoS uses AND/OR natively so no operator conversion is needed. The query is
    wrapped in TS=(...), which searches Title + Abstract + Author Keywords +
    Keywords Plus — the WoS equivalent of Scopus's TITLE-ABS-KEY.

    Three token-level transformations are applied:

    1. Quoted single-word wildcards are unquoted so WoS expands them.
       WoS does NOT expand wildcards inside quoted phrases — "LLM*" would
       be treated as a literal search for the string LLM* with an asterisk.
       Quotes are stripped: "LLM*" -> LLM*, "transformer*" -> transformer*.

    2. Bare tokens containing / are quoted to prevent misparse. 
        Quoting forces literal phrase matching: "Twitter/X".

    3. Bare tokens containing - are quoted.
       WoS treats hyphens as spaces, so bare alt-tech searches as "alt tech"
       (any paper containing both words anywhere). Quoting restricts to the phrase.

    Known limitations:
    - Case insensitivity: WoS is always case-insensitive. "Signal" (the platform)
      and "signal" (the general word) cannot be distinguished.
    - Wildcard + hyphen: hyphens are token separators in WoS, so LLM* matches LLMs
      but NOT LLM-based; transformer* does not match transformer-based. Add explicit
      hyphenated variants to keywords.yml if needed (e.g. "LLM-based").

    Note on the "Exact Search" UI toggle in WoS:
    Enabling it prevents stemming on bare single-word terms (e.g. without it,
    "climate" may also return "climatic"). Recommended for predictable results.
    """
    def transform(m):
        token = m.group(0)
        if token.startswith('"') and token.endswith('"'):
            inner = token[1:-1]
            # Single-word wildcard inside quotes: strip quotes so WoS expands the wildcard
            if ' ' not in inner and ('*' in inner or '?' in inner):
                return inner
            return token
        # Bare token with / or -: quote to prevent proximity misparse or hyphen-as-space expansion
        if '/' in token or '-' in token:
            return f'"{token}"'
        return token

    transformed = re.sub(r'"[^"]*"|[^\s()]+', transform, query_string)
    return f"TS=({transformed})"


def build_wos_query(logic_string, keywords_path):
    """
    Load keywords, build generic query, convert to WoS syntax.
    Parameters:
    - logic_string: a string like "climate AND nlp AND (misinformation OR polarization)"
    - keywords_path: path to YAML file defining keyword groups

    The result can be copied and pasted directly into the WoS Advanced Search
    interface. Recommend enabling "Exact Search" in the UI to prevent unintended
    stemming on bare single-word terms.
    """
    groups = load_keywords(keywords_path)
    generic_query = build_query(logic_string, groups)

    return to_wos(generic_query)

# ─────────────────────────────────────────────
# ACL ANTHOLOGY TEXT MATCHING
# ─────────────────────────────────────────────

def keyword_to_regex(keyword):
    """
    Convert a keyword (possibly with a trailing * wildcard) to a compiled regex.
    Anchored at word boundaries.

    Case sensitivity: if the keyword contains any uppercase letter, matching is
    case-sensitive (e.g. "Discord" matches "Discord" but not "discord"). If the
    keyword is all-lowercase, matching is case-insensitive.

    Examples:
        "transformer*"   -> case-insensitive prefix (matches transformer, Transformer, transformers...)
        "Discord"        -> case-sensitive (matches Discord, not discord)
        "climate change" -> case-insensitive phrase (matches climate change, Climate Change, Climate change...)
        "NLP"            -> case-sensitive (matches NLP, not nlp)
    """
    kw = keyword.strip('"')
    flags = 0 if any(c.isupper() for c in kw) else re.IGNORECASE
    if kw.endswith("*") and kw.count("*") == 1:
        stem = re.escape(kw[:-1])
        pattern = rf"\b{stem}\w*\b"
    else:
        pattern = rf"\b{re.escape(kw)}\b"
    return re.compile(pattern, flags)


def build_acl_text_matcher(logic_string, groups):
    """
    Build a text-matching function from a boolean keyword logic string.

    The returned function takes a plain text string and returns True if the text
    satisfies the boolean expression. Each group name in the logic string is
    treated as a condition that is True when any keyword in that group is found
    in the text.

    Keyword matching details:
    - Wildcards: a trailing * matches any suffix, e.g. "transformer*" matches
      "transformer", "transformers", "transformer-based", etc.
    - Case sensitivity: keywords with any uppercase letter are matched
      case-sensitively (e.g. "Discord" matches "Discord" but not "discord").
      All-lowercase keywords are matched case-insensitively.
    - Word boundaries: matches must start and end at a word boundary, so
      "climate" will not match "climated" (unless written as "climate*").

    Operator precedence: AND binds tighter than OR, matching standard boolean
    logic. So "A AND B OR C" is equivalent to "(A AND B) OR C". Use explicit
    parentheses to override, e.g. "A AND (B OR C)".

    Args:
        logic_string: Boolean expression over group names, e.g.
            "climate AND nlp"
            "climate AND (misinformation OR polarization)"
        groups: dict mapping each group name to a list of keyword strings,
            e.g. {"climate": ["climate change", "global warming"],
                  "nlp": ["NLP", "transformer*"]}

    Returns:
        A callable (text: str) -> bool

    Example:
        >>> groups = {"climate": ["climate change"], "nlp": ["NLP", "transformer*"]}
        >>> matcher = build_acl_text_matcher("climate AND nlp", groups)
        >>> matcher("A study on climate change using transformer models")
        True
        >>> matcher("Climate change policy without any NLP methods")
        False
    """
    
    tokens = tokenize_logic_string(logic_string)
    valid_ops = {"AND", "OR", "(", ")"}
    for tok in tokens:
        if tok not in valid_ops and tok not in groups:
            raise ValueError(
                f"Unrecognized token '{tok}'. "
                f"Expected AND, OR, parentheses, or one of: {list(groups.keys())}"
            )
    compiled = {
        name: [keyword_to_regex(kw) for kw in groups[name]]
        for name in tokens if name in groups
    }

    def make_matcher(toks):
        pos = 0

        def parse_binary(parse_child, op, combine_fn):
            nonlocal pos
            left = parse_child()
            while pos < len(toks) and toks[pos] == op:
                pos += 1
                r = parse_child()
                left = (lambda l, r: lambda t: combine_fn(l(t), r(t)))(left, r)
            return left

        def parse_expr():
            return parse_binary(parse_term, "OR", lambda a, b: a or b)

        def parse_term():
            return parse_binary(parse_factor, "AND", lambda a, b: a and b)

        def parse_factor():
            nonlocal pos
            if toks[pos] == "(":
                pos += 1
                inner = parse_expr()
                if pos >= len(toks) or toks[pos] != ")":
                    raise ValueError("Unmatched parenthesis in logic string")
                pos += 1
                return inner
            name = toks[pos]
            pos += 1
            pats = compiled[name]
            return lambda t, p=pats: any(pat.search(t) for pat in p)

        return parse_expr()

    return make_matcher(tokens)
