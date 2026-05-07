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


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

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