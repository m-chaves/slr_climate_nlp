"""
Text normalization utilities for the data harmonization pipeline.

normalize_title_for_matching() is fully implemented — it is called at merge
time to enable fuzzy title comparison.

normalize_venue() and normalize_paper_type() are stubs that return their
input unchanged.  They define the interface and document each source's
vocabulary so that implementations can be added without changing callers.
"""

import re
import unicodedata


# ──────────────────────────────────────────────────────────────────────────────
# Title normalization  (fully implemented)
# ──────────────────────────────────────────────────────────────────────────────

def normalize_title_for_matching(title: str | None) -> str | None:
    """
    Return a canonical lowercase title string for fuzzy comparison.

    Applied at merge time only — the normalized form is never stored.
    The pipeline handles real encoding variants found in the data:
      - U+00A0 non-breaking spaces (Scopus titles)
      - Smart quotes / curly apostrophes (U+2018, U+2019, U+201C, U+201D)
      - Em/en dashes vs hyphens
      - Ligatures (ﬁ → fi, etc.)

    Steps applied in order:
      1. NFKD decomposition  — decomposes ligatures and smart punctuation
      2. ASCII encode+decode — strips all remaining non-ASCII bytes
      3. Lowercase
      4. Strip punctuation   — removes everything that is not word char or space
      5. Collapse whitespace
    """
    if not title:
        return None
    s = unicodedata.normalize("NFKD", title)
    s = s.encode("ascii", "ignore").decode()
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


# ──────────────────────────────────────────────────────────────────────────────
# Venue normalization  (stub)
# ──────────────────────────────────────────────────────────────────────────────

def normalize_venue(venue: str | None, source: str) -> str | None:
    """
    Return a canonical venue name.  Currently a stub — returns input unchanged.

    TODO: Implement entity resolution to map variant strings to a canonical
    venue name.  Known variants for the same venue observed across sources:

      Scopus publicationName:
        "Proceedings of the 2024 Conference on Empirical Methods in NLP"
        "Findings of the Association for Computational Linguistics: EMNLP 2024"
      WoS SO:
        "PROCEEDINGS OF THE 2024 CONFERENCE ON EMPIRICAL METHODS IN NLP"
        (WoS SO is always uppercase)
      Semantic Scholar publicationVenue.name:
        "Conference on Empirical Methods in Natural Language Processing"
        or publicationVenue.alternate_names: ["EMNLP"]
      ACL venue_acronym: "EMNLP"
      ACL volume_title: "Proceedings of the 2024 Conference on EMNLP"

    Recommended implementation strategy:
      1. First pass: lookup in data/reference/venue_aliases.json
         (a curated dict mapping alias strings → canonical name, e.g.
          {"EMNLP": "EMNLP", "conf empirical methods nlp": "EMNLP", ...})
      2. Second pass: rapidfuzz fuzzy match against known canonical names
         for strings not found in the lookup table
      3. WoS SO requires .lower() before lookup (field is always uppercase)

    Args:
        venue: raw venue string from the source record
        source: one of "scopus", "acl", "semantic_scholar", "wos", "acm"
    """
    return venue


# ──────────────────────────────────────────────────────────────────────────────
# Paper type normalization  (stub)
# ──────────────────────────────────────────────────────────────────────────────

def normalize_paper_type(paper_type: str | None, source: str) -> str | None:
    """
    Return a canonical paper type.  Currently a stub — returns input unchanged.

    TODO: Map each source's vocabulary to a shared controlled vocabulary:
      Target canonical values (proposed):
        "journal_article", "conference_paper", "workshop_paper",
        "review", "book_chapter", "editorial", "preprint", "other"

    Source vocabularies:
      Scopus subtypeDescription:
        "Article"          → "journal_article"
        "Conference Paper" → "conference_paper"
        "Review"           → "review"
        "Book Chapter"     → "book_chapter"
        "Editorial"        → "editorial"
        "Conference Review"→ "review"  (Scopus-specific, peer-reviewed conf review)

      WoS DT (Document Type):
        "Article"                → "journal_article"
        "Review"                 → "review"
        "Article; Early Access"  → "journal_article"
        "Proceedings Paper"      → "conference_paper"

      Semantic Scholar publicationTypes (list):
        ["JournalArticle"]  → "journal_article"
        ["Conference"]      → "conference_paper"
        ["Review"]          → "review"
        ["Preprint"]        → "preprint"

      ACL paper_type:
        "paper"             → "conference_paper" or "workshop_paper"
                              (distinguish via volume_type: "proceedings")
        "frontmatter"       → "other"
        "backmatter"        → "other"

    Args:
        paper_type: raw paper type string from the source record
        source: one of "scopus", "acl", "semantic_scholar", "wos", "acm"
    """
    return paper_type
