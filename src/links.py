"""Compatibility exports for link analysis helpers."""

from src.analyzer import check_lookalike_domains, extract_links, is_ip_url
from src.analyzer.lookalike import levenshtein as _levenshtein
from src.analyzer.lookalike import normalize_homoglyphs as _normalize_homoglyphs

_is_ip_url = is_ip_url

__all__ = [
    "extract_links",
    "check_lookalike_domains",
    "is_ip_url",
    "_is_ip_url",
    "_levenshtein",
    "_normalize_homoglyphs",
]
