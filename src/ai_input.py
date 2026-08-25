"""Compact, privacy-preserving email body input shared by the AI models."""

import re
import unicodedata


_HTML_LINK_RE = re.compile(
    r"(<a\b[^>]*\bhref\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s>]+)[^>]*>)(.*?)(</a\s*>)",
    re.IGNORECASE | re.DOTALL,
)
_HTML_TAG_RE = re.compile(
    r"</?[A-Za-z][A-Za-z0-9:-]*(?:\s[^>]*)?/?>",
    re.DOTALL,
)

_PLACEHOLDER_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    (
        re.compile(
            r"(?i)(?<!\w)(?:https?://|ftp://|www\.)[^\s<>\"']+|"
            r"(?<!\w)mailto:[^\s<>\"']+"
        ),
        "[URL LINK]",
    ),
    (
        re.compile(r"\b(?:[A-Za-z0-9._%+-]+)@(?:[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b"),
        "[EMAIL ADDRESS]",
    ),
    (re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b", re.IGNORECASE), "[IBAN]"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[IP ADDRESS]"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[CARD OR ACCOUNT NUMBER]"),
    (re.compile(r"(?<!\w)\+?\d[\d .()/-]{7,}\d\b"), "[PHONE NUMBER]"),
    (
        re.compile(
            r"(?<![A-Za-z0-9@_+=./-])"
            r"[A-Za-z0-9][A-Za-z0-9@_+=./-]{159,}"
            r"(?![A-Za-z0-9@_+=./-])"
        ),
        "[OBFUSCATED DATA]",
    ),
)


def _mark_html_links(value: str) -> str:
    """Keep anchor labels while making their hidden destinations visible as a token."""

    return _HTML_LINK_RE.sub(r"\1\2 [URL LINK]\3", value)


def compact_ai_body(body: str, *, has_extracted_links: bool = False) -> str:
    """Return only useful visible body text with noisy values replaced by labels."""

    if not body:
        return ""

    text = unicodedata.normalize("NFKC", str(body))
    if _HTML_TAG_RE.search(text):
        text = _mark_html_links(text)
        try:
            from src.analyzer.html_utils import strip_html
        except Exception:
            text = re.sub(r"<[^>]+>", " ", text)
        else:
            text = strip_html(text)

    text = "".join(
        char if char in "\n\t" or unicodedata.category(char) != "Cc" else " "
        for char in text
    )
    for pattern, replacement in _PLACEHOLDER_PATTERNS:
        text = pattern.sub(replacement, text)

    # An HTML-only destination may have been extracted after visible-text
    # parsing. Preserve that security-relevant fact without sending the URL.
    if has_extracted_links and "[URL LINK]" not in text:
        text = f"{text.rstrip()}\n[URL LINK]"

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
