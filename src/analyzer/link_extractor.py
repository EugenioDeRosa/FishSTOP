"""
analyzer/link_extractor.py - URL extraction from email bodies.

Espone:
  - extract_links(body_plain, body_html) : lista di link strutturati

Handles plain text and HTML while preserving visible link text.
"""

import re
from urllib.parse import urlparse

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - fallback per ambienti minimali
    BeautifulSoup = None

from .html_utils import strip_html
from .lookalike import is_ip_url


_URL_RE = re.compile(
    r"""(?i)\b(?:https?://|ftp://|www\.)"""
    r"""(?:[^\W_][\w\-]*\.)+[^\W_]{2,}"""
    r"""(?::\d{1,5})?"""
    r"""(?:/[^\s"'<>\]\)]*)?""",
    re.VERBOSE,
)
_BARE_DOMAIN_RE = re.compile(
    r"""(?i)(?<![@\w.-])(?:[^\W_][\w\-]*\.)+[^\W_]{2,}(?![\w.-])""",
    re.VERBOSE,
)
_HREF_RE = re.compile(r"""href\s*=\s*["']?(https?://[^\s"'<>]+)""", re.IGNORECASE)
_ANCHOR_RE = re.compile(
    r"""<a\b[^>]*href\s*=\s*["']?(?P<href>https?://[^\s"'<>]+)["']?[^>]*>(?P<text>.*?)</a>""",
    re.IGNORECASE | re.DOTALL,
)
_OPAQUE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{4,32}$")


def _contains_non_ascii(value: str) -> bool:
    return any(ord(ch) > 127 for ch in value or "")


def _with_scheme(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    if value.lower().startswith("www.") or "://" not in value:
        return "http://" + value
    return value


def _registered_domain(host: str) -> str:
    parts = (host or "").lower().rstrip(".").split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host or ""


def _extract_display_destination(display: str) -> tuple[str, str]:
    text = display or ""
    match = _URL_RE.search(text) or _BARE_DOMAIN_RE.search(text)
    if not match:
        return "", ""
    candidate = _with_scheme(match.group(0).strip().rstrip(".,;)"))
    try:
        parsed = urlparse(candidate)
    except Exception:
        return "", ""
    return candidate, (parsed.hostname or "").lower()


def _looks_opaque_token(segment: str) -> bool:
    if not _OPAQUE_TOKEN_RE.match(segment or ""):
        return False
    has_alpha = any(ch.isalpha() for ch in segment)
    has_digit = any(ch.isdigit() for ch in segment)
    mixed_case = any(ch.islower() for ch in segment) and any(ch.isupper() for ch in segment)
    return len(segment) >= 5 and has_alpha and (has_digit or mixed_case or "_" in segment or "-" in segment)


def _possible_shortener(url: str, host: str) -> tuple[bool, str]:
    try:
        parsed = urlparse(url)
    except Exception:
        return False, ""
    segments = [seg for seg in parsed.path.split("/") if seg]
    if not host or not segments:
        return False, ""

    first = segments[0]
    sld = _registered_domain(host).split(".")[0]
    if len(sld) <= 5 and _OPAQUE_TOKEN_RE.match(first):
        return True, "short domain with opaque token in path"
    if len(segments) <= 2 and not parsed.query and _looks_opaque_token(first) and len(first) <= 16:
        return True, "percorso breve e opaco tipico di redirector/short link"
    return False, ""


def _same_registered_domain(left: str, right: str) -> bool:
    return bool(left and right and _registered_domain(left) == _registered_domain(right))


def extract_links(body_plain: str, body_html: str) -> list[dict]:
    """
    Extracts all links from the email body.

    Besides the real destination, HTML links keep the visible text
    and are flagged when the text shows a different domain than the href destination.
    """
    seen: set[str] = set()
    links: list[dict] = []

    def _add(url: str, display: str, source: str) -> None:
        url = _with_scheme((url or "").strip().rstrip(".,;)"))
        if not url or url in seen:
            return
        seen.add(url)
        try:
            parsed = urlparse(url)
            host = (parsed.hostname or "").lower()
            scheme = parsed.scheme.lower()
        except Exception:
            return

        display_text = (display or "").strip()
        display_url, display_host = _extract_display_destination(display_text)
        is_shortener, shortener_reason = _possible_shortener(url, host)

        links.append({
            "url": url,
            "display_text": display_text[:120],
            "display_url": display_url,
            "display_host": display_host,
            "display_mismatch": bool(display_host and host and not _same_registered_domain(display_host, host)),
            "host": host,
            "scheme": scheme,
            "source": source,
            "is_ip": is_ip_url(host),
            "is_possible_shortener": is_shortener,
            "shortener_reason": shortener_reason,
        })

    def _add_unicode_bare_domains(text: str, source: str) -> None:
        text = text or ""
        for m in _BARE_DOMAIN_RE.finditer(text):
            prefix = text[max(0, m.start() - 8):m.start()].lower()
            if prefix.endswith(("http://", "https://", "ftp://")):
                continue
            domain = m.group(0)
            if _contains_non_ascii(domain):
                _add(domain, "", source)

    if body_html:
        if BeautifulSoup is not None:
            soup = BeautifulSoup(body_html, "html.parser")
            for anchor in soup.find_all("a"):
                href = anchor.get("href")
                if href:
                    _add(href, anchor.get_text(" ", strip=True), "html_href")
        else:
            matched_spans = []
            for m in _ANCHOR_RE.finditer(body_html):
                matched_spans.append(m.span())
                _add(m.group("href"), strip_html(m.group("text")), "html_href")
            for m in _HREF_RE.finditer(body_html):
                if any(start <= m.start() < end for start, end in matched_spans):
                    continue
                _add(m.group(1), "", "html_href")

        html_stripped = strip_html(body_html)
        for m in _URL_RE.finditer(html_stripped):
            _add(m.group(0), "", "html_text")
        _add_unicode_bare_domains(html_stripped, "html_domain")

    if body_plain:
        for m in _URL_RE.finditer(body_plain):
            _add(m.group(0), "", "plain_text")
        _add_unicode_bare_domains(body_plain, "plain_domain")

    return links
