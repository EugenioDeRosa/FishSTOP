"""
Utilities for selecting the email text that should be sent to BERT.

Replies and forwards often contain more than one conversation layer. The full
body remains useful for link extraction and manual inspection, but the model
should see the most relevant layer only.
"""

import re


_FORWARD_MARKER_RE = re.compile(
    r"^\s*(?:"
    r"-{2,}\s*forwarded message\s*-{2,}|"
    r"begin forwarded message:|"
    r"forwarded message|"
    r"messaggio inoltrato|"
    r"inizio messaggio inoltrato"
    r")\s*$",
    re.IGNORECASE,
)

_REPLY_MARKER_RE = re.compile(
    r"^\s*(?:"
    r"on .+ wrote:|"
    r"il giorno .+ ha scritto:|"
    r"le .+ a ecrit\s*:|"
    r"am .+ schrieb .+:|"
    r"-{2,}\s*original message\s*-{2,}|"
    r"-{2,}\s*messaggio originale\s*-{2,}"
    r")\s*$",
    re.IGNORECASE,
)

_FORWARDED_HEADER_RE = re.compile(
    r"^\s*(?:"
    r"from|da|de|sent|inviato|date|data|to|a|cc|bcc|subject|oggetto"
    r")\s*:",
    re.IGNORECASE,
)

_OUTLOOK_REPLY_FROM_RE = re.compile(r"^\s*(?:from|da|de)\s*:", re.IGNORECASE)
_OUTLOOK_REPLY_HEADER_RE = re.compile(
    r"^\s*(?:sent|inviato|date|data|to|a|cc|bcc|subject|oggetto)\s*:",
    re.IGNORECASE,
)


def _normalize_lines(text: str) -> list[str]:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    return [line.rstrip() for line in text.split("\n")]


def _join_significant(lines: list[str]) -> str:
    text = "\n".join(lines).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _remove_quoted_lines(lines: list[str]) -> tuple[list[str], int]:
    kept: list[str] = []
    removed = 0
    for line in lines:
        if line.lstrip().startswith(">"):
            removed += 1
            continue
        kept.append(line)
    return kept, removed


def _strip_forwarded_headers(lines: list[str]) -> tuple[list[str], int]:
    stripped = 0
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            stripped += 1
            index += 1
            continue
        if _FORWARDED_HEADER_RE.match(line):
            stripped += 1
            index += 1
            continue
        break
    return lines[index:], stripped


def _looks_like_outlook_reply_header(lines: list[str], index: int) -> bool:
    if not _OUTLOOK_REPLY_FROM_RE.match(lines[index]):
        return False

    nearby_headers = 0
    for line in lines[index + 1:index + 7]:
        stripped = line.strip()
        if not stripped:
            continue
        if _OUTLOOK_REPLY_HEADER_RE.match(stripped):
            nearby_headers += 1
            continue
        if nearby_headers:
            break

    return nearby_headers >= 2


def select_body_for_ai(body_clean: str) -> dict:
    """
    Returns a BERT-focused body while preserving metadata about the selection.

    - Forwarded emails: use the forwarded payload after the separator.
    - Replies: use the new message before the quoted conversation.
    - Normal emails: use the cleaned body, without quoted ``>`` lines.
    """
    lines = _normalize_lines(body_clean)
    if not any(line.strip() for line in lines):
        return {
            "body_ai": "",
            "body_context": "empty",
            "body_ai_removed_quoted_lines": 0,
            "body_ai_removed_header_lines": 0,
        }

    for index, line in enumerate(lines):
        if _FORWARD_MARKER_RE.match(line):
            selected, removed_headers = _strip_forwarded_headers(lines[index + 1:])
            selected, removed_quotes = _remove_quoted_lines(selected)
            body_ai = _join_significant(selected)
            return {
                "body_ai": body_ai,
                "body_context": "forwarded",
                "body_ai_removed_quoted_lines": removed_quotes,
                "body_ai_removed_header_lines": removed_headers,
            }

    for index, line in enumerate(lines):
        if _REPLY_MARKER_RE.match(line):
            selected, removed_quotes = _remove_quoted_lines(lines[:index])
            body_ai = _join_significant(selected)
            if body_ai:
                return {
                    "body_ai": body_ai,
                    "body_context": "reply",
                    "body_ai_removed_quoted_lines": removed_quotes,
                    "body_ai_removed_header_lines": 0,
                }
            break

    for index, line in enumerate(lines):
        if _looks_like_outlook_reply_header(lines, index):
            selected, removed_quotes = _remove_quoted_lines(lines[:index])
            body_ai = _join_significant(selected)
            if body_ai:
                return {
                    "body_ai": body_ai,
                    "body_context": "reply",
                    "body_ai_removed_quoted_lines": removed_quotes,
                    "body_ai_removed_header_lines": len(lines[index:]),
                }
            break

    selected, removed_quotes = _remove_quoted_lines(lines)
    return {
        "body_ai": _join_significant(selected),
        "body_context": "normal",
        "body_ai_removed_quoted_lines": removed_quotes,
        "body_ai_removed_header_lines": 0,
    }
