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

_THREAD_SEPARATOR_RE = re.compile(r"^\s*[_=*-]{10,}\s*$")
_THREAD_SEPARATOR_HEADER_RE = re.compile(
    r"^\s*(?:from|da|de|sent|inviato|date|data|to|a|cc|bcc|subject|oggetto)\s*:",
    re.IGNORECASE,
)


_AI_TAIL_BOILERPLATE_RE = re.compile(
    r"^\s*(?:"
    r"please consider the impact on the environment before printing|"
    r"this e-?mail may contain|"
    r"this message may contain|"
    r"this e-?mail was sent to you by|"
    r"legal disclosure\b|"
    r"privacy statement\b|"
    r"if you no longer wish to receive|"
    r"if you'd like me to stop sending you emails|"
    r"if you would like me to stop sending you emails|"
    r"unsubscribe\b|"
    r"informativa privacy\b|"
    r"riservatezza\b|"
    r"avvertenza di riservatezza\b|"
    r"nota di riservatezza\b"
    r")",
    re.IGNORECASE,
)

_AI_SIGNATURE_START_RE = re.compile(
    r"^\s*(?:"
    r"cordiali saluti|"
    r"distinti saluti|"
    r"un saluto|"
    r"saluti|"
    r"best regards|"
    r"kind regards|"
    r"regards|"
    r"thanks|"
    r"thank you"
    r"|good luck"
    r"|sincerely"
    r"|cheers"
    r")\s*,?\s*$",
    re.IGNORECASE,
)


def _meaningful_line_count(lines: list[str]) -> int:
    return sum(1 for line in lines if line.strip())


def _trim_ai_tail(lines: list[str]) -> tuple[list[str], int]:
    """Remove signatures, legal footers and unsubscribe blocks from the AI body."""
    end = len(lines)
    while end > 0 and not lines[end - 1].strip():
        end -= 1
    lines = lines[:end]

    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        before = lines[:index]
        if _AI_TAIL_BOILERPLATE_RE.match(stripped) and _meaningful_line_count(before) >= 1:
            return before, len(lines) - index
        if _AI_SIGNATURE_START_RE.match(stripped) and _meaningful_line_count(before) >= 2:
            return before, len(lines) - index
    return lines, 0


def _finalize_body_ai(lines: list[str], removed_quotes: int = 0, removed_headers: int = 0) -> tuple[str, int, int, int]:
    selected, removed_tail = _trim_ai_tail(lines)
    return _join_significant(selected), removed_quotes, removed_headers, removed_tail


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


def _looks_like_thread_separator(lines: list[str], index: int) -> bool:
    if not _THREAD_SEPARATOR_RE.match(lines[index]):
        return False
    nearby_headers = sum(
        1
        for line in lines[index + 1:index + 9]
        if _THREAD_SEPARATOR_HEADER_RE.match(line.strip())
    )
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
            "body_ai_removed_tail_lines": 0,
        }

    for index, line in enumerate(lines):
        if _FORWARD_MARKER_RE.match(line):
            selected, removed_headers = _strip_forwarded_headers(lines[index + 1:])
            selected, removed_quotes = _remove_quoted_lines(selected)
            body_ai, removed_quotes, removed_headers, removed_tail = _finalize_body_ai(
                selected, removed_quotes, removed_headers
            )
            return {
                "body_ai": body_ai,
                "body_context": "forwarded",
                "body_ai_removed_quoted_lines": removed_quotes,
                "body_ai_removed_header_lines": removed_headers,
                "body_ai_removed_tail_lines": removed_tail,
            }

    for index, line in enumerate(lines):
        if _REPLY_MARKER_RE.match(line):
            selected, removed_quotes = _remove_quoted_lines(lines[:index])
            body_ai, removed_quotes, removed_headers, removed_tail = _finalize_body_ai(
                selected, removed_quotes, 0
            )
            if body_ai:
                return {
                    "body_ai": body_ai,
                    "body_context": "reply",
                    "body_ai_removed_quoted_lines": removed_quotes,
                    "body_ai_removed_header_lines": removed_headers,
                    "body_ai_removed_tail_lines": removed_tail,
                }
            break

    for index, line in enumerate(lines):
        if _looks_like_outlook_reply_header(lines, index):
            selected, removed_quotes = _remove_quoted_lines(lines[:index])
            body_ai, removed_quotes, removed_headers, removed_tail = _finalize_body_ai(
                selected, removed_quotes, len(lines[index:])
            )
            if body_ai:
                return {
                    "body_ai": body_ai,
                    "body_context": "reply",
                    "body_ai_removed_quoted_lines": removed_quotes,
                    "body_ai_removed_header_lines": removed_headers,
                    "body_ai_removed_tail_lines": removed_tail,
                }
            break

    for index, line in enumerate(lines):
        if _looks_like_thread_separator(lines, index):
            selected, removed_quotes = _remove_quoted_lines(lines[:index])
            body_ai, removed_quotes, removed_headers, removed_tail = _finalize_body_ai(
                selected, removed_quotes, len(lines[index:])
            )
            if body_ai:
                return {
                    "body_ai": body_ai,
                    "body_context": "reply",
                    "body_ai_removed_quoted_lines": removed_quotes,
                    "body_ai_removed_header_lines": removed_headers,
                    "body_ai_removed_tail_lines": removed_tail,
                }
            break

    selected, removed_quotes = _remove_quoted_lines(lines)
    body_ai, removed_quotes, removed_headers, removed_tail = _finalize_body_ai(
        selected, removed_quotes, 0
    )
    return {
        "body_ai": body_ai,
        "body_context": "normal",
        "body_ai_removed_quoted_lines": removed_quotes,
        "body_ai_removed_header_lines": removed_headers,
        "body_ai_removed_tail_lines": removed_tail,
    }
