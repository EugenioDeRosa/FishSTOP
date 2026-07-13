"""Attachment analysis helpers."""

import hashlib
import io
import re
from collections import Counter
from typing import Optional

from .constants import CONTENT_TYPE_TO_EXT, MAGIC_BYTES

ZIP_CONTAINER_EXTS = {"docx", "xlsx", "pptx", "zip"}

PDF_ANALYSIS_MAX_BYTES = 25 * 1024 * 1024
PDF_OBJECT_WALK_LIMIT = 5000

PDF_RISK_DEFINITIONS: dict[str, dict] = {
    "javascript": {
        "names": {"/JavaScript", "/JS"},
        "label": "embedded JavaScript",
        "severity": "high",
        "weight": 45,
    },
    "open_action": {
        "names": {"/OpenAction"},
        "label": "automatic action on document open",
        "severity": "high",
        "weight": 35,
    },
    "additional_action": {
        "names": {"/AA"},
        "label": "additional automatic action",
        "severity": "high",
        "weight": 30,
    },
    "launch_action": {
        "names": {"/Launch"},
        "label": "launch action",
        "severity": "critical",
        "weight": 60,
    },
    "embedded_file": {
        "names": {"/EmbeddedFile", "/Filespec", "/EmbeddedFiles"},
        "label": "embedded file or attachment reference",
        "severity": "high",
        "weight": 35,
    },
    "acroform": {
        "names": {"/AcroForm"},
        "label": "interactive form",
        "severity": "low",
        "weight": 8,
    },
    "xfa": {
        "names": {"/XFA"},
        "label": "XFA form content",
        "severity": "high",
        "weight": 35,
    },
    "uri": {
        "names": {"/URI"},
        "label": "external URI action",
        "severity": "medium",
        "weight": 15,
    },
    "submit_form": {
        "names": {"/SubmitForm"},
        "label": "form submission action",
        "severity": "high",
        "weight": 35,
    },
    "rich_media": {
        "names": {"/RichMedia", "/Movie", "/Sound", "/3D"},
        "label": "active media content",
        "severity": "high",
        "weight": 30,
    },
    "remote_goto": {
        "names": {"/GoToR", "/GoToE"},
        "label": "remote or embedded go-to action",
        "severity": "medium",
        "weight": 20,
    },
    "import_data": {
        "names": {"/ImportData"},
        "label": "external data import action",
        "severity": "high",
        "weight": 35,
    },
    "jbig2": {
        "names": {"/JBIG2Decode"},
        "label": "JBIG2 compressed stream",
        "severity": "medium",
        "weight": 12,
    },
    "object_stream": {
        "names": {"/ObjStm", "/XRefStm"},
        "label": "compressed object/xref stream",
        "severity": "low",
        "weight": 6,
    },
}

PDF_NAME_TO_KEY = {
    name: key
    for key, definition in PDF_RISK_DEFINITIONS.items()
    for name in definition["names"]
}

PDF_ACTIVE_CONTENT_KEYS = {
    "javascript",
    "open_action",
    "additional_action",
    "launch_action",
    "embedded_file",
    "xfa",
    "submit_form",
    "rich_media",
    "remote_goto",
    "import_data",
}

PDF_NAME_RE = re.compile(r"/[A-Za-z0-9_.:+#-]+")
PDF_HEX_ESCAPE_RE = re.compile(r"#([0-9A-Fa-f]{2})")
URL_RE = re.compile(rb"https?://|mailto:", re.IGNORECASE)


def identify_magic_bytes(raw: bytes) -> Optional[str]:
    """Return the format identified by magic bytes, if known."""
    for fmt, signatures in MAGIC_BYTES.items():
        if any(raw.startswith(signature) for signature in signatures):
            return fmt
    return None


def ext_from_filename(filename: str) -> Optional[str]:
    """Extract a lower-case extension from a filename."""
    if "." not in filename:
        return None
    return filename.rsplit(".", 1)[-1].lower()


def _payload_to_bytes(raw_payload) -> tuple[bytes | None, str | None]:
    if raw_payload is None:
        return None, "Attachment payload empty or not decodable"
    if isinstance(raw_payload, bytes):
        return raw_payload, None
    if isinstance(raw_payload, bytearray):
        return bytes(raw_payload), None
    if isinstance(raw_payload, str):
        return raw_payload.encode("utf-8", errors="ignore"), (
            "Attachment payload was not decoded by email parser"
        )
    return None, f"Unsupported attachment payload type: {type(raw_payload).__name__}"


def _decode_pdf_name_escapes(value: str) -> str:
    def repl(match: re.Match) -> str:
        return chr(int(match.group(1), 16))

    return PDF_HEX_ESCAPE_RE.sub(repl, value)


def _pdf_text_for_static_scan(raw: bytes) -> str:
    return _decode_pdf_name_escapes(raw.decode("latin-1", errors="ignore"))


def _add_indicator(counter: Counter, key: str, count: int = 1) -> None:
    if key and count > 0:
        counter[key] += count


def _indicator_list(counter: Counter) -> list[dict]:
    indicators = []
    for key, count in counter.items():
        definition = PDF_RISK_DEFINITIONS.get(key, {})
        indicators.append({
            "key": key,
            "label": definition.get("label", key),
            "severity": definition.get("severity", "info"),
            "count": count,
            "weight": definition.get("weight", 0),
        })
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    return sorted(
        indicators,
        key=lambda item: (severity_order.get(item["severity"], 9), -item["count"], item["label"]),
    )


def _static_pdf_indicators(raw: bytes) -> tuple[Counter, dict]:
    counter: Counter = Counter()
    raw_text = raw.decode("latin-1", errors="ignore")
    suspicious_name_escapes = len(PDF_HEX_ESCAPE_RE.findall(raw_text))
    text = _decode_pdf_name_escapes(raw_text)
    names = PDF_NAME_RE.findall(text)
    name_counts = Counter(names)

    for name, count in name_counts.items():
        key = PDF_NAME_TO_KEY.get(name)
        if key:
            _add_indicator(counter, key, count)

    uri_count = len(URL_RE.findall(raw))
    object_count = len(re.findall(rb"\b\d+\s+\d+\s+obj\b", raw))
    stream_count = len(re.findall(rb"\bstream\b", raw))
    encrypted = b"/Encrypt" in raw or "/Encrypt" in text
    eof_count = raw.count(b"%%EOF")

    return counter, {
        "uri_count": uri_count,
        "object_count": object_count,
        "stream_count": stream_count,
        "encrypted": encrypted,
        "suspicious_name_escapes": suspicious_name_escapes,
        "eof_count": eof_count,
    }


def _safe_pdf_str(value) -> str:
    try:
        return str(value)
    except Exception:
        return ""


def _scan_pypdf_object(obj, counter: Counter, stats: dict, seen: set[int], depth: int = 0) -> None:
    if obj is None or depth > 35 or stats["walked_nodes"] >= PDF_OBJECT_WALK_LIMIT:
        return

    obj_id = id(obj)
    if obj_id in seen:
        return
    seen.add(obj_id)
    stats["walked_nodes"] += 1

    try:
        if hasattr(obj, "get_object") and obj.__class__.__name__ == "IndirectObject":
            _scan_pypdf_object(obj.get_object(), counter, stats, seen, depth + 1)
            return
    except Exception as exc:
        stats["parser_warnings"].append(f"Indirect object read failed: {exc}")
        return

    if isinstance(obj, dict):
        for raw_key, raw_value in obj.items():
            key = _decode_pdf_name_escapes(_safe_pdf_str(raw_key))
            indicator_key = PDF_NAME_TO_KEY.get(key)
            if indicator_key:
                _add_indicator(counter, indicator_key)
            value_name = _decode_pdf_name_escapes(_safe_pdf_str(raw_value))
            value_indicator_key = PDF_NAME_TO_KEY.get(value_name)
            if value_indicator_key:
                _add_indicator(counter, value_indicator_key)
            _scan_pypdf_object(raw_value, counter, stats, seen, depth + 1)
        return

    if isinstance(obj, (list, tuple)):
        for item in obj:
            _scan_pypdf_object(item, counter, stats, seen, depth + 1)
        return

    obj_text = _decode_pdf_name_escapes(_safe_pdf_str(obj))
    indicator_key = PDF_NAME_TO_KEY.get(obj_text)
    if indicator_key:
        _add_indicator(counter, indicator_key)


def _pypdf_structural_scan(raw: bytes) -> tuple[Counter, dict]:
    counter: Counter = Counter()
    stats = {
        "parser": "pypdf-unavailable",
        "parser_available": False,
        "parser_error": "pypdf is not installed",
        "parser_warnings": [],
        "page_count": None,
        "field_count": 0,
        "embedded_attachment_count": 0,
        "is_encrypted": False,
        "is_decrypted_with_empty_password": False,
        "has_xfa": False,
        "has_open_destination": False,
        "page_mode": None,
        "walked_nodes": 0,
    }

    try:
        from pypdf import PdfReader
    except Exception:
        return counter, stats

    stats.update({"parser": "pypdf", "parser_available": True, "parser_error": None})
    try:
        reader = PdfReader(io.BytesIO(raw), strict=False, root_object_recovery_limit=20000)
        stats["is_encrypted"] = bool(reader.is_encrypted)
        if reader.is_encrypted:
            try:
                stats["is_decrypted_with_empty_password"] = bool(reader.decrypt(""))
            except Exception as exc:
                stats["parser_warnings"].append(f"Encrypted PDF could not be decrypted with empty password: {exc}")

        try:
            stats["page_count"] = len(reader.pages)
        except Exception as exc:
            stats["parser_warnings"].append(f"Page count unavailable: {exc}")

        try:
            fields = reader.get_fields() or {}
            stats["field_count"] = len(fields)
            if fields:
                _add_indicator(counter, "acroform")
        except Exception as exc:
            stats["parser_warnings"].append(f"Form fields unavailable: {exc}")

        try:
            xfa = getattr(reader, "xfa", None)
            stats["has_xfa"] = bool(xfa)
            if xfa:
                _add_indicator(counter, "xfa")
        except Exception as exc:
            stats["parser_warnings"].append(f"XFA unavailable: {exc}")

        try:
            embedded = getattr(reader, "attachments", {}) or {}
            stats["embedded_attachment_count"] = sum(len(value) for value in embedded.values())
            if stats["embedded_attachment_count"]:
                _add_indicator(counter, "embedded_file", stats["embedded_attachment_count"])
        except Exception as exc:
            stats["parser_warnings"].append(f"Embedded attachments unavailable: {exc}")

        try:
            open_destination = getattr(reader, "open_destination", None)
            stats["has_open_destination"] = bool(open_destination)
            if open_destination:
                _add_indicator(counter, "open_action")
        except Exception as exc:
            stats["parser_warnings"].append(f"Open destination unavailable: {exc}")

        try:
            stats["page_mode"] = _safe_pdf_str(getattr(reader, "page_mode", None) or "") or None
            if stats["page_mode"] == "/UseAttachments":
                _add_indicator(counter, "embedded_file")
        except Exception as exc:
            stats["parser_warnings"].append(f"Page mode unavailable: {exc}")

        try:
            _scan_pypdf_object(reader.root_object, counter, stats, set())
        except Exception as exc:
            stats["parser_warnings"].append(f"Catalog walk failed: {exc}")
    except Exception as exc:
        stats["parser_error"] = str(exc)

    return counter, stats


def _risk_level(score: int, indicators: list[dict], encrypted: bool, parser_error: str | None) -> str:
    if any(item["key"] == "launch_action" for item in indicators):
        return "critical"
    if score >= 60 or any(item["severity"] == "critical" for item in indicators):
        return "critical"
    if score >= 35 or any(item["severity"] == "high" for item in indicators):
        return "high"
    if score >= 15 or encrypted or parser_error:
        return "medium"
    if score > 0:
        return "low"
    return "clean"


def analyze_pdf_security(raw: bytes) -> dict:
    """Run static PDF risk checks without executing or rendering the document."""
    if not raw.startswith(b"%PDF"):
        return {
            "is_pdf": False,
            "risk_level": "not_pdf",
            "suspicious": False,
            "score": 0,
            "indicators": [],
            "summary": "Not a PDF document",
            "uri_count": 0,
            "object_count": 0,
            "stream_count": 0,
            "encrypted": False,
            "parser": "not_pdf",
            "parser_available": False,
            "parser_error": None,
            "parser_warnings": [],
        }

    if len(raw) > PDF_ANALYSIS_MAX_BYTES:
        return {
            "is_pdf": True,
            "risk_level": "medium",
            "suspicious": True,
            "score": 20,
            "indicators": [],
            "summary": f"PDF too large for deep static scan ({len(raw)} bytes)",
            "uri_count": 0,
            "object_count": 0,
            "stream_count": 0,
            "encrypted": False,
            "parser": "skipped-size-limit",
            "parser_available": False,
            "parser_error": "PDF exceeds static analysis size limit",
            "parser_warnings": [],
        }

    static_counter, static_stats = _static_pdf_indicators(raw)
    structural_counter, structural_stats = _pypdf_structural_scan(raw)
    total_counter = static_counter + structural_counter
    indicators = _indicator_list(total_counter)

    encrypted = bool(static_stats["encrypted"] or structural_stats.get("is_encrypted"))
    parser_error = structural_stats.get("parser_error")
    parser_error_for_risk = parser_error if structural_stats.get("parser_available") else None
    score = sum(item["weight"] * min(item["count"], 3) for item in indicators)
    if encrypted:
        score += 15
    if static_stats["suspicious_name_escapes"]:
        score += min(static_stats["suspicious_name_escapes"], 5) * 3
    if static_stats["eof_count"] > 1:
        score += 8
    if parser_error_for_risk:
        score += 15

    risk_level = _risk_level(score, indicators, encrypted, parser_error_for_risk)
    suspicious = risk_level in {"critical", "high"} or any(
        item["key"] in PDF_ACTIVE_CONTENT_KEYS for item in indicators
    )

    summary_parts = [f"{item['label']} x{item['count']}" for item in indicators[:8]]
    if encrypted:
        summary_parts.append("encrypted PDF")
    if static_stats["suspicious_name_escapes"]:
        summary_parts.append(f"obfuscated PDF names x{static_stats['suspicious_name_escapes']}")
    if static_stats["eof_count"] > 1:
        summary_parts.append(f"multiple EOF markers x{static_stats['eof_count']}")
    if static_stats["uri_count"] and not any(item["key"] == "uri" for item in indicators):
        summary_parts.append(f"URL-like strings x{static_stats['uri_count']}")
    if parser_error_for_risk:
        summary_parts.append(f"structured parser error: {parser_error_for_risk}")

    return {
        "is_pdf": True,
        "risk_level": risk_level,
        "suspicious": suspicious,
        "score": score,
        "indicators": indicators,
        "summary": "; ".join(summary_parts) if summary_parts else "No active PDF features detected",
        "uri_count": static_stats["uri_count"],
        "object_count": static_stats["object_count"],
        "stream_count": static_stats["stream_count"],
        "encrypted": encrypted,
        "suspicious_name_escapes": static_stats["suspicious_name_escapes"],
        "eof_count": static_stats["eof_count"],
        "parser": structural_stats.get("parser"),
        "parser_available": structural_stats.get("parser_available"),
        "parser_error": parser_error,
        "parser_warnings": structural_stats.get("parser_warnings", [])[:5],
        "page_count": structural_stats.get("page_count"),
        "field_count": structural_stats.get("field_count"),
        "embedded_attachment_count": structural_stats.get("embedded_attachment_count"),
        "has_xfa": structural_stats.get("has_xfa"),
        "has_open_destination": structural_stats.get("has_open_destination"),
        "page_mode": structural_stats.get("page_mode"),
        "walked_nodes": structural_stats.get("walked_nodes"),
    }


def analyze_attachment(
    filename: str,
    content_type: str,
    encoding: str,
    raw_payload,
) -> dict:
    """Analyze an attachment and flag extension/content/magic-byte mismatches."""
    entry: dict = {
        "filename": filename,
        "content_type": content_type,
        "encoding": encoding,
        "magic_bytes_hex": None,
        "magic_detected_format": None,
        "extension_from_filename": ext_from_filename(filename),
        "extension_match": None,
        "anomaly": None,
        "hash_md5": None,
        "hash_sha1": None,
        "hash_sha256": None,
        "size_bytes": None,
        "pdf_security": None,
    }

    raw_bytes, payload_warning = _payload_to_bytes(raw_payload)
    if raw_bytes is None:
        entry["anomaly"] = payload_warning
        return entry

    entry["magic_bytes_hex"] = raw_bytes[:16].hex().upper()
    entry["magic_detected_format"] = identify_magic_bytes(raw_bytes)
    entry["size_bytes"] = len(raw_bytes)
    entry["hash_md5"] = hashlib.md5(raw_bytes).hexdigest()
    entry["hash_sha1"] = hashlib.sha1(raw_bytes).hexdigest()
    entry["hash_sha256"] = hashlib.sha256(raw_bytes).hexdigest()

    if entry["magic_detected_format"] == "pdf" or entry["extension_from_filename"] == "pdf":
        entry["pdf_security"] = analyze_pdf_security(raw_bytes)

    ct_base = content_type.split(";", 1)[0].strip().lower()
    expected_exts = CONTENT_TYPE_TO_EXT.get(ct_base, [])
    file_ext = entry["extension_from_filename"]
    magic_fmt = entry["magic_detected_format"]

    mismatches = []
    if file_ext and expected_exts and file_ext not in expected_exts:
        mismatches.append(
            f"Content-Type '{ct_base}' expects {expected_exts} but filename has '.{file_ext}'"
        )
    if magic_fmt and file_ext and magic_fmt != file_ext:
        if not (magic_fmt == "zip" and file_ext in ZIP_CONTAINER_EXTS):
            mismatches.append(
                f"Magic bytes identify format as '{magic_fmt}' but filename extension is '.{file_ext}'"
            )
    if magic_fmt and expected_exts and magic_fmt not in expected_exts:
        if not (magic_fmt == "zip" and bool(set(expected_exts) & ZIP_CONTAINER_EXTS)):
            mismatches.append(
                f"Magic bytes identify '{magic_fmt}' but Content-Type expects {expected_exts}"
            )

    entry["extension_match"] = not mismatches
    anomaly_parts = [part for part in (payload_warning, "; ".join(mismatches)) if part]
    pdf_security = entry.get("pdf_security") or {}
    if pdf_security.get("suspicious"):
        anomaly_parts.append(
            f"PDF risk {str(pdf_security.get('risk_level')).upper()}: {pdf_security.get('summary')}"
        )
    entry["anomaly"] = "; ".join(anomaly_parts) if anomaly_parts else None
    return entry
