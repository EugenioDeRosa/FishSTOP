"""Attachment analysis helpers."""

import hashlib
from typing import Optional

from .constants import CONTENT_TYPE_TO_EXT, MAGIC_BYTES

ZIP_CONTAINER_EXTS = {"docx", "xlsx", "pptx", "zip"}


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
    entry["anomaly"] = "; ".join(anomaly_parts) if anomaly_parts else None
    return entry
