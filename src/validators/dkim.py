"""DKIM header-only check.

Legge solo le firme DKIM gia presenti nell'EML. Il risultato autorevole
pass/fail viene preso dagli header Authentication-Results nella vista di analisi.
"""

import email
import re


def check_dkim(raw_eml_bytes: bytes) -> dict:
    msg = email.message_from_bytes(raw_eml_bytes)
    sig_headers = [
        value
        for key, value in msg.items()
        if key.lower() == "dkim-signature"
    ]

    if not sig_headers:
        return {
            "status": "none",
            "signatures": [],
            "message": "No DKIM signature present in the email",
            "library": "header-only",
        }

    signatures = []
    for idx, sig_raw in enumerate(sig_headers):
        d_match = re.search(r"\bd=([^\s;]+)", sig_raw)
        s_match = re.search(r"\bs=([^\s;]+)", sig_raw)
        d_domain = d_match.group(1).rstrip(";") if d_match else ""
        selector = s_match.group(1).rstrip(";") if s_match else ""
        signatures.append({
            "index": idx,
            "raw_header": sig_raw,
            "d_domain": d_domain or "?",
            "selector": selector or "?",
            "result": "present",
            "message": "DKIM signature present in the EML; no live check executed",
        })

    return {
        "status": "present",
        "signatures": signatures,
        "message": f"{len(signatures)} DKIM signature(s) present; no live check executed",
        "library": "header-only",
    }
