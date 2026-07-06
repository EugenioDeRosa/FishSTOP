"""SPF header-only check.

FishSTOP usa il risultato SPF gia calcolato dall'MTA e presente negli header
dell'EML.
"""

import re
from typing import Optional


def _extract_address(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    match = re.search(r"<([^>]+)>", raw)
    if match and match.group(1).strip():
        return match.group(1).strip()
    fallback = re.search(r"[\w.+\-]+@[\w.\-]+", raw)
    return fallback.group(0).strip() if fallback else None


def _extract_domain(email: str) -> str:
    match = re.search(r"@([\w.\-]+)", email or "")
    return match.group(1).lower() if match else ""


def check_spf(
    resolver,
    sender_ip: str,
    mail_from: str,
    helo_domain: str = "",
) -> dict:
    address = _extract_address(mail_from) or ""
    domain = _extract_domain(address) or (helo_domain or "").lower().strip()

    return {
        "status": "none",
        "record": "",
        "domain": domain,
        "sender_ip": sender_ip,
        "mail_from": address or "<>",
        "message": "Controllo SPF live disattivato: usare il risultato SPF presente negli header EML",
        "library": "header-only",
    }
