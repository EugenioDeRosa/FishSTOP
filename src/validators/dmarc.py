"""DMARC header-only check.

FishSTOP usa il risultato DMARC gia calcolato dall'MTA e presente negli header
dell'EML.
"""

import re
from typing import Optional


def _extract_address(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    match = re.search(r"<([^>]+)>", raw)
    if match:
        return match.group(1).strip()
    fallback = re.search(r"[\w.+\-]+@[\w.\-]+", raw)
    return fallback.group(0).strip() if fallback else None


def _extract_domain(email_or_raw: str) -> str:
    address = _extract_address(email_or_raw) or email_or_raw
    match = re.search(r"@([\w.\-]+)", address or "")
    return match.group(1).lower() if match else ""


def _domains_aligned(check_domain: str, from_domain: str, mode: str = "r") -> bool:
    check_domain = (check_domain or "").lower().lstrip(".")
    from_domain = (from_domain or "").lower().lstrip(".")
    if not check_domain or not from_domain:
        return False
    if mode == "s":
        return check_domain == from_domain

    def org(domain: str) -> str:
        parts = domain.split(".")
        return ".".join(parts[-2:]) if len(parts) >= 2 else domain

    return org(check_domain) == org(from_domain)


def check_dmarc(
    resolver,
    from_address: str,
    spf_result: str,
    spf_domain: str,
    dkim_results: list,
) -> dict:
    from_addr = _extract_address(from_address) or from_address
    from_domain = _extract_domain(from_addr)

    base = {
        "domain": from_domain,
        "record": "",
        "policy": "none",
        "subdomain_policy": "none",
        "pct": 100,
        "adkim": "r",
        "aspf": "r",
        "spf_aligned": False,
        "dkim_aligned": False,
        "rua": "",
        "ruf": "",
    }

    if not from_domain:
        return {
            **base,
            "status": "error",
            "message": "Impossibile estrarre il dominio dall'header From",
        }

    spf_aligned = (
        spf_result == "pass"
        and bool(spf_domain)
        and _domains_aligned(spf_domain, from_domain)
    )
    dkim_aligned = any(
        sig.get("result") == "pass"
        and _domains_aligned(sig.get("d_domain", ""), from_domain)
        for sig in (dkim_results or [])
    )

    if spf_aligned or dkim_aligned:
        aligned_via = []
        if spf_aligned:
            aligned_via.append("SPF")
        if dkim_aligned:
            aligned_via.append("DKIM")
        return {
            **base,
            "spf_aligned": spf_aligned,
            "dkim_aligned": dkim_aligned,
            "status": "pass",
            "message": (
                f"DMARC PASS stimato dagli header locali tramite {' + '.join(aligned_via)}; "
                "nessun controllo live eseguito"
            ),
        }

    return {
        **base,
        "status": "none",
        "message": "Controllo DMARC live disattivato: usare il risultato DMARC presente negli header EML",
    }
