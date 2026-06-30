"""
validators/urlhaus.py - Lookup reputazione URL e host tramite URLhaus.

URLhaus espone endpoint pubblici senza API key per verificare se una URL o
un host sono presenti nel suo feed di malware URL.
"""

from __future__ import annotations

from urllib.parse import urlparse

import requests


URLHAUS_URL_ENDPOINT = "https://urlhaus-api.abuse.ch/v1/url/"
URLHAUS_HOST_ENDPOINT = "https://urlhaus-api.abuse.ch/v1/host/"

_session = requests.Session()
_session.headers.update({"Accept": "application/json"})


def _headers(api_key: str = "") -> dict:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Auth-Key"] = api_key
    return headers


def _host_from_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url if "://" in url else f"http://{url}")
    return (parsed.hostname or "").lower()


def _base(url: str, host: str = "") -> dict:
    return {
        "status": "skipped",
        "query_status": "",
        "url": url or "",
        "host": host or _host_from_url(url),
        "url_status": "",
        "threat": "",
        "tags": [],
        "blacklists": {},
        "payloads": [],
        "source": "",
        "permalink": "",
        "message": "",
    }


def _format_url_result(data: dict, base: dict) -> dict:
    url_status = data.get("url_status") or ""
    threat = data.get("threat") or ""
    tags = data.get("tags") or []
    payloads = data.get("payloads") or []

    status = "malicious" if url_status == "online" else "suspicious"
    message = "URL presente su URLhaus"
    if threat:
        message += f" - threat: {threat}"
    if url_status:
        message += f" - stato: {url_status}"

    return {
        **base,
        "status": status,
        "query_status": data.get("query_status", "ok"),
        "url_status": url_status,
        "threat": threat,
        "tags": tags,
        "blacklists": data.get("blacklists") or {},
        "payloads": payloads,
        "source": "url",
        "permalink": data.get("urlhaus_reference") or "",
        "message": message,
    }


def _format_host_result(data: dict, base: dict) -> dict:
    urls = data.get("urls") or []
    online = sum(1 for item in urls if item.get("url_status") == "online")
    status = "malicious" if online else "suspicious"

    return {
        **base,
        "status": status,
        "query_status": data.get("query_status", "ok"),
        "source": "host",
        "host": data.get("host") or base["host"],
        "payloads": urls[:10],
        "message": (
            f"Host presente su URLhaus: {len(urls)} URL note"
            + (f", {online} ancora online" if online else "")
        ),
    }


def check_urlhaus(url: str, host: str = "", api_key: str = "") -> dict:
    """
    Controlla prima la URL completa; se non e' presente, controlla l'host.

    Returns:
        status:
          malicious  - URL/host noto e ancora online
          suspicious - URL/host noto ma non online o solo match host
          not_found  - nessun risultato URLhaus
          skipped    - input mancante
          error      - problema HTTP/API
    """
    host = (host or _host_from_url(url)).lower()
    base = _base(url, host)

    if not url and not host:
        return {**base, "status": "skipped", "message": "Nessuna URL o host da verificare"}

    try:
        if url:
            response = _session.post(
                URLHAUS_URL_ENDPOINT,
                data={"url": url},
                headers=_headers(api_key),
                timeout=8,
            )
            response.raise_for_status()
            data = response.json()
            query_status = data.get("query_status", "")
            if query_status == "ok":
                return _format_url_result(data, base)
            if query_status not in ("no_results", "invalid_url"):
                return {
                    **base,
                    "status": "error",
                    "query_status": query_status,
                    "message": f"URLhaus ha restituito query_status={query_status}",
                }

        if host:
            response = _session.post(
                URLHAUS_HOST_ENDPOINT,
                data={"host": host},
                headers=_headers(api_key),
                timeout=8,
            )
            response.raise_for_status()
            data = response.json()
            query_status = data.get("query_status", "")
            if query_status == "ok":
                return _format_host_result(data, base)
            if query_status not in ("no_results", "invalid_host"):
                return {
                    **base,
                    "status": "error",
                    "query_status": query_status,
                    "message": f"URLhaus host query_status={query_status}",
                }

        return {
            **base,
            "status": "not_found",
            "query_status": "no_results",
            "message": "URL/host non presente su URLhaus",
        }

    except requests.exceptions.Timeout:
        return {**base, "status": "error", "message": "Timeout durante il lookup URLhaus"}
    except requests.exceptions.RequestException as exc:
        return {**base, "status": "error", "message": f"Errore HTTP URLhaus: {exc}"}
    except Exception as exc:
        return {**base, "status": "error", "message": f"Errore URLhaus: {exc}"}
