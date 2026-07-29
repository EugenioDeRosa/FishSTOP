import datetime
import base64
import ipaddress
import os
import socket
from urllib.parse import urljoin, urlparse

import requests

VIRUSTOTAL_ENDPOINT = "https://www.virustotal.com/api/v3/files"
VIRUSTOTAL_URL_ENDPOINT = "https://www.virustotal.com/api/v3/urls"
# The module-level requests API creates an independent Session per call, avoiding
# sharing a mutable requests.Session across the URL lookup thread pool.
_session = requests
MAX_REDIRECTS = 5
URL_DESTINATION_CHECK_ENABLED = (
    os.getenv("FISHSTOP_ENABLE_URL_DESTINATION_CHECK", "0").strip().lower()
    in {"1", "true", "yes", "on"}
)


def _epoch_to_iso(val) -> str:
    if not val:
        return ""
    try:
        return datetime.datetime.fromtimestamp(
            int(val), tz=datetime.timezone.utc
        ).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return str(val)

def _format_vt_file(data: dict, base: dict) -> dict:
    attrs = data.get("data", {}).get("attributes", {})
    stats = attrs.get("last_analysis_stats", {})

    malicious  = int(stats.get("malicious",  0))
    suspicious = int(stats.get("suspicious", 0))
    undetected = int(stats.get("undetected", 0))
    harmless   = int(stats.get("harmless",   0))
    total      = malicious + suspicious + undetected + harmless

    ptc          = attrs.get("popular_threat_classification") or {}
    threat_label = ptc.get("suggested_threat_label", "")
    file_type    = attrs.get("type_description", "")
    names        = attrs.get("names") or []
    file_name    = names[0] if names else ""

    status = "unknown"
    if malicious > 0: status = "malicious"
    elif suspicious > 0: status = "suspicious"
    elif total > 0: status = "clean"

    detection_ratio = f"{malicious + suspicious} / {total}" if total else "0 / 0"
    message_parts = [f"{malicious} engine su {total} lo segnalano come malevolo"]
    if suspicious: message_parts.append(f"{suspicious} come sospetto")
    if threat_label: message_parts.append(f"minaccia rilevata: {threat_label}")

    return {
        **base,
        "status":           status,
        "malicious":        malicious,
        "suspicious":       suspicious,
        "undetected":       undetected,
        "total_engines":    total,
        "detection_ratio":  detection_ratio,
        "threat_label":     threat_label,
        "file_type":        file_type,
        "file_name":        file_name,
        "first_submission": _epoch_to_iso(attrs.get("first_submission_date")),
        "last_analysis":    _epoch_to_iso(attrs.get("last_analysis_date")),
        "message":          " - ".join(message_parts),
    }

def check_file_hash(api_key: str, sha256: str) -> dict:
    base = {
        "sha256":           sha256,
        "malicious":        0,
        "suspicious":       0,
        "undetected":       0,
        "total_engines":    0,
        "detection_ratio":  "-",
        "threat_label":     "",
        "file_type":        "",
        "file_name":        "",
        "first_submission": "",
        "last_analysis":    "",
        "permalink":        f"https://www.virustotal.com/gui/file/{sha256}",
    }

    if not sha256:
        return {**base, "status": "skipped", "message": "Nessun hash fornito"}
    if not api_key:
        return {**base, "status": "skipped", "message": "API key VirusTotal non configurata - lookup saltato"}

    url = f"{VIRUSTOTAL_ENDPOINT}/{sha256}"
    headers = {"x-apikey": api_key, "Accept": "application/json"}

    try:
        resp = _session.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        return _format_vt_file(resp.json(), base)
        
    except requests.exceptions.HTTPError as exc:
        code = exc.response.status_code
        if code == 404:
            return {**base, "status": "not_found", "message": "Hash non trovato su VirusTotal"}
        if code == 401:
            return {**base, "status": "error", "message": "API key VirusTotal non valida (HTTP 401)"}
        if code == 429:
            return {**base, "status": "error", "message": "Rate limit VirusTotal superato (4 req/min)"}
        return {**base, "status": "error", "message": f"VirusTotal HTTP {code}: {exc.response.reason}"}
    except Exception as exc:
        return {**base, "status": "error", "message": f"Error VirusTotal: {exc}"}


def _url_id(url: str) -> str:
    encoded = base64.urlsafe_b64encode(url.encode("utf-8")).decode("ascii")
    return encoded.rstrip("=")


def _host(value: str) -> str:
    try:
        return (urlparse(value or "").hostname or "").lower()
    except Exception:
        return ""


def _registered_domain(host: str) -> str:
    parts = (host or "").lower().rstrip(".").split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host or ""


def _destination_base(url: str) -> dict:
    return {
        "final_url": url or "",
        "final_host": _host(url),
        "destination_status": "not_checked",
        "destination_match": False,
        "destination_message": "",
        "redirect_count": 0,
    }


def _public_http_url(url: str) -> tuple[bool, str]:
    parsed = urlparse(url or "")
    if parsed.scheme.lower() not in {"http", "https"}:
        return False, "Only HTTP(S) destinations are allowed"
    if not parsed.hostname:
        return False, "Destination hostname is missing"
    if parsed.username or parsed.password:
        return False, "Destination URLs containing credentials are not allowed"

    try:
        addresses = {
            item[4][0]
            for item in socket.getaddrinfo(
                parsed.hostname,
                parsed.port or (443 if parsed.scheme.lower() == "https" else 80),
                type=socket.SOCK_STREAM,
            )
        }
    except (socket.gaierror, socket.timeout, UnicodeError, ValueError) as exc:
        return False, f"Destination DNS lookup failed: {exc}"

    if not addresses:
        return False, "Destination DNS lookup returned no addresses"
    try:
        if any(not ipaddress.ip_address(address).is_global for address in addresses):
            return False, "Private, loopback, reserved, or otherwise non-public destinations are blocked"
    except ValueError:
        return False, "Destination resolved to an invalid IP address"
    return True, ""


def _check_destination(url: str) -> dict:
    info = _destination_base(url)
    if not url:
        info.update({"destination_status": "skipped", "destination_message": "No URL provided"})
        return info

    original_host = _host(url)
    current_url = url
    redirect_count = 0
    try:
        while True:
            allowed, reason = _public_http_url(current_url)
            if not allowed:
                info.update({
                    "final_url": current_url,
                    "final_host": _host(current_url),
                    "destination_status": "blocked",
                    "destination_message": reason,
                    "redirect_count": redirect_count,
                })
                return info

            resp = _session.get(
                current_url,
                allow_redirects=False,
                headers={"User-Agent": "FishStop/1.0"},
                timeout=5,
                stream=True,
            )
            status_code = int(getattr(resp, "status_code", 200) or 200)
            location = (getattr(resp, "headers", {}) or {}).get("Location")
            if status_code in {301, 302, 303, 307, 308} and location:
                close = getattr(resp, "close", None)
                if callable(close):
                    close()
                redirect_count += 1
                if redirect_count > MAX_REDIRECTS:
                    info.update({
                        "final_url": current_url,
                        "final_host": _host(current_url),
                        "destination_status": "unavailable",
                        "destination_message": f"Too many redirects (limit: {MAX_REDIRECTS})",
                        "redirect_count": redirect_count,
                    })
                    return info
                current_url = urljoin(current_url, location)
                continue
            final_url = getattr(resp, "url", current_url) or current_url
            break

        final_host = _host(final_url)
        destination_match = bool(
            original_host
            and final_host
            and _registered_domain(original_host) == _registered_domain(final_host)
        )
        info.update({
            "final_url": final_url,
            "final_host": final_host,
            "destination_status": "match" if destination_match else "mismatch",
            "destination_match": destination_match,
            "destination_message": "Destination matches original domain" if destination_match else "Destination differs from original domain",
            "redirect_count": redirect_count,
        })
        close = getattr(resp, "close", None)
        if callable(close):
            close()
        return info
    except requests.exceptions.Timeout as exc:
        info.update({
            "destination_status": "unavailable",
            "destination_message": f"Timeout while checking destination: {exc}",
        })
        return info
    except Exception as exc:
        info.update({
            "destination_status": "unavailable",
            "destination_message": f"Destination check unavailable: {exc}",
        })
        return info


def _as_list(value) -> list:
    if not value:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("data", "items", "results"):
            if isinstance(value.get(key), list):
                return value[key]
        if value.get("attributes") or value.get("title") or value.get("severity") or value.get("context"):
            return [value]
        return list(value.values())
    return [value]


def _short_text(value, limit: int = 220) -> str:
    value = str(value or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "..."


def _normalize_context_details(value) -> list[str]:
    details: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if item in (None, "", [], {}):
                continue
            details.append(_short_text(f"{key}: {item}"))
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                details.extend(_normalize_context_details(item))
            elif item not in (None, ""):
                details.append(_short_text(item))
    elif value not in (None, ""):
        details.append(_short_text(value))
    return details[:8]


def _extract_crowdsourced_context(attrs: dict) -> list[dict]:
    raw_items = (
        attrs.get("crowdsourced_context")
        or attrs.get("crowdsourced_contexts")
        or attrs.get("crowdsourced_context_items")
        or []
    )
    context_items: list[dict] = []
    for raw in _as_list(raw_items):
        if not isinstance(raw, dict):
            continue
        item = raw.get("attributes") if isinstance(raw.get("attributes"), dict) else raw
        title = (
            item.get("title")
            or item.get("name")
            or item.get("summary")
            or item.get("context")
            or item.get("indicator")
            or "Crowdsourced context"
        )
        severity = str(item.get("severity") or item.get("level") or item.get("type") or "INFO").upper()
        if severity not in {"HIGH", "MEDIUM", "LOW", "INFO", "SUCCESS"}:
            severity = "INFO"
        source = item.get("source") or item.get("source_name") or item.get("provider") or ""
        details = []
        for key in ("details", "description", "classification_description", "contextual_indicators", "metadata"):
            details.extend(_normalize_context_details(item.get(key)))
        context_items.append({
            "severity": severity,
            "title": _short_text(title, 180),
            "source": _short_text(source, 120),
            "date": _epoch_to_iso(item.get("timestamp") or item.get("date") or item.get("created") or item.get("created_at")),
            "details": details[:8],
        })
    return context_items[:10]


def _summarize_crowdsourced_context(items: list[dict]) -> str:
    if not items:
        return ""
    counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "INFO": 0, "SUCCESS": 0}
    for item in items:
        severity = item.get("severity") or "INFO"
        counts[severity] = counts.get(severity, 0) + 1
    count_text = ", ".join(f"{key}={value}" for key, value in counts.items() if value)
    highlights = []
    for item in items[:3]:
        source = f" source={item['source']}" if item.get("source") else ""
        highlights.append(f"{item.get('severity', 'INFO')} {item.get('title', 'context')}{source}")
    return "Crowdsourced context: " + count_text + "; " + " | ".join(highlights)


def _format_vt_url(data: dict, base: dict) -> dict:
    url_object = data.get("data", {})
    attrs = url_object.get("attributes", {})
    stats = attrs.get("last_analysis_stats", {})

    malicious = int(stats.get("malicious", 0))
    suspicious = int(stats.get("suspicious", 0))
    harmless = int(stats.get("harmless", 0))
    undetected = int(stats.get("undetected", 0))
    timeout = int(stats.get("timeout", 0))
    total = malicious + suspicious + harmless + undetected + timeout
    detections = malicious + suspicious

    status = "unknown"
    if malicious > 0:
        status = "malicious"
    elif suspicious > 0:
        status = "suspicious"
    elif total > 0:
        status = "clean"

    crowdsourced_context = _extract_crowdsourced_context(attrs)
    crowdsourced_context_summary = _summarize_crowdsourced_context(crowdsourced_context)

    return {
        **base,
        "permalink": (
            f"https://www.virustotal.com/gui/url/{url_object['id']}"
            if url_object.get("id")
            else base.get("permalink", "")
        ),
        "status": status,
        "malicious": malicious,
        "suspicious": suspicious,
        "harmless": harmless,
        "undetected": undetected,
        "timeout": timeout,
        "total_engines": total,
        "detection_ratio": f"{detections} / {total}" if total else "0 / 0",
        "last_analysis": _epoch_to_iso(attrs.get("last_analysis_date")),
        "reputation": attrs.get("reputation", 0),
        "title": attrs.get("title", ""),
        "crowdsourced_context": crowdsourced_context,
        "crowdsourced_context_summary": crowdsourced_context_summary,
        "message": f"{detections} engine su {total} segnalano questa URL",
    }


def check_url(api_key: str, url: str) -> dict:
    destination = _destination_base(url)
    base = {
        "url": url or "",
        "status": "skipped",
        "malicious": 0,
        "suspicious": 0,
        "harmless": 0,
        "undetected": 0,
        "timeout": 0,
        "total_engines": 0,
        "detection_ratio": "0 / 0",
        "last_analysis": "",
        "reputation": 0,
        "title": "",
        "crowdsourced_context": [],
        "crowdsourced_context_summary": "",
        "permalink": "https://www.virustotal.com/gui/home/url",
        "message": "",
        **destination,
    }

    if not url:
        return {**base, "status": "skipped", "message": "No URL provided"}
    if not api_key:
        return {
            **base,
            "status": "skipped",
            "message": "VirusTotal API key is not configured - lookup skipped",
        }

    if URL_DESTINATION_CHECK_ENABLED:
        destination = _check_destination(url)
        base.update(destination)
        if destination.get("destination_status") == "blocked":
            return {
                **base,
                "status": "blocked",
                "message": destination.get("destination_message") or "Unsafe destination blocked",
            }
    else:
        base.update({
            "destination_status": "skipped",
            "destination_message": (
                "Direct destination checks are disabled by default to prevent server-side "
                "requests to attacker-controlled URLs"
            ),
        })

    headers = {"x-apikey": api_key, "Accept": "application/json"}
    try:
        resp = _session.get(f"{VIRUSTOTAL_URL_ENDPOINT}/{_url_id(url)}", headers=headers, timeout=10)
        resp.raise_for_status()
        return _format_vt_url(resp.json(), base)
    except requests.exceptions.HTTPError as exc:
        code = exc.response.status_code
        if code == 404:
            return {
                **base,
                "status": "not_found",
                "message": "URL not found su VirusTotal",
            }
        if code == 401:
            return {**base, "status": "error", "message": "API key VirusTotal non valida (HTTP 401)"}
        if code == 429:
            return {**base, "status": "error", "message": "Rate limit VirusTotal superato"}
        return {**base, "status": "error", "message": f"VirusTotal HTTP {code}: {exc.response.reason}"}
    except requests.exceptions.Timeout:
        return {**base, "status": "error", "message": "Timeout durante il lookup VirusTotal"}
    except Exception as exc:
        return {**base, "status": "error", "message": f"Error VirusTotal URL: {exc}"}
