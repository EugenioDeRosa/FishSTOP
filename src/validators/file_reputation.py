import datetime
import base64
import requests

VIRUSTOTAL_ENDPOINT = "https://www.virustotal.com/api/v3/files"
VIRUSTOTAL_URL_ENDPOINT = "https://www.virustotal.com/api/v3/urls"
_session = requests.Session()

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
        "message":          " — ".join(message_parts),
    }

def check_file_hash(api_key: str, sha256: str) -> dict:
    base = {
        "sha256":           sha256,
        "malicious":        0,
        "suspicious":       0,
        "undetected":       0,
        "total_engines":    0,
        "detection_ratio":  "—",
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
        return {**base, "status": "skipped", "message": "API key VirusTotal non configurata — lookup saltato"}

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
        return {**base, "status": "error", "message": f"Errore VirusTotal: {exc}"}


def _url_id(url: str) -> str:
    encoded = base64.urlsafe_b64encode(url.encode("utf-8")).decode("ascii")
    return encoded.rstrip("=")


def _format_vt_url(data: dict, base: dict) -> dict:
    attrs = data.get("data", {}).get("attributes", {})
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

    return {
        **base,
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
        "final_url": attrs.get("last_final_url") or base["url"],
        "message": f"{detections} engine su {total} segnalano questa URL",
    }


def check_url(api_key: str, url: str) -> dict:
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
        "final_url": url or "",
        "permalink": f"https://www.virustotal.com/gui/url/{_url_id(url) if url else ''}",
        "message": "",
    }

    if not url:
        return {**base, "status": "skipped", "message": "Nessuna URL fornita"}
    if not api_key:
        return {**base, "status": "skipped", "message": "API key VirusTotal non configurata - lookup saltato"}

    headers = {"x-apikey": api_key, "Accept": "application/json"}
    try:
        resp = _session.get(f"{VIRUSTOTAL_URL_ENDPOINT}/{_url_id(url)}", headers=headers, timeout=10)
        resp.raise_for_status()
        return _format_vt_url(resp.json(), base)
    except requests.exceptions.HTTPError as exc:
        code = exc.response.status_code
        if code == 404:
            return {**base, "status": "not_found", "message": "URL non trovata su VirusTotal"}
        if code == 401:
            return {**base, "status": "error", "message": "API key VirusTotal non valida (HTTP 401)"}
        if code == 429:
            return {**base, "status": "error", "message": "Rate limit VirusTotal superato"}
        return {**base, "status": "error", "message": f"VirusTotal HTTP {code}: {exc.response.reason}"}
    except requests.exceptions.Timeout:
        return {**base, "status": "error", "message": "Timeout durante il lookup VirusTotal"}
    except Exception as exc:
        return {**base, "status": "error", "message": f"Errore VirusTotal URL: {exc}"}
