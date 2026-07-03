import re
import json
import socket
import ipaddress
import urllib.request
import urllib.parse
import urllib.error
import dns.resolver
import requests
from typing import Optional

# Config centralizzata
from src.config import ABUSEIPDB_API_KEY, VIRUSTOTAL_API_KEY

ABUSEIPDB_ENDPOINT  = "https://api.abuseipdb.com/api/v2/check"
VIRUSTOTAL_ENDPOINT = "https://www.virustotal.com/api/v3/files"

_IPAPI_FIELDS = (
    "status,message,country,countryCode,regionName,city,"
    "zip,lat,lon,timezone,isp,org,as,proxy,hosting,query"
)
IPAPI_ENDPOINT = "http://ip-api.com/json/{ip}?fields=" + _IPAPI_FIELDS

# Inizializziamo una sessione riutilizzabile Keep-Alive a livello di modulo
_session = requests.Session()
_session.headers.update({"Accept": "application/json"})

# Resolver di fallback a livello di modulo
_resolver = dns.resolver.Resolver()
_resolver.nameservers = [
    "1.1.1.1",  # Cloudflare
    "8.8.8.8",  # Google
    "9.9.9.9",  # Quad9
]
_resolver.timeout = 2.0
_resolver.lifetime = 6.0

def _resolve_domain_a(domain: str) -> Optional[str]:
    """Risolve un dominio usando il resolver di SISTEMA (bypassa dns.resolver/porta 53 custom)."""
    try:
        return socket.gethostbyname(domain)
    except (socket.gaierror, socket.timeout, UnicodeError):
        return None
    
def _extract_address(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    m = re.search(r"<([^>]+)>", raw)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"[\w.+\-]+@[\w.\-]+", raw)
    return m2.group(0).strip() if m2 else None


def _extract_domain(email_or_raw: str) -> str:
    addr = _extract_address(email_or_raw) or email_or_raw
    m = re.search(r"@([\w.\-]+)", addr)
    return m.group(1).lower() if m else ""


def _parse_dmarc_record(record: str) -> dict:
    tags: dict = {}
    for part in record.split(";"):
        part = part.strip()
        if "=" in part:
            k, _, v = part.partition("=")
            tags[k.strip().lower()] = v.strip().lower()
    return tags


# ── UTILS DI FORMATTAZIONE E CHIAMATE API ─────────────────────────────────

def _abuseipdb_call(ip: str) -> dict:
    """Esegue la chiamata ad AbuseIPDB usando la sessione Keep-Alive globale."""
    params = {"ipAddress": ip, "maxAgeInDays": "90"}
    headers = {"Key": ABUSEIPDB_API_KEY}
    response = _session.get(ABUSEIPDB_ENDPOINT, params=params, headers=headers, timeout=4)
    response.raise_for_status()
    return response.json().get("data", {})


def _format_abuseipdb(data: dict, lookup_key: str) -> dict:
    score = int(data.get("abuseConfidenceScore") or 0)
    ip    = data.get("ipAddress", lookup_key)
    return {
        "status":               "ok",
        "ip":                   ip,
        "abuseConfidenceScore": score,
        "totalReports":         int(data.get("totalReports") or 0),
        "numDistinctUsers":     int(data.get("numDistinctUsers") or 0),
        "countryCode":          data.get("countryCode") or "",
        "isp":                  data.get("isp") or "",
        "domain":               data.get("domain") or "",
        "isWhitelisted":        bool(data.get("isWhitelisted")),
        "usageType":            data.get("usageType") or "",
        "lastReportedAt":       data.get("lastReportedAt"),
        "url":                  f"https://www.abuseipdb.com/check/{ip}",
        "message": f"Score: {score}/100 — {int(data.get('totalReports') or 0)} segnalazioni.",
    }


def _format_vt_file(data: dict, base: dict) -> dict:
    attrs = data.get("data", {}).get("attributes", {})
    stats = attrs.get("last_analysis_stats", {})
    malicious = int(stats.get("malicious", 0))
    suspicious = int(stats.get("suspicious", 0))
    total = sum(int(v) for v in stats.values())
    return {
        **base,
        "status": "malicious" if malicious > 0 else "clean",
        "malicious": malicious,
        "suspicious": suspicious,
        "total_engines": total,
        "message": f"{malicious} engine rilevano minacce su {total}.",
    }


# ── FUNZIONI PRINCIPALI EXPORTATE DA __INIT__.PY ──────────────────────────

def check_ip_reputation(ip: str) -> dict:
    base = {"ip": ip, "abuseConfidenceScore": 0, "totalReports": 0, "numDistinctUsers": 0, "isWhitelisted": False}
    if not ip: return {**base, "status": "skipped", "message": "Nessun IP"}
    try:
        if not ipaddress.ip_address(ip.strip("[]")).is_global:
            return {**base, "status": "skipped", "message": "IP non pubblico geolocalizzabile"}
    except ValueError:
        return {**base, "status": "skipped", "message": "IP non valido"}
    if not ABUSEIPDB_API_KEY: return {**base, "status": "skipped", "message": "API key assente"}
    try:
        data = _abuseipdb_call(ip)
        return _format_abuseipdb(data, ip)
    except Exception as exc:
        return {**base, "status": "error", "message": f"Errore AbuseIPDB: {exc}"}


import subprocess
import re
import socket
from typing import Optional

def check_domain_reputation(domain: str, resolver = None) -> dict:
    """
    Risolve il dominio emulando NSLOOKUP di sistema.
    Bypassa completamente i socket interni di Python e le librerie DNS instabili.
    """
    base = {"domain_queried": domain, "resolved_ip": "", "lookup_method": "error", "abuseConfidenceScore": 0}
    if not domain: return {**base, "status": "skipped", "message": "Nessun dominio"}
    if not ABUSEIPDB_API_KEY: return {**base, "status": "skipped", "message": "API key assente"}

    resolved_ip = None

    try:
        # Eseguiamo letteralmente il comando 'nslookup' come fai da terminale
        # Impostiamo un timeout di 4 secondi per evitare blocchi infiniti
        result = subprocess.run(
            ["nslookup", domain], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            timeout=4
        )
        
        if result.returncode == 0:
            output = result.stdout
            
            # Estraiamo gli indirizzi IP IPv4 validi dall'output di nslookup.
            # Saltiamo la prima occorrenza (che di solito è l'IP del server DNS locale stesso)
            ip_finder = re.findall(r"Address:\s*([0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3})", output)
            
            if len(ip_finder) > 0:
                # Se l'output ha un'unica stringa 'Address', o prendiamo l'ultimo trovato se mostra prima il server DNS
                resolved_ip = ip_finder[-1]
    
    except subprocess.TimeoutExpired:
        pass # Se il processo appende, scatta il fallback sotto
    except Exception:
        pass

    # Se l'approccio nslookup fallisce per problemi di parsing, facciamo un fallback rapido sul socket standard
    if not resolved_ip:
        try:
            resolved_ip = socket.gethostbyname(domain)
        except Exception as exc:
            return {**base, "status": "skipped", "message": f"Nslookup e Socket falliti per il dominio. Errore: {exc}"}

    # Chiamata ad AbuseIPDB con l'IP pulito ottenuto da nslookup
    try:
        data = _abuseipdb_call(resolved_ip)
        result_dict = _format_abuseipdb(data, resolved_ip)
        result_dict.update({
            "domain_queried": domain, 
            "resolved_ip": resolved_ip, 
            "lookup_method": "system-nslookup-subprocess"
        })
        return result_dict
    except Exception as exc:
        return {**base, "status": "error", "message": f"Errore API AbuseIPDB su IP {resolved_ip}: {exc}"}
    
def check_file_hash(sha256: str) -> dict:
    base = {"sha256": sha256, "malicious": 0, "suspicious": 0, "total_engines": 0}
    if not sha256: return {**base, "status": "skipped", "message": "No hash"}
    if not VIRUSTOTAL_API_KEY: return {**base, "status": "skipped", "message": "VT key assente"}
    
    url = f"{VIRUSTOTAL_ENDPOINT}/{sha256}"
    headers = {"x-apikey": VIRUSTOTAL_API_KEY}
    try:
        response = _session.get(url, headers=headers, timeout=5)
        if response.status_code == 404:
            return {**base, "status": "not_found", "message": "Hash non trovato su VT"}
        response.raise_for_status()
        return _format_vt_file(response.json(), base)
    except Exception as exc:
        return {**base, "status": "error", "message": f"Errore VT: {exc}"}


def geolocate_ip(ip: str) -> dict:
    base = {"ip": ip, "country": "", "is_proxy": False, "is_hosting": False}
    if not ip or ip.startswith("10.") or ip.startswith("192.168.") or ip.startswith("127."):
        return {**base, "status": "skipped", "message": "IP privato o assente"}
    try:
        response = _session.get(IPAPI_ENDPOINT.format(ip=ip), timeout=4)
        response.raise_for_status()
        data = response.json()
        if data.get("status") != "success":
            return {**base, "status": "skipped", "message": data.get("message", "fail")}
        return {
            "status": "ok", "ip": data.get("query", ip), "country": data.get("country", ""),
            "is_proxy": bool(data.get("proxy")), "is_hosting": bool(data.get("hosting")),
            "message": f"{data.get('city')}, {data.get('country')}",
        }
    except Exception as exc:
        return {**base, "status": "error", "message": f"Errore geo: {exc}"}


if __name__ == "__main__":
    print("=== TEST REPUTAZIONE IP STANDALONE ===")
    # Test veloce di controllo locale se viene eseguito direttamente il file
    if ABUSEIPDB_API_KEY:
        print(json.dumps(check_ip_reputation("8.8.8.8"), indent=2))
    else:
        print("API Key non configurata. Salta il test locale.")
