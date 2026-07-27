import re
import json
import socket
import ipaddress
import subprocess
import requests

# Config centralizzata
from src.config import get_secret

ABUSEIPDB_ENDPOINT  = "https://api.abuseipdb.com/api/v2/check"

# ── UTILS DI FORMATTAZIONE E CHIAMATE API ─────────────────────────────────

def _abuseipdb_call(ip: str) -> dict:
    """Esegue la chiamata ad AbuseIPDB usando la sessione Keep-Alive globale."""
    params = {"ipAddress": ip, "maxAgeInDays": "90"}
    headers = {"Key": get_secret("ABUSEIPDB_API_KEY")}
    response = requests.get(
        ABUSEIPDB_ENDPOINT,
        params=params,
        headers={**headers, "Accept": "application/json"},
        timeout=4,
    )
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
        "message": f"Score: {score}/100 - {int(data.get('totalReports') or 0)} reports.",
    }


# ── FUNZIONI PRINCIPALI EXPORTATE DA __INIT__.PY ──────────────────────────

def check_ip_reputation(ip: str) -> dict:
    base = {"ip": ip, "abuseConfidenceScore": 0, "totalReports": 0, "numDistinctUsers": 0, "isWhitelisted": False}
    if not ip: return {**base, "status": "skipped", "message": "No IP"}
    try:
        if not ipaddress.ip_address(ip.strip("[]")).is_global:
            return {**base, "status": "skipped", "message": "IP is not public/geolocatable"}
    except ValueError:
        return {**base, "status": "skipped", "message": "Invalid IP"}
    if not get_secret("ABUSEIPDB_API_KEY"): return {**base, "status": "skipped", "message": "API key missing"}
    try:
        data = _abuseipdb_call(ip)
        return _format_abuseipdb(data, ip)
    except Exception as exc:
        return {**base, "status": "error", "message": f"Error AbuseIPDB: {exc}"}


def check_domain_reputation(domain: str, resolver = None) -> dict:
    """
    Resolves the domain by emulating system NSLOOKUP.
    Bypassa completamente i socket interni di Python e le librerie DNS instabili.
    """
    base = {"domain_queried": domain, "resolved_ip": "", "lookup_method": "error", "abuseConfidenceScore": 0}
    if not domain: return {**base, "status": "skipped", "message": "No domain"}
    if not get_secret("ABUSEIPDB_API_KEY"): return {**base, "status": "skipped", "message": "API key missing"}

    resolved_ip = None
    lookup_method = "system-nslookup-subprocess"

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
            # Skip the first occurrence, which is usually the local DNS server IP itself
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
            lookup_method = "system-socket-fallback"
        except Exception as exc:
            return {**base, "status": "skipped", "message": f"Nslookup and socket failed for the domain. Error: {exc}"}

    # AbuseIPDB call with the clean IP obtained from nslookup
    try:
        data = _abuseipdb_call(resolved_ip)
        result_dict = _format_abuseipdb(data, resolved_ip)
        result_dict.update({
            "domain_queried": domain, 
            "resolved_ip": resolved_ip, 
            "lookup_method": lookup_method
        })
        return result_dict
    except Exception as exc:
        return {**base, "status": "error", "message": f"Error API AbuseIPDB su IP {resolved_ip}: {exc}"}
    
if __name__ == "__main__":
    print("=== TEST REPUTAZIONE IP STANDALONE ===")
    # Test veloce di controllo locale se viene eseguito direttamente il file
    if get_secret("ABUSEIPDB_API_KEY"):
        print(json.dumps(check_ip_reputation("8.8.8.8"), indent=2))
    else:
        print("API key not configured. Skipping the local test.")
