import ipaddress

import requests

_IPAPI_FIELDS = (
    "status,message,country,countryCode,regionName,city,"
    "zip,lat,lon,timezone,isp,org,as,proxy,hosting,query"
)
IPAPI_ENDPOINT = "http://ip-api.com/json/{ip}?fields=" + _IPAPI_FIELDS

_session = requests.Session()
_session.headers.update({"User-Agent": "FishStop/1.0", "Accept": "application/json"})


def _is_geolocatable_ip(ip: str) -> bool:
    try:
        return ipaddress.ip_address((ip or "").strip("[]")).is_global
    except ValueError:
        return False


def geolocate_ip(ip: str) -> dict:
    base = {
        "ip": ip,
        "country": "",
        "country_code": "",
        "region": "",
        "city": "",
        "zip": "",
        "lat": None,
        "lon": None,
        "timezone": "",
        "isp": "",
        "org": "",
        "asn": "",
        "is_proxy": False,
        "is_hosting": False,
    }

    if not ip:
        return {**base, "status": "skipped", "message": "Nessun IP fornito"}

    if not _is_geolocatable_ip(ip):
        return {
            **base,
            "status": "skipped",
            "message": f"`{ip}` non e' un indirizzo pubblico geolocalizzabile",
        }

    try:
        resp = _session.get(IPAPI_ENDPOINT.format(ip=ip), timeout=5)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as exc:
        return {**base, "status": "error", "message": f"Errore ip-api.com: {exc}"}

    if data.get("status") != "success":
        return {
            **base,
            "status": "skipped",
            "message": f"ip-api.com: {data.get('message', 'risposta non valida')} per `{ip}`",
        }

    city = data.get("city", "")
    region = data.get("regionName", "")
    country = data.get("country", "")
    country_code = data.get("countryCode", "")
    proxy_note = " - Proxy/VPN rilevato" if data.get("proxy") else ""
    hosting_note = " - Datacenter/Hosting" if data.get("hosting") else ""

    return {
        "status": "ok",
        "ip": data.get("query", ip),
        "country": country,
        "country_code": country_code,
        "region": region,
        "city": city,
        "zip": data.get("zip", ""),
        "lat": data.get("lat"),
        "lon": data.get("lon"),
        "timezone": data.get("timezone", ""),
        "isp": data.get("isp", ""),
        "org": data.get("org", ""),
        "asn": data.get("as", ""),
        "is_proxy": bool(data.get("proxy")),
        "is_hosting": bool(data.get("hosting")),
        "message": f"{city}, {region}, {country} ({country_code}){proxy_note}{hosting_note}",
    }
