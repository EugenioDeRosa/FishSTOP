import ipaddress

import requests

IPWHO_ENDPOINT = "https://ipwho.is/{ip}"


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
        "security_data_available": False,
        "provider": "ipwho.is",
    }

    if not ip:
        return {**base, "status": "skipped", "message": "No IP fornito"}

    if not _is_geolocatable_ip(ip):
        return {
            **base,
            "status": "skipped",
            "message": f"`{ip}` is not a public IP address (private, reserved, or invalid)",
        }

    normalized_ip = str(ipaddress.ip_address(ip.strip("[]")))

    try:
        resp = requests.get(
            IPWHO_ENDPOINT.format(ip=normalized_ip),
            headers={"User-Agent": "FishStop/1.0", "Accept": "application/json"},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.HTTPError as exc:
        status_code = getattr(exc.response, "status_code", None)
        if status_code == 429:
            message = "ipwho.is rate limit exceeded; try again later"
        else:
            message = f"ipwho.is HTTP error: {exc}"
        return {**base, "status": "error", "message": message}
    except requests.exceptions.RequestException as exc:
        return {**base, "status": "error", "message": f"Error ipwho.is: {exc}"}
    except (TypeError, ValueError) as exc:
        return {**base, "status": "error", "message": f"Invalid ipwho.is response: {exc}"}

    if not isinstance(data, dict):
        return {**base, "status": "error", "message": "Invalid ipwho.is response"}

    if not data.get("success", False):
        return {
            **base,
            "status": "skipped",
            "message": f"ipwho.is: {data.get('message', 'risposta non valida')} per `{normalized_ip}`",
        }

    connection = data.get("connection") or {}
    timezone = data.get("timezone") or {}
    security = data.get("security")
    security_data_available = isinstance(security, dict)
    security = security if security_data_available else {}

    city = data.get("city", "")
    region = data.get("region", "")
    country = data.get("country", "")
    country_code = data.get("country_code", "")
    location = ", ".join(part for part in (city, region, country) if part)
    if country_code:
        location = f"{location} ({country_code})" if location else country_code

    return {
        **base,
        "status": "ok",
        "ip": data.get("ip", normalized_ip),
        "country": country,
        "country_code": country_code,
        "region": region,
        "city": city,
        "zip": data.get("postal", ""),
        "lat": data.get("latitude"),
        "lon": data.get("longitude"),
        "timezone": timezone.get("id", ""),
        "isp": connection.get("isp", ""),
        "org": connection.get("org", ""),
        "asn": connection.get("asn", ""),
        "is_proxy": bool(
            security.get("proxy") or security.get("vpn") or security.get("tor")
        ),
        "is_hosting": bool(security.get("hosting")),
        "security_data_available": security_data_available,
        "message": location or f"Geolocation available for {normalized_ip}",
    }
