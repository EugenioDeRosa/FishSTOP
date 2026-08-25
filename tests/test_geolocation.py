import requests

from src.validators import geolocation


class FakeResponse:
    def __init__(self, payload=None, *, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            response = requests.Response()
            response.status_code = self.status_code
            raise requests.exceptions.HTTPError(
                f"{self.status_code} response",
                response=response,
            )

    def json(self):
        return self.payload


def test_geolocate_ip_uses_https_and_maps_ipwho_response(monkeypatch):
    captured = {}

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return FakeResponse({
            "success": True,
            "ip": "8.8.8.8",
            "country": "United States",
            "country_code": "US",
            "region": "California",
            "city": "Mountain View",
            "postal": "94043",
            "latitude": 37.4,
            "longitude": -122.1,
            "timezone": {"id": "America/Los_Angeles"},
            "connection": {
                "asn": 15169,
                "org": "Google LLC",
                "isp": "Google LLC",
            },
        })

    monkeypatch.setattr(geolocation.requests, "get", fake_get)

    result = geolocation.geolocate_ip("8.8.8.8")

    assert captured["url"] == "https://ipwho.is/8.8.8.8"
    assert captured["kwargs"]["timeout"] == 5
    assert result["status"] == "ok"
    assert result["country_code"] == "US"
    assert result["timezone"] == "America/Los_Angeles"
    assert result["asn"] == 15169
    assert result["security_data_available"] is False
    assert result["is_proxy"] is False
    assert result["is_hosting"] is False


def test_geolocate_ip_maps_optional_security_data(monkeypatch):
    monkeypatch.setattr(
        geolocation.requests,
        "get",
        lambda *args, **kwargs: FakeResponse({
            "success": True,
            "ip": "1.1.1.1",
            "security": {
                "proxy": False,
                "vpn": True,
                "tor": False,
                "hosting": True,
            },
        }),
    )

    result = geolocation.geolocate_ip("1.1.1.1")

    assert result["security_data_available"] is True
    assert result["is_proxy"] is True
    assert result["is_hosting"] is True


def test_geolocate_ip_rejects_non_public_ip_without_network_call(monkeypatch):
    def unexpected_get(*args, **kwargs):
        raise AssertionError("Network request should not be made")

    monkeypatch.setattr(geolocation.requests, "get", unexpected_get)

    result = geolocation.geolocate_ip("127.0.0.1")

    assert result["status"] == "skipped"
    assert result["provider"] == "ipwho.is"


def test_geolocate_ip_handles_provider_error_response(monkeypatch):
    monkeypatch.setattr(
        geolocation.requests,
        "get",
        lambda *args, **kwargs: FakeResponse({
            "success": False,
            "message": "Reserved range",
        }),
    )

    result = geolocation.geolocate_ip("8.8.8.8")

    assert result["status"] == "skipped"
    assert "Reserved range" in result["message"]


def test_geolocate_ip_handles_rate_limit(monkeypatch):
    monkeypatch.setattr(
        geolocation.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(status_code=429),
    )

    result = geolocation.geolocate_ip("8.8.8.8")

    assert result["status"] == "error"
    assert "rate limit" in result["message"].lower()
