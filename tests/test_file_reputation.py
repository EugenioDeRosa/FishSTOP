import socket

import requests

from src.validators import file_reputation as fr


class FakeResponse:
    def __init__(self, url: str, status_code: int = 200, location: str | None = None):
        self.url = url
        self.status_code = status_code
        self.headers = {"Location": location} if location else {}

    def close(self):
        return None


class FakeSession:
    def __init__(self, responses: list[FakeResponse] | None = None, exc: Exception | None = None):
        self.responses = list(responses or [])
        self.exc = exc
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.exc is not None:
            raise self.exc
        return self.responses.pop(0) if self.responses else FakeResponse(url)


class FakeAPIResponse:
    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self.reason = "test response"
        self._payload = payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)


def _allow_public_dns(monkeypatch):
    monkeypatch.setattr(fr, "_public_http_url", lambda url: (True, ""))


def test_check_url_reports_redirect_match(monkeypatch):
    _allow_public_dns(monkeypatch)
    monkeypatch.setattr(fr, "_session", FakeSession([
        FakeResponse("https://example.com/start", 302, "/welcome"),
        FakeResponse("https://example.com/welcome"),
    ]))

    result = fr._check_destination("https://example.com/start")

    assert result["final_url"] == "https://example.com/welcome"
    assert result["final_host"] == "example.com"
    assert result["destination_status"] == "match"
    assert result["destination_match"] is True
    assert result["redirect_count"] == 1


def test_check_url_reports_redirect_mismatch(monkeypatch):
    _allow_public_dns(monkeypatch)
    monkeypatch.setattr(fr, "_session", FakeSession([
        FakeResponse("https://example.com/start", 302, "https://redirect.example.org/next"),
        FakeResponse("https://redirect.example.org/next", 302, "https://evil.example.net/login"),
        FakeResponse("https://evil.example.net/login"),
    ]))

    result = fr._check_destination("https://example.com/start")

    assert result["final_url"] == "https://evil.example.net/login"
    assert result["final_host"] == "evil.example.net"
    assert result["destination_status"] == "mismatch"
    assert result["destination_match"] is False
    assert result["redirect_count"] == 2


def test_check_url_reports_timeout_as_unavailable(monkeypatch):
    _allow_public_dns(monkeypatch)
    monkeypatch.setattr(fr, "_session", FakeSession(exc=requests.exceptions.Timeout()))

    result = fr._check_destination("https://example.com/start")

    assert result["destination_status"] == "unavailable"
    assert result["destination_match"] is False
    assert "Timeout" in result["destination_message"]


def test_check_url_does_not_contact_destination_without_api_key(monkeypatch):
    session = FakeSession()
    monkeypatch.setattr(fr, "_session", session)

    result = fr.check_url("", "https://example.com/start")

    assert result["status"] == "skipped"
    assert session.calls == []


def test_check_url_blocks_private_destination(monkeypatch):
    monkeypatch.setattr(fr, "URL_DESTINATION_CHECK_ENABLED", True)
    monkeypatch.setattr(
        fr.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))],
    )
    session = FakeSession()
    monkeypatch.setattr(fr, "_session", session)

    result = fr.check_url("token", "http://localhost/internal")

    assert result["status"] == "blocked"
    assert result["destination_status"] == "blocked"
    assert session.calls == []


def test_direct_destination_check_is_disabled_by_default(monkeypatch):
    monkeypatch.setattr(fr, "URL_DESTINATION_CHECK_ENABLED", False)
    session = FakeSession()
    monkeypatch.setattr(fr, "_session", session)

    result = fr.check_url("token", "https://example.com/start")

    assert result["destination_status"] == "skipped"
    assert len(session.calls) == 1
    assert session.calls[0][0].startswith(fr.VIRUSTOTAL_URL_ENDPOINT)


def test_url_report_permalink_uses_canonical_object_id():
    result = fr._format_vt_url(
        {
            "data": {
                "id": "a" * 64,
                "attributes": {
                    "last_analysis_stats": {"harmless": 70},
                },
            }
        },
        {"permalink": "https://www.virustotal.com/gui/home/url"},
    )

    assert result["permalink"] == f"https://www.virustotal.com/gui/url/{'a' * 64}"


def test_unknown_url_is_not_submitted_automatically(monkeypatch):
    session = FakeSession([FakeAPIResponse(404)])
    monkeypatch.setattr(fr, "_session", session)

    result = fr.check_url("token", "https://example.test/unknown")

    assert result["status"] == "not_found"
    assert result["permalink"] == "https://www.virustotal.com/gui/home/url"
    assert len(session.calls) == 1
