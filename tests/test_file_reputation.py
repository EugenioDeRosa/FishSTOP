from types import SimpleNamespace

import requests

from src.validators import file_reputation as fr


class FakeResponse:
    def __init__(self, url: str, history_len: int = 0):
        self.url = url
        self.history = [SimpleNamespace(status_code=302)] * history_len

    def close(self):
        return None


class FakeSession:
    def __init__(self, final_url: str | None = None, exc: Exception | None = None, history_len: int = 0):
        self.final_url = final_url
        self.exc = exc
        self.history_len = history_len
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.exc is not None:
            raise self.exc
        return FakeResponse(self.final_url or url, self.history_len)


def test_check_url_reports_redirect_match(monkeypatch):
    monkeypatch.setattr(fr, "_session", FakeSession(final_url="https://example.com/welcome", history_len=1))

    result = fr.check_url("", "https://example.com/start")

    assert result["status"] == "skipped"
    assert result["final_url"] == "https://example.com/welcome"
    assert result["final_host"] == "example.com"
    assert result["destination_status"] == "match"
    assert result["destination_match"] is True
    assert result["redirect_count"] == 1


def test_check_url_reports_redirect_mismatch(monkeypatch):
    monkeypatch.setattr(fr, "_session", FakeSession(final_url="https://evil.example.net/login", history_len=2))

    result = fr.check_url("", "https://example.com/start")

    assert result["status"] == "skipped"
    assert result["final_url"] == "https://evil.example.net/login"
    assert result["final_host"] == "evil.example.net"
    assert result["destination_status"] == "mismatch"
    assert result["destination_match"] is False
    assert result["redirect_count"] == 2


def test_check_url_reports_timeout_as_unavailable(monkeypatch):
    monkeypatch.setattr(fr, "_session", FakeSession(exc=requests.exceptions.Timeout()))

    result = fr.check_url("", "https://example.com/start")

    assert result["status"] == "skipped"
    assert result["destination_status"] == "unavailable"
    assert result["destination_match"] is False
    assert "Timeout" in result["destination_message"]