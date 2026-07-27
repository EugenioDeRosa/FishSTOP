from contextlib import nullcontext

from src import error_handling


def test_production_error_hides_exception_and_returns_reference(monkeypatch):
    rendered = []
    exception_calls = []
    monkeypatch.setattr(error_handling, "is_production_mode", lambda: True)
    monkeypatch.setattr(
        error_handling.st,
        "error",
        lambda value: rendered.append(str(value)),
    )
    monkeypatch.setattr(
        error_handling.st,
        "caption",
        lambda value: rendered.append(str(value)),
    )
    monkeypatch.setattr(
        error_handling.st,
        "exception",
        lambda value: exception_calls.append(value),
    )

    reference = error_handling.render_unexpected_error(
        "Public failure message.",
        RuntimeError("secret-token-and-server-path"),
        context="test production handler",
    )

    assert reference.startswith("FS-")
    assert "Public failure message." in rendered
    assert any(reference in value for value in rendered)
    assert all("secret-token-and-server-path" not in value for value in rendered)
    assert exception_calls == []


def test_development_error_keeps_diagnostics_available(monkeypatch):
    exception_calls = []
    monkeypatch.setattr(error_handling, "is_production_mode", lambda: False)
    monkeypatch.setattr(error_handling.st, "error", lambda _value: None)
    monkeypatch.setattr(error_handling.st, "caption", lambda _value: None)
    monkeypatch.setattr(
        error_handling.st,
        "expander",
        lambda *_args, **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        error_handling.st,
        "exception",
        lambda value: exception_calls.append(value),
    )
    error = RuntimeError("development details")

    error_handling.render_unexpected_error(
        "Failure.",
        error,
        context="test development handler",
    )

    assert exception_calls == [error]
