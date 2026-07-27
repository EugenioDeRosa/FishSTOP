from src import app
from src import config
from src.views import dataset_sources


def test_production_mode_recognizes_public_deployment_values(monkeypatch):
    for value in ("prod", "production", "public", " Production "):
        monkeypatch.setenv("APP_MODE", value)
        assert config.is_production_mode() is True

    monkeypatch.setenv("APP_MODE", "development")
    assert config.is_production_mode() is False


def test_dataset_page_is_not_available_in_production(monkeypatch):
    monkeypatch.setattr(app, "is_production_mode", lambda: True)

    assert "dataset_sources" not in app.available_pages()
    assert app._allowed_page("dataset_sources") == "analyze"


def test_dataset_page_remains_available_locally(monkeypatch):
    monkeypatch.setattr(app, "is_production_mode", lambda: False)

    assert app.available_pages()["dataset_sources"] == "src.views.dataset_sources"
    assert app._allowed_page("dataset_sources") == "dataset_sources"


def test_forced_dataset_page_render_is_blocked_in_production(monkeypatch):
    messages = []
    page_intro_calls = []
    monkeypatch.setattr(dataset_sources, "is_production_mode", lambda: True)
    monkeypatch.setattr(dataset_sources.st, "error", messages.append)
    monkeypatch.setattr(
        dataset_sources,
        "page_intro",
        lambda *args, **kwargs: page_intro_calls.append((args, kwargs)),
    )

    dataset_sources.render()

    assert messages == [
        "Training dataset tools are disabled on the public website."
    ]
    assert page_intro_calls == []
