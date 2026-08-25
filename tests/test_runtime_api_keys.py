import inspect

import src.config as config
from src.analyzer import llm_context_analyzer
from src.views import settings


def test_user_secret_has_priority_over_environment(monkeypatch):
    monkeypatch.setenv("VIRUSTOTAL_API_KEY", "env-token")
    monkeypatch.setattr(config, "_read_user_secret", lambda key, default="": "user-token" if key == "VIRUSTOTAL_API_KEY" else default)
    monkeypatch.setattr(config, "_is_streamlit_cloud", lambda: False)

    assert config.get_secret("VIRUSTOTAL_API_KEY") == "user-token"


def test_environment_used_when_user_secret_missing(monkeypatch):
    monkeypatch.setenv("ABUSEIPDB_API_KEY", "env-abuse")
    monkeypatch.setattr(config, "_read_user_secret", lambda key, default="": default)
    monkeypatch.setattr(config, "_is_streamlit_cloud", lambda: False)

    assert config.get_secret("ABUSEIPDB_API_KEY") == "env-abuse"


def test_phi4_token_is_read_at_runtime(monkeypatch):
    monkeypatch.setattr(llm_context_analyzer, "get_secret", lambda key, default="": "runtime-token" if key == "GITHUB_MODELS_TOKEN" else default)

    assert llm_context_analyzer._llm_enabled() is True
    assert llm_context_analyzer._github_models_token() == "runtime-token"


def test_session_key_changes_do_not_clear_the_global_model_cache():
    source = inspect.getsource(settings.render)

    assert "init_content_model.clear" not in source
    assert "cache_resource.clear" not in source
