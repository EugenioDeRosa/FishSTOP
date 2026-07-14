from src.analyzer import llm_context_analyzer as llm


def test_auto_backend_uses_ollama_when_available(monkeypatch):
    monkeypatch.setattr(llm, "LLM_PROVIDER", "auto")
    monkeypatch.setattr(llm, "_ollama_available", lambda timeout=0.8: True)
    monkeypatch.setattr(llm, "_github_models_token", lambda: "")

    assert llm._use_ollama() is True
    assert llm._llm_enabled() is True
    assert llm.active_llm_backend().startswith("ollama")


def test_auto_backend_falls_back_to_github_when_ollama_missing(monkeypatch):
    monkeypatch.setattr(llm, "LLM_PROVIDER", "auto")
    monkeypatch.setattr(llm, "_ollama_available", lambda timeout=0.8: False)
    monkeypatch.setattr(llm, "_github_models_token", lambda: "token")

    assert llm._use_ollama() is False
    assert llm._llm_enabled() is True
    assert llm.active_llm_backend().startswith("github models")


def test_provider_github_disables_ollama(monkeypatch):
    monkeypatch.setattr(llm, "LLM_PROVIDER", "github")
    monkeypatch.setattr(llm, "_ollama_available", lambda timeout=0.8: True)

    assert llm._use_ollama() is False
