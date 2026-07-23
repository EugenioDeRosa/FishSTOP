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


def test_phi4_compact_schema_reaches_stable_policy_without_retry(monkeypatch):
    captured = {}

    def fake_stream(messages, model, timeout):
        captured["messages"] = messages
        yield {
            "status": "ok",
            "model": model,
            "text": (
                '{"action":"payment","channel":"link",'
                '"summary":"The email body requests payment through a supplied link, a financial phishing pattern."}'
            ),
        }

    monkeypatch.setattr(llm, "_use_ollama", lambda: False)
    monkeypatch.setattr(llm, "_github_models_token", lambda: "token")
    monkeypatch.setattr(llm, "_stream_github_models", fake_stream)
    monkeypatch.setattr(
        llm,
        "_request_content_summary",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected summary retry")),
    )

    events = list(llm.stream_phi4_email_analysis({
        "body_for_ai": "Please pay using [URL LINK].",
        "links": [{"url": "https://example.test/pay", "host": "example.test"}],
        "auth_results": {
            "SPF": {"status": "pass"},
            "DKIM": {"status": "pass"},
            "DMARC": {"status": "pass"},
        },
    }))

    analysis = events[-1]["analysis"]
    assert analysis["requested_action"] == "pay_or_transfer"
    assert analysis["action_channel"] == "supplied_link"
    assert analysis["content_risk"] == "suspicious"

    user_prompt = captured["messages"][1]["content"]
    assert '"signals"' not in user_prompt
    assert '"check_relation"' not in user_prompt
    assert '"asks_for_payment"' not in user_prompt
    assert '"content_risk"' not in user_prompt


def test_ollama_uses_schema_constrained_output_and_zero_temperature(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            return None

        def iter_lines(self, decode_unicode=True):
            yield '{"message":{"content":"{}"},"done":true}'

    def fake_post(url, json, stream, timeout):
        captured["payload"] = json
        return FakeResponse()

    monkeypatch.setattr(llm.requests, "post", fake_post)
    list(llm._stream_ollama(
        [{"role": "system", "content": "system"}, {"role": "user", "content": "user"}],
        llm.OLLAMA_MODEL,
        10,
    ))

    assert captured["payload"]["format"] == llm.PHI4_OUTPUT_SCHEMA
    assert captured["payload"]["options"]["temperature"] == 0.0


def test_natural_model_summary_is_not_replaced_by_generic_visit_link_fallback(monkeypatch):
    def fake_stream(messages, model, timeout):
        yield {
            "status": "ok",
            "model": model,
            "text": (
                '{"action":"visit_link","channel":"link",'
                '"summary":"This message announces a routine monthly update and offers an optional website link."}'
            ),
        }

    monkeypatch.setattr(llm, "_use_ollama", lambda: False)
    monkeypatch.setattr(llm, "_github_models_token", lambda: "token")
    monkeypatch.setattr(llm, "_stream_github_models", fake_stream)

    events = list(llm.stream_phi4_email_analysis({
        "subject": "Monthly update",
        "body_for_ai": "Here is the monthly update. Read more on our website.",
        "links": [{"url": "https://example.test/news", "host": "example.test"}],
        "auth_results": {
            "SPF": {"status": "pass"},
            "DKIM": {"status": "pass"},
            "DMARC": {"status": "pass"},
        },
    }))

    text = events[-1]["text"]
    assert "This message announces a routine monthly update" in text
    assert "a pattern that requires destination verification" not in text
