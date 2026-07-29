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


def test_auto_backend_reuses_short_lived_ollama_health_check(monkeypatch):
    calls = 0

    def available(timeout=0.8):
        nonlocal calls
        calls += 1
        return False

    monkeypatch.setattr(llm, "LLM_PROVIDER", "auto")
    monkeypatch.setattr(llm, "OLLAMA_AVAILABILITY_TTL", 5.0)
    monkeypatch.setattr(llm, "_OLLAMA_AVAILABILITY_CACHE", None)
    monkeypatch.setattr(llm, "_ollama_available", available)
    monkeypatch.setattr(llm, "_github_models_token", lambda: "")

    assert llm._use_ollama() is False
    assert llm._use_ollama() is False
    assert llm.active_llm_backend() == "not configured"
    assert calls == 1


def test_phi4_compact_schema_reaches_stable_policy_without_retry(monkeypatch):
    captured = {}

    def fake_stream(messages, model, timeout):
        captured["messages"] = messages
        yield {
            "status": "ok",
            "model": model,
            "text": (
                '{"action":"payment","channel":"link",'
                '"evidence":"Please pay using [URL LINK]."}'
            ),
        }

    monkeypatch.setattr(llm, "_use_ollama", lambda: False)
    monkeypatch.setattr(llm, "_github_models_token", lambda: "token")
    monkeypatch.setattr(llm, "_stream_github_models", fake_stream)
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
    assert '"signals"' in user_prompt
    assert '"credential_type"' in user_prompt
    assert '"claimed_brand"' in user_prompt
    assert '"check_relation"' not in user_prompt
    assert '"asks_for_payment"' not in user_prompt
    assert '"content_risk"' not in user_prompt
    assert '"evidence"' in user_prompt
    assert '"summary"' not in user_prompt


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


def test_ollama_stream_events_expose_only_the_incremental_delta(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            return None

        def iter_lines(self, decode_unicode=True):
            yield '{"message":{"content":"first"},"done":false}'
            yield '{"message":{"content":"second"},"done":true}'

    monkeypatch.setattr(
        llm.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(),
    )

    events = list(llm._stream_ollama([], llm.OLLAMA_MODEL, 10))
    stream_events = [
        event for event in events if event["status"] == "stream"
    ]

    assert [event["delta"] for event in stream_events] == [
        "first",
        "second",
    ]
    assert all("text" not in event for event in stream_events)
    assert events[-1]["text"] == "firstsecond"


def test_github_stream_events_expose_only_the_incremental_delta(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            return None

        def iter_lines(self, decode_unicode=True):
            yield 'data: {"choices":[{"delta":{"content":"first"}}]}'
            yield 'data: {"choices":[{"delta":{"content":"second"}}]}'
            yield "data: [DONE]"

    monkeypatch.setattr(
        llm.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(),
    )

    events = list(llm._stream_github_models([], "model", 10, token="token"))
    stream_events = [
        event for event in events if event["status"] == "stream"
    ]

    assert [event["delta"] for event in stream_events] == [
        "first",
        "second",
    ]
    assert all("text" not in event for event in stream_events)
    assert events[-1]["text"] == "firstsecond"


def test_summary_is_derived_from_verified_action(monkeypatch):
    def fake_stream(messages, model, timeout):
        yield {
            "status": "ok",
            "model": model,
            "text": (
                '{"action":"visit_link","channel":"link",'
                '"evidence":"Read more on our website."}'
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
    assert "ask the recipient to follow a supplied link" in text
    assert "requires destination verification" not in text


def test_ambiguous_credential_link_uses_targeted_verifier(monkeypatch):
    calls = []

    def fake_stream(messages, model, timeout):
        calls.append(messages)
        if len(calls) == 1:
            text = (
                '{"action":"visit_link","channel":"link",'
                '"evidence":"open the form"}'
            )
        else:
            text = (
                '{"action":"provide_credentials","channel":"form",'
                '"evidence":"enter your email address and password"}'
            )
        yield {"status": "ok", "model": model, "text": text}

    monkeypatch.setattr(llm, "_use_ollama", lambda: False)
    monkeypatch.setattr(llm, "_github_models_token", lambda: "token")
    monkeypatch.setattr(llm, "_stream_github_models", fake_stream)

    events = list(llm.stream_phi4_email_analysis({
        "subject": "Mailbox verification",
        "body_for_ai": (
            "To keep your mailbox active, open the form and enter your email "
            "address and password now."
        ),
        "links": [{"url": "https://example.test/form", "host": "example.test"}],
        "auth_results": {},
    }))

    analysis = events[-1]["analysis"]
    assert len(calls) == 2
    assert analysis["requested_action"] == "provide_credentials"
    assert analysis["action_channel"] == "external_form"
    assert analysis["semantic_extraction"]["intent_verifier_used"] is True
    assert analysis["semantic_extraction"]["primary_requested_action"] == "visit_link"
    assert analysis["semantic_extraction"]["evidence_phrase"] == (
        "enter your email address and password"
    )


def test_long_email_prompt_preserves_late_intent():
    body = (
        ("This is background information about the service and its features. " * 90)
        + "To complete the request, enter your password and recovery code in the form."
    )

    prompt = llm.build_fast_email_prompt({
        "subject": "Service information",
        "body_for_ai": body,
        "links": [{"url": "https://example.test/form", "host": "example.test"}],
    })

    assert prompt.count(
        "This is background information about the service and its features."
    ) == 90
    assert "[EMAIL BEGINNING]" not in prompt
    assert "[ACTION-BEARING SENTENCES]" not in prompt
    assert "[EMAIL ENDING]" not in prompt
    assert "enter your password and recovery code" in prompt


def test_complete_body_split_keeps_every_middle_section():
    paragraphs = [
        f"Paragraph {index}: unique context value {index}."
        for index in range(1, 31)
    ]
    body = "\n\n".join(paragraphs)

    chunks = llm._split_complete_email_body(
        body,
        limit=180,
        overlap=40,
    )

    assert len(chunks) > 1
    for paragraph in paragraphs:
        assert any(paragraph in chunk for chunk in chunks)


def test_long_email_analyzes_all_sections_and_merges_middle_intent(monkeypatch):
    calls = []

    def fake_stream(messages, model, timeout):
        prompt = messages[1]["content"]
        calls.append(prompt)
        if "enter your password and recovery code" in prompt:
            text = (
                '{"action":"provide_credentials","channel":"form",'
                '"evidence":"enter your password and recovery code"}'
            )
        else:
            text = '{"action":"info","channel":"none","evidence":""}'
        yield {"status": "ok", "model": model, "text": text}

    background = (
        "This paragraph contains ordinary background information about the service. "
        * 70
    )
    body = (
        background
        + "\n\nTo continue, enter your password and recovery code in the form.\n\n"
        + background
        + background
    )
    monkeypatch.setattr(llm, "_use_ollama", lambda: False)
    monkeypatch.setattr(llm, "_github_models_token", lambda: "token")
    monkeypatch.setattr(llm, "_stream_github_models", fake_stream)

    events = list(llm.stream_phi4_email_analysis({
        "subject": "Service information",
        "body_for_ai": body,
        "links": [{"url": "https://example.test/form", "host": "example.test"}],
        "auth_results": {},
    }))

    progress = [
        event for event in events
        if event.get("status") == "progress"
        and event.get("stage") == "content"
    ]
    analysis = events[-1]["analysis"]
    assert len(calls) > 1
    assert len(progress) == len(calls)
    assert progress[-1]["current"] == progress[-1]["total"]
    assert analysis["requested_action"] == "provide_credentials"
    assert analysis["intent_evidence"] == "enter your password and recovery code"
    assert events[-1]["analyzed_sections"] == len(calls)


def test_impossible_link_channel_is_removed():
    semantic = llm.normalize_semantic_extraction(
        {
            "action": "verify_account",
            "channel": "link",
            "evidence": "confirm whether this login was yours",
        },
        {
            "subject": "Login alert",
            "body_for_ai": "Confirm whether this login was yours.",
            "links": [],
        },
    )

    corrected = llm._correlate_semantic_with_message_structure(
        {
            "subject": "Login alert",
            "body_for_ai": "Confirm whether this login was yours.",
            "links": [],
        },
        semantic,
    )

    assert corrected["requested_action"] == "verify_account"
    assert corrected["action_channel"] == "unclear"
    assert corrected["asks_to_click_link"] is False


def test_reply_channel_requires_an_explicit_email_reply():
    semantic = llm.normalize_semantic_extraction(
        {
            "action": "verify_account",
            "channel": "reply",
            "evidence": "confirm whether this login was yours",
        },
        {
            "subject": "Login alert",
            "body_for_ai": "Confirm whether this login was yours.",
        },
    )

    corrected = llm._correlate_semantic_with_message_structure(
        {
            "subject": "Login alert",
            "body_for_ai": "Confirm whether this login was yours.",
        },
        semantic,
    )

    assert corrected["requested_action"] == "verify_account"
    assert corrected["action_channel"] == "unclear"


def test_model_evidence_must_exist_in_email():
    semantic = llm.normalize_semantic_extraction(
        {
            "action": "payment",
            "channel": "reply",
            "evidence": "wire the money immediately",
        },
        {
            "subject": "Quarterly planning",
            "body_for_ai": "Please reply with your availability for the meeting.",
        },
    )

    assert semantic["evidence_phrase"] == ""


def test_targeted_verifier_refines_payment_without_losing_specific_scam_type():
    merged = llm._merge_targeted_intent(
        {
            "payment_asset": "",
            "amount": "950€",
            "scam_type": "sextortion",
            "threat_type": "private_material_exposure",
        },
        {
            "action": "pay_or_transfer",
            "payment_method": "cryptocurrency",
            "payment_asset": "Bitcoin",
            "amount": "",
            "scam_type": "extortion",
            "threat_type": "data_exposure",
        },
    )

    assert merged["payment_method"] == "cryptocurrency"
    assert merged["payment_asset"] == "Bitcoin"
    assert merged["amount"] == "950€"
    assert merged["scam_type"] == "sextortion"
    assert merged["threat_type"] == "private_material_exposure"


def test_unicode_variation_selectors_are_removed_before_phi4():
    obfuscated = (
        "Please enter your old private "
        "wa\U000e0139\U000e0139l\U000e0139\U000e0139let "
        "p\U000e0139\U000e0139hrase."
    )

    prompt = llm.build_fast_email_prompt({
        "subject": "Tr\U000e0139ust Wallet",
        "body_for_ai": obfuscated,
        "links": [],
    })

    assert "\U000e0139" not in prompt
    assert "wallet phrase" in prompt
    assert "Trust Wallet" in prompt


def test_long_obfuscated_wallet_evidence_remains_valid():
    body = (
        "Please click the Re-validate link and enter your old 12 or 24-word "
        "private wa\U000e0139\U000e0139llet phrase on the next page."
    )
    semantic = llm.normalize_semantic_extraction(
        {
            "action": "provide_credentials",
            "channel": "link",
            "evidence": (
                "Please click the Re-validate link and enter your old 12 or "
                "24-word private wallet phrase on the next page."
            ),
            "credential_type": "wallet_seed",
        },
        {
            "subject": "Wallet security update",
            "body_for_ai": body,
            "links": [{"url": "https://example.test", "host": "example.test"}],
        },
    )

    assert "private wallet phrase" in semantic["evidence_phrase"]
    assert semantic["credential_type"] == "wallet_seed"
