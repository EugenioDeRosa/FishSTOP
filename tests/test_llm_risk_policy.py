from src.analyzer.llm_context_analyzer import apply_email_risk_policy


def _semantic(**overrides):
    result = {
        "requested_action": "informational",
        "action_channel": "none",
        "asks_to_click_link": False,
        "asks_to_open_attachment": False,
        "asks_for_credentials": False,
        "asks_for_sensitive_information": False,
        "asks_for_payment": False,
        "asks_to_change_account_settings": False,
        "asks_to_bypass_procedure": False,
        "urgency_present": False,
        "urgency_targets_risky_action": False,
        "impersonation_or_deception": False,
        "content_risk": "benign",
        "confidence": 0.8,
        "reason": "Test extraction",
    }
    result.update(overrides)
    return result


def test_authentication_failures_alone_do_not_create_phishing_verdict():
    soc = {
        "auth_results": {
            "SPF": {"status": "fail"},
            "DKIM": {"status": "fail"},
            "DMARC": {"status": "fail"},
        }
    }

    analysis = apply_email_risk_policy(soc, _semantic())

    assert analysis["final_verdict"] == "legitimate"
    assert analysis["content_risk"] == "benign"
    assert analysis["identity_risk"] == "uncertain"


def test_urgent_password_change_via_normal_procedure_is_benign():
    analysis = apply_email_risk_policy(
        {"auth_results": {}},
        _semantic(
            requested_action="change_account_settings",
            action_channel="normal_known_procedure",
            # Phi-4 can overreact to the word "password". The normal known
            # channel must prevent this isolated inconsistent flag from deciding.
            asks_for_credentials=True,
            asks_to_change_account_settings=True,
            urgency_present=True,
            urgency_targets_risky_action=False,
        ),
    )

    assert analysis["final_verdict"] == "legitimate"
    assert analysis["content_risk"] == "benign"


def test_password_expiration_reminder_without_channel_or_link_is_benign():
    analysis = apply_email_risk_policy(
        {
            "auth_results": {
                "SPF": {"status": "fail"},
                "DKIM": {"status": "fail"},
            },
            "links": [],
            "attachments": [],
        },
        _semantic(
            requested_action="change_account_settings",
            action_channel="unclear",
            asks_to_change_account_settings=True,
            urgency_present=True,
            urgency_targets_risky_action=True,
        ),
    )

    assert analysis["final_verdict"] == "legitimate"
    assert analysis["content_risk"] == "benign"
    assert analysis["identity_risk"] == "uncertain"


def test_password_change_through_supplied_urgent_link_requires_review():
    analysis = apply_email_risk_policy(
        {"auth_results": {}},
        _semantic(
            requested_action="change_account_settings",
            action_channel="supplied_link",
            asks_to_click_link=True,
            asks_to_change_account_settings=True,
            urgency_present=True,
            urgency_targets_risky_action=True,
        ),
    )

    assert analysis["final_verdict"] == "review"
    assert analysis["content_risk"] == "suspicious"


def test_extracted_link_overrides_model_calling_account_change_normal():
    url = "https://unknown.example/changepassword"
    analysis = apply_email_risk_policy(
        {
            "auth_results": {
                "SPF": {"status": "pass"},
                "DKIM": {"status": "pass"},
                "DMARC": {"status": "pass"},
            },
            "links": [{"url": url, "host": "unknown.example", "source": "plain_text"}],
            "link_reputation": {url: {"status": "not_found"}},
        },
        _semantic(
            requested_action="informational",
            action_channel="normal_known_procedure",
            asks_to_change_account_settings=True,
            asks_to_click_link=False,
            urgency_present=True,
        ),
    )

    assert analysis["final_verdict"] == "review"
    assert analysis["content_risk"] == "suspicious"
    assert analysis["action_channel"] == "supplied_link"
    assert analysis["technical_risk"] == "clean"


def test_reward_claim_through_supplied_link_requires_review():
    url = "https://storage.example/claim"
    analysis = apply_email_risk_policy(
        {
            "auth_results": {
                "SPF": {"status": "pass"},
                "DKIM": {"status": "pass"},
                "DMARC": {"status": "pass"},
            },
            "links": [{"url": url, "host": "storage.example", "source": "html_href"}],
            "link_reputation": {url: {"status": "not_found"}},
        },
        _semantic(
            requested_action="claim_reward",
            action_channel="unclear",
            asks_to_click_link=True,
            asks_to_claim_reward=True,
            financial_incentive_present=True,
        ),
    )

    assert analysis["final_verdict"] == "review"
    assert analysis["content_risk"] == "suspicious"
    assert analysis["action_channel"] == "supplied_link"
    assert "reward or financial benefit" in analysis["evidence"]["content"][0]


def test_credential_request_or_malicious_url_produces_phishing():
    credential_analysis = apply_email_risk_policy(
        {"auth_results": {}},
        _semantic(
            requested_action="provide_credentials",
            action_channel="email_reply",
            asks_for_credentials=True,
        ),
    )
    url_analysis = apply_email_risk_policy(
        {
            "auth_results": {},
            "link_reputation": {"https://example.test": {"status": "malicious"}},
        },
        _semantic(),
    )

    assert credential_analysis["final_verdict"] == "phishing"
    assert url_analysis["final_verdict"] == "phishing"
