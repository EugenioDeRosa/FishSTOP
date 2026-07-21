from src.analyzer.llm_context_analyzer import (
    _identity_risk,
    _technical_context_lines,
    _fallback_content_summary,
    _valid_content_summary,
    apply_email_risk_policy,
    format_email_risk_analysis,
)


def test_missing_dkim_is_reported_as_absent_not_failed():
    soc = {
        "effective_auth_results": {
            "SPF": {"status": "softfail"},
            "DKIM": {"status": "none"},
            "DMARC": {"status": "fail"},
        },
        "dkim_signature_present": False,
    }

    risk, reasons = _identity_risk(soc)
    context = _technical_context_lines(soc)

    assert risk == "uncertain"
    assert reasons == [
        "SPF did not pass (softfail)",
        "DKIM signature is absent",
        "DMARC did not pass (fail)",
    ]
    assert any("DKIM signature is absent" in line for line in context)


def test_missing_dkim_is_included_in_readable_explanation():
    text = format_email_risk_analysis({
        "final_verdict": "phishing",
        "content_summary": "The subject and body indicate a phishing attempt",
        "identity_risk": "uncertain",
        "technical_risk": "clean",
        "evidence": {
            "identity": [
                "SPF did not pass (softfail)",
                "DKIM signature is absent",
                "DMARC did not pass (fail)",
            ],
            "technical": ["no strong technical threat was detected"],
        },
    })

    assert "SPF did not pass (softfail)" in text
    assert "the message has no DKIM signature" in text
    assert "DMARC did not pass (fail)" in text


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


def test_phi4_result_is_presented_as_a_plain_sentence():
    assert format_email_risk_analysis({
        "final_verdict": "legitimate",
        "content_summary": "The subject and body provide routine meeting information without requesting risky actions",
        "identity_risk": "verified",
        "technical_risk": "clean",
        "evidence": {
            "identity": ["sender authentication passed"],
            "technical": ["no strong technical threat was detected"],
        },
    }) == (
        "This email is not suspicious. The subject and body provide routine meeting information without requesting risky actions.\n"
        "The technical analysis supports this assessment because the sender is authenticated and no technical threats were detected."
    )
    assert format_email_risk_analysis({
        "final_verdict": "review",
        "content_summary": "The subject and body contain a cryptocurrency offer and direct the recipient to a link, a pattern commonly used in phishing",
        "identity_risk": "uncertain",
        "technical_risk": "clean",
        "evidence": {
            "identity": [
                "SPF did not pass (temperror)",
                "DMARC did not pass (temperror)",
                "Return-Path differs from the visible sender domain",
            ],
            "technical": ["no strong technical threat was detected"],
        },
    }) == (
        "This email is suspicious and requires verification. The subject and body contain a cryptocurrency offer and direct the recipient to a link, a pattern commonly used in phishing.\n"
        "The technical analysis does not prove a threat on its own, but supports caution because SPF did not pass (temperror); "
        "DMARC did not pass (temperror); the Return-Path differs from the visible sender."
    )
    assert format_email_risk_analysis({
        "final_verdict": "phishing",
        "content_summary": "The subject and body ask the recipient to enter credentials on a linked page, a strong phishing pattern",
        "identity_risk": "verified",
        "technical_risk": "malicious",
        "evidence": {
            "identity": ["sender authentication passed"],
            "technical": ["a URL is detected as malicious"],
        },
    }) == (
        "This email is suspicious. The subject and body ask the recipient to enter credentials on a linked page, a strong phishing pattern.\n"
        "The technical analysis supports this assessment because a URL was detected as malicious."
    )


def test_missing_model_summary_has_a_useful_deterministic_fallback():
    summary = _fallback_content_summary(
        {
            "subject": "Lido Community Rewards",
            "links": [{"url": "https://example.test/claim"}],
        },
        {"requested_action": "claim_reward"},
    )

    assert summary == (
        "The subject and body contain a cryptocurrency or reward offer and ask the recipient to claim it through a supplied link, "
        "a pattern commonly used in phishing."
    )


def test_literal_content_recap_is_not_accepted_as_security_analysis():
    assert not _valid_content_summary(
        "Lido Community Rewards: stETH airdrop live, snapshot taken on 21 July."
    )
    assert not _valid_content_summary(
        "The subject and body announce an airdrop and list the snapshot date."
    )
    assert not _valid_content_summary(
        "The subject and body urge account verification via an official portal, which could be used maliciously if intercepted."
    )
    assert _valid_content_summary(
        "The subject and body contain a cryptocurrency reward offer, a pattern commonly used in phishing."
    )


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


def test_bank_account_verification_sample_corrects_phi4_false_negative():
    url = "https://status.nativendo.de/storage/_drs/"
    analysis = apply_email_risk_policy(
        {
            "subject": "Sicherheitscheck durchführen",
            "from_": "easy bank Deutschland <support-team@easybank-app.kunden.com>",
            "body_clean": (
                "Ihr Konto wurde für eine Sicherheitsprüfung ausgewählt. Frist zur Verifizierung: 12. Mai 2026. "
                "Bitte bestätigen Sie Ihr Konto über unser zertifiziertes Prüfportal. Konto verifizieren. "
                "Stellen Sie sicher, dass Sie Ihre easy bank App oder das mobile TAN-Verfahren bereithalten."
            ),
            "auth_results": {
                "SPF": {"status": "none"},
                "DKIM": {"status": "none"},
                "DMARC": {"status": "none"},
            },
            "authentication_results_raw": (
                "spf=none; dkim=none; dmarc=none header.from=easybank-app.kunden.com; compauth=fail reason=001"
            ),
            "links": [{"url": url, "host": "status.nativendo.de", "source": "html_href"}],
            "link_reputation": {},
        },
        # Reproduce the weak and inaccurate extraction returned by Phi-4 mini.
        _semantic(
            requested_action="informational",
            action_channel="unclear",
            asks_to_click_link=False,
            content_summary=(
                "The subject and body contain a security check notification urging verification via an official portal. "
                "The date could be used maliciously if intercepted"
            ),
        ),
    )

    assert analysis["final_verdict"] == "phishing"
    assert analysis["content_risk"] == "suspicious"
    assert analysis["technical_risk"] == "uncertain"
    assert analysis["requested_action"] == "verify_account"
    assert analysis["action_channel"] == "supplied_link"
    assert analysis["content_summary"] == (
        "The subject and body claim to be from a bank and ask the recipient to verify an account "
        "through a supplied link, a common credential-phishing pattern"
    )
    assert "Microsoft composite authentication failed" in analysis["evidence"]["identity"]
    assert "domain unrelated to the sender" in analysis["evidence"]["technical"][0]


def test_authenticated_account_verification_link_is_review_not_automatic_phishing():
    url = "https://security.service.example/verify"
    analysis = apply_email_risk_policy(
        {
            "subject": "Verify your bank account",
            "from_": "Bank <security@bank.example>",
            "body_clean": "Please verify your account using the supplied security link.",
            "auth_results": {
                "SPF": {"status": "pass"},
                "DKIM": {"status": "pass"},
                "DMARC": {"status": "pass"},
            },
            "links": [{"url": url, "host": "security.service.example", "source": "html_href"}],
        },
        _semantic(),
    )

    assert analysis["final_verdict"] == "review"
    assert analysis["content_risk"] == "suspicious"
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


def test_crypto_offer_button_corrects_phi4_informational_false_negative():
    url = "https://marketplace-notification.example/inspect/123"
    analysis = apply_email_risk_policy(
        {
            "subject": "Prompt NFT Offer",
            "body_clean": (
                "A buyer has received on your asset. See info below.\n"
                "Offer Amount:\n1.53 ETH\nOffered By:\nChainMage\n"
                "Inspect Proposal\nYou are receiving this email because you are "
                "subscribed to notifications from OpenSea."
            ),
            "auth_results": {
                "SPF": {"status": "temperror"},
                "DKIM": {"status": "unknown"},
                "DMARC": {"status": "temperror"},
            },
            "return_path_domain_mismatch": True,
            "links": [{"url": url, "host": "marketplace-notification.example", "source": "html_href"}],
            "link_reputation": {url: {"status": "not_found"}},
        },
        # Reproduce the incorrect extraction returned by Phi-4 mini.
        _semantic(
            requested_action="informational",
            action_channel="unclear",
            asks_to_click_link=False,
            asks_to_claim_reward=False,
            financial_incentive_present=False,
        ),
    )

    assert analysis["final_verdict"] == "review"
    assert analysis["content_risk"] == "suspicious"
    assert analysis["requested_action"] == "claim_reward"
    assert analysis["action_channel"] == "supplied_link"


def test_crypto_amount_without_offer_action_remains_benign():
    url = "https://newsletter.example/market-report"
    analysis = apply_email_risk_policy(
        {
            "subject": "Weekly market report",
            "body_clean": "ETH traded at 3,500 USDT this week. Read our general market report.",
            "auth_results": {},
            "links": [{"url": url, "host": "newsletter.example", "source": "html_href"}],
        },
        _semantic(),
    )

    assert analysis["final_verdict"] == "legitimate"
    assert analysis["content_risk"] == "benign"


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
