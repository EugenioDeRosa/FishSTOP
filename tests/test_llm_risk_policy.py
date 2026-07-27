from src.analyzer.llm_context_analyzer import (
    _identity_risk,
    _technical_context_lines,
    _fallback_content_summary,
    _valid_content_summary,
    apply_email_risk_policy,
    format_email_risk_analysis,
    normalize_semantic_extraction,
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


def test_dkim_absence_is_retained_when_dmarc_passes():
    risk, reasons = _identity_risk({
        "effective_auth_results": {
            "SPF": {"status": "pass"},
            "DKIM": {"status": "none"},
            "DMARC": {"status": "pass"},
        },
    })

    assert risk == "verified"
    assert reasons == ["sender authentication passed", "DKIM signature is absent"]


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


def test_phi4_result_is_presented_as_a_fluent_soc_summary():
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
        "Our analysis indicates that this email is likely legitimate. "
        "The subject and body provide routine meeting information without requesting risky actions. "
        "Independent technical checks support this assessment because the sender is authenticated "
        "and no confirmed technical threat was detected."
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
            "Our analysis indicates that this email requires manual verification before the recipient takes action. "
            "The subject and body contain a cryptocurrency offer and direct the recipient to a link, a pattern commonly used in phishing. "
            "Independent technical checks support this assessment because SPF did not pass (temperror), "
            "DMARC did not pass (temperror), and the Return-Path differs from the visible sender."
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
            "Our analysis indicates that this email is likely a phishing attempt. "
            "The subject and body ask the recipient to enter credentials on a linked page, a strong phishing pattern. "
            "Independent technical checks support this assessment because a URL was detected as malicious."
        )

    rendered = format_email_risk_analysis({
        "final_verdict": "review",
        "content_summary": "The email body asks the recipient to inspect a supplied link",
        "identity_risk": "uncertain",
        "technical_risk": "uncertain",
        "evidence": {
            "identity": ["DKIM signature is absent"],
            "technical": ["a URL has suspicious reputation"],
        },
    })
    assert "\n" not in rendered
    assert not rendered.lstrip().startswith("{")


def test_missing_model_summary_has_a_useful_deterministic_fallback():
    summary = _fallback_content_summary(
        {
            "subject": "Lido Community Rewards",
            "links": [{"url": "https://example.test/claim"}],
        },
        {"requested_action": "claim_reward"},
    )

    assert summary == (
        "The subject and body contain a reward or promotional benefit and ask the recipient to claim it through a supplied link, "
        "a pattern commonly used in phishing."
    )

    compact_summary = _fallback_content_summary(
        {"links": [{"url": "https://example.test/claim"}]},
        {"action": "claim_reward"},
    )
    assert compact_summary == summary


def test_compact_phi4_schema_expands_to_policy_fields():
    semantic = normalize_semantic_extraction({
        "action": "payment",
        "channel": "link",
        "signals": ["click", "payment", "urgency", "risky_urgency", "deception"],
        "confidence": 0.91,
        "reason": "Requests an urgent transfer through a supplied link.",
        "summary": "The email body requests an urgent linked payment, a financial phishing pattern.",
    })

    assert semantic["requested_action"] == "pay_or_transfer"
    assert semantic["action_channel"] == "supplied_link"
    assert semantic["asks_to_click_link"] is True
    assert semantic["asks_for_payment"] is True
    assert semantic["urgency_present"] is True
    assert semantic["urgency_targets_risky_action"] is True
    assert semantic["impersonation_or_deception"] is True
    assert semantic["content_summary"].startswith("The email body")


def test_compact_phi4_schema_derives_action_signals_without_repeated_booleans():
    semantic = normalize_semantic_extraction({
        "action": "provide_credentials",
        "channel": "form",
        "signals": [],
        "confidence": 0.8,
        "reason": "Credentials are requested in a form.",
        "summary": "The email body requests credentials through a supplied form, a strong phishing pattern.",
    })

    assert semantic["asks_for_credentials"] is True
    assert semantic["action_channel"] == "external_form"


def test_compact_provide_information_action_is_sensitive_without_old_signals_field():
    semantic = normalize_semantic_extraction({
        "action": "provide_information",
        "channel": "form",
        "summary": "Submit personal details through the supplied form.",
    })

    assert semantic["requested_action"] == "provide_information"
    assert semantic["asks_for_sensitive_information"] is True


def test_explicit_reward_claim_corrects_generic_link_or_payment_label():
    soc = {
        "subject": "Your free spins expire tonight — claim now",
        "body_for_ai": "Claim your free spins using the link.",
        "links": [{"url": "https://example.test/claim", "host": "example.test"}],
        "auth_results": {},
    }

    analysis = apply_email_risk_policy(soc, {
        "action": "visit_link",
        "channel": "link",
        "summary": "Claim free spins using the supplied link.",
    })

    assert analysis["requested_action"] == "claim_reward"
    assert analysis["content_risk"] == "suspicious"

    payment_misread = apply_email_risk_policy(soc, {
        "action": "payment",
        "channel": "link",
        "summary": "Request for payment of a bonus.",
    })
    assert payment_misread["requested_action"] == "claim_reward"
    assert "ask the recipient to claim it" in payment_misread["content_summary"]


def test_explicit_password_creation_corrects_generic_link_label():
    analysis = apply_email_risk_policy({
        "subject": "Cree una contraseña para SIX DEGREES IT",
        "body_for_ai": "Cree una contraseña mediante este enlace.",
        "links": [{"url": "https://example.test/password", "host": "example.test"}],
        "auth_results": {},
    }, {
        "action": "visit_link",
        "channel": "link",
        "summary": "Create a password for SIX DEGREES IT.",
    })

    assert analysis["requested_action"] == "change_account_settings"
    assert analysis["content_risk"] == "suspicious"


def test_reward_survey_requesting_personal_data_uses_more_specific_information_action():
    analysis = apply_email_risk_policy({
        "subject": "WOW TV Standard-Abo",
        "body_for_ai": (
            "Sie können sich Ihren Gewinn sichern. Nehmen Sie an unserer Umfrage teil. "
            "Tragen Sie auf den nachfolgenden Seiten Ihre Datein ein."
        ),
        "links": [{"url": "https://example.test/survey", "host": "example.test"}],
        "auth_results": {},
    }, {
        "action": "reply",
        "channel": "link",
        "summary": "Participate in a survey for a free year of WOW TV.",
    })

    assert analysis["requested_action"] == "provide_information"
    assert analysis["content_risk"] == "suspicious"
    assert "submit personal information" in analysis["content_summary"]


def test_bypass_label_is_downgraded_without_evasion_language():
    analysis = apply_email_risk_policy({
        "subject": "Save 80% on energy costs",
        "body_for_ai": "Discover this promotional offer on our website.",
        "links": [{"url": "https://example.test/offer", "host": "example.test"}],
        "auth_results": {},
    }, {
        "action": "bypass",
        "channel": "link",
        "summary": "Promotional energy-saving offer.",
    })

    assert analysis["requested_action"] == "visit_link"
    assert analysis["content_risk"] == "benign"
    assert "bypass normal procedures" not in analysis["content_summary"]


def test_natural_content_summaries_are_accepted_without_a_fixed_prefix():
    assert _valid_content_summary(
        "Lido Community Rewards: stETH airdrop live, snapshot taken on 21 July."
    )
    assert _valid_content_summary(
        "The subject and body announce an airdrop and list the snapshot date."
    )
    assert not _valid_content_summary(
        "The subject and body urge account verification via an official portal, which could be used maliciously if intercepted."
    )
    assert _valid_content_summary(
        "The subject and body contain a cryptocurrency reward offer, a pattern commonly used in phishing."
    )
    assert _valid_content_summary(
        "This message announces a routine monthly update and offers an optional website link."
    )
    assert _valid_content_summary(
        "Aave airdrop claim with a supplied link."
    )
    assert not _valid_content_summary(
        "Untrusted email requests account verification."
    )
    assert not _valid_content_summary(
        "The email is a phishing attempt."
    )
    assert not _valid_content_summary(
        "The email requests account verification, and BERT supports this interpretation."
    )


def test_internal_tool_recommendation_is_legitimate_and_reports_missing_dkim():
    soc = {
        "subject": "Accesso remoto",
        "body_for_ai": (
            "Ciao Gianluca, ok grazie per il feedback. Nessuna problematica all'uso di TeamViewer, "
            "vi chiederei di usare in primis Neurons anche per testarlo, soprattutto per connessioni dentro Cefla."
        ),
        "links": [
            {"url": "https://example.test/1", "host": "example.test"},
            {"url": "https://example.test/2", "host": "example.test"},
        ],
        "attachments": [{}, {}],
        "effective_auth_results": {
            "SPF": {"status": "pass"},
            "DKIM": {"status": "none"},
            "DMARC": {"status": "pass"},
        },
    }
    analysis = apply_email_risk_policy(soc, {
        "action": "info",
        "channel": "reply",
        "summary": "The subject and body recommend an internal remote-access tool.",
    })
    rendered = format_email_risk_analysis(analysis)

    assert analysis["final_verdict"] == "legitimate"
    assert analysis["content_risk"] == "benign"
    assert analysis["requested_action"] == "informational"
    assert "recommend an internal remote-access tool" in rendered
    assert "social engineering" not in rendered
    assert (
        "Independent technical checks support this assessment because the sender is authenticated "
        "and no confirmed technical threat was detected. However, the message has no DKIM signature."
    ) in rendered


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
        {
            "auth_results": {},
            "links": [{
                "url": "https://unknown.example/change-password",
                "host": "unknown.example",
                "source": "plain_text",
            }],
        },
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


def test_unusual_signin_report_link_corrects_phi4_informational_false_negative():
    url = "https://account-alert.example/report"
    body = (
        "Microsoft account\nUnusual sign.in activity\n"
        "We detected something unusual about a recent sign-in to the Microsoft account.\n"
        "A user from Russia/Moscow just logged into your account from a new device. "
        "If this wasn't you, please report the user. If this was you, we'll trust similar activity.\n"
        "Report The User"
    )
    analysis = apply_email_risk_policy(
        {
            "subject": "Microsoft account - Unusual sign.in activity",
            "body_clean": body,
            "auth_results": {
                "SPF": {"status": "pass"},
                "DKIM": {"status": "none"},
                "DMARC": {"status": "pass"},
            },
            "reply_to_mismatch": True,
            "bert_ai_result": "phishing",
            "links": [{"url": url, "host": "account-alert.example", "source": "html_href"}],
            "link_reputation": {url: {"status": "clean"}},
        },
        # Reproduce Phi-4 mini noticing the alert but missing its requested action.
        _semantic(
            requested_action="informational",
            action_channel="unclear",
            asks_to_click_link=False,
            content_summary=(
                "The subject and body provide information without a clearly identified risky request."
            ),
        ),
    )

    assert analysis["final_verdict"] == "phishing"
    assert analysis["content_risk"] == "suspicious"
    assert analysis["requested_action"] == "verify_account"
    assert analysis["action_channel"] == "supplied_link"
    assert analysis["content_summary"] == (
        "The subject and body claim suspicious account activity and direct the recipient to respond "
        "through a supplied link, a common account-security phishing lure"
    )


def test_unusual_signin_notice_without_supplied_channel_remains_informational():
    analysis = apply_email_risk_policy(
        {
            "subject": "Unusual sign-in activity",
            "body_clean": (
                "We detected a new sign-in to your account. If this wasn't you, "
                "open the service manually and review recent activity."
            ),
            "auth_results": {
                "SPF": {"status": "pass"},
                "DKIM": {"status": "pass"},
                "DMARC": {"status": "pass"},
            },
            "links": [],
        },
        _semantic(
            requested_action="informational",
            action_channel="normal_known_procedure",
            asks_to_click_link=False,
        ),
    )

    assert analysis["final_verdict"] == "legitimate"
    assert analysis["content_risk"] == "benign"
    assert analysis["requested_action"] == "informational"


def test_authenticated_unusual_signin_report_link_requires_review_not_automatic_phishing():
    url = "https://security.service.example/report"
    analysis = apply_email_risk_policy(
        {
            "subject": "Unusual account sign-in",
            "body_clean": (
                "We detected an unusual sign-in to your account. "
                "If this wasn't you, report the activity using the supplied button."
            ),
            "auth_results": {
                "SPF": {"status": "pass"},
                "DKIM": {"status": "pass"},
                "DMARC": {"status": "pass"},
            },
            "links": [{"url": url, "host": "security.service.example"}],
            "link_reputation": {url: {"status": "clean"}},
        },
        _semantic(),
    )

    assert analysis["final_verdict"] == "review"
    assert analysis["content_risk"] == "suspicious"
    assert analysis["identity_risk"] == "verified"
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


def test_toll_debt_lure_downgrades_unsupported_information_label():
    analysis = apply_email_risk_policy(
        {
            "from_": "Pedagio Digital <nao-responder@pedagio-4166963>",
            "subject": "Notificação de débito em aberto",
            "body_for_ai": (
                "Pendência identificada. Débitos de pedágio serão encaminhados "
                "ao DETRAN, resultando em multa, pontuação e restrição veicular. "
                "R$ 195,23. Consultar Minha Placa Agora."
            ),
            "links": [{
                "url": "https://flowtrackr.site/cloaker/",
                "host": "flowtrackr.site",
            }],
            "auth_results": {
                "SPF": {"status": "none"},
                "DKIM": {"status": "none"},
                "DMARC": {"status": "none"},
            },
        },
        {
            "action": "provide_information",
            "channel": "link",
            "evidence": "R$ 195,23",
            "claimed_brand": "Pedagio Digital",
        },
    )

    assert analysis["requested_action"] == "visit_link"
    assert analysis["intent_evidence"] == "Consultar Minha Placa Agora."
    assert "financial_pretext" in analysis["intent_signals"]
    assert "threat" in analysis["intent_signals"]
    assert analysis["technical_risk"] == "uncertain"
    assert analysis["final_verdict"] == "phishing"


def test_toll_debt_without_payment_instruction_rejects_inferred_payment():
    analysis = apply_email_risk_policy(
        {
            "from_": "Pedagio Digital <nao-responder@pedagio-4166963>",
            "subject": "Notificação de débito em aberto",
            "body_for_ai": (
                "Débitos de pedágio serão encaminhados ao DETRAN, resultando "
                "em multa e restrição veicular. Consultar Minha Placa Agora."
            ),
            "links": [{
                "url": "https://flowtrackr.site/cloaker/",
                "host": "flowtrackr.site",
            }],
            "auth_results": {},
        },
        {
            "action": "payment",
            "channel": "link",
            "evidence": "R$ 195,23",
        },
    )

    assert analysis["requested_action"] == "visit_link"
    assert analysis["semantic_extraction"]["asks_for_payment"] is False
    assert analysis["intent_evidence"] == "Consultar Minha Placa Agora."


def test_authenticated_unrelated_domain_does_not_verify_claimed_wallet_brand():
    analysis = apply_email_risk_policy(
        {
            "from_": "Trust Wallet <so@viajesbereber.com>",
            "subject": "Wallet security update",
            "body_for_ai": (
                "Please click the Re-validate link and enter your old "
                "12 or 24-word private wallet phrase on the next page."
            ),
            "links": [{
                "url": "https://www.viajesbereber.com",
                "host": "www.viajesbereber.com",
            }],
            "auth_results": {
                "SPF": {"status": "pass"},
                "DKIM": {"status": "pass"},
                "DMARC": {"status": "pass"},
            },
        },
        {
            "action": "provide_credentials",
            "channel": "link",
            "evidence": "enter your old 12 or 24-word private wallet phrase",
            "credential_type": "wallet_seed",
            "claimed_brand": "Trust Wallet",
        },
    )

    assert analysis["requested_action"] == "provide_credentials"
    assert analysis["credential_type"] == "wallet_seed"
    assert analysis["semantic_extraction"]["asks_to_click_link"] is True
    assert analysis["identity_risk"] == "spoofing_evidence"
    assert analysis["final_verdict"] == "phishing"


def test_casino_deposit_is_primary_action_and_bonus_is_secondary_signal():
    analysis = apply_email_risk_policy(
        {
            "from_": "VIP ChatGPT Casino <noreply@glacierco.firebaseapp.com>",
            "subject": "Letzter Aufruf: Begrenztes Casino-Angebot",
            "body_for_ai": (
                "Willkommensbonus bis zu 3000 Euro. Jetzt bei Sportuna "
                "einzahlen. Nur noch 9 freie Plätze. Zugang läuft heute ab."
            ),
            "reply_to_mismatch": True,
            "links": [{
                "url": "https://studyingukraine.com/sports",
                "host": "studyingukraine.com",
            }],
            "auth_results": {
                "SPF": {"status": "pass"},
                "DKIM": {"status": "pass"},
                "DMARC": {"status": "permerror"},
            },
        },
        {
            "action": "claim_reward",
            "channel": "link",
            "evidence": "Jetzt kostenlos",
            "claimed_brand": "Sportuna",
        },
    )

    assert analysis["requested_action"] == "pay_or_transfer"
    assert analysis["semantic_extraction"]["asks_for_payment"] is True
    assert "incentive" in analysis["intent_signals"]
    assert "urgency" in analysis["intent_signals"]
    assert analysis["final_verdict"] == "phishing"


def test_malicious_attachment_and_hop_are_independent_technical_evidence():
    attachment_analysis = apply_email_risk_policy(
        {
            "auth_results": {},
            "attachments": [{
                "file_reputation": {
                    "status": "malicious",
                    "malicious": 4,
                },
            }],
        },
        _semantic(),
    )
    hop_analysis = apply_email_risk_policy(
        {
            "auth_results": {},
            "hop_reputation": {
                "203.0.113.9": {
                    "status": "ok",
                    "abuseConfidenceScore": 90,
                },
            },
        },
        _semantic(),
    )

    assert attachment_analysis["final_verdict"] == "phishing"
    assert "an attachment is detected as malicious" in attachment_analysis["evidence"]["technical"]
    assert hop_analysis["final_verdict"] == "phishing"
    assert "a routing hop has malicious IP reputation" in hop_analysis["evidence"]["technical"]
