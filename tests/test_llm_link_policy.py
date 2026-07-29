import inspect
from pathlib import Path

from src.analyzer import llm_context_analyzer
from src.analyzer.llm_context_analyzer import (
    apply_email_risk_policy,
    build_fast_email_prompt,
    stream_phi4_email_analysis,
)
from src.analyzer.soc_analyzer import EmlSOCAnalyzer


def test_phi4_prompt_says_link_alone_is_not_suspicious():
    soc = {
        "from_": "newsletter@example.com",
        "to": "user@example.net",
        "subject": "Monthly update",
        "body_ai": "Here is our monthly update. Read more on our website.",
        "body_source": "text/plain",
        "auth_results": {"SPF": {"status": "pass"}, "DKIM": {"status": "pass"}, "DMARC": {"status": "pass"}},
        "arc_auth_results": {},
        "dkim_signature_present": True,
        "reply_to_mismatch": False,
        "return_path_domain_mismatch": False,
        "display_name_spoofing": None,
        "bert_ai_result": "phishing",
        "links": [{"url": "https://example.com/news", "host": "example.com", "is_ip": False}],
        "link_reputation": {"https://example.com/news": {"status": "clean", "detection_ratio": "0 / 90"}},
        "lookalike_alerts": [],
        "flags": [],
        "attachments": [],
    }

    prompt = build_fast_email_prompt(soc)
    prompt_source = llm_context_analyzer.TASK_INSTRUCTIONS

    assert "Here is our monthly update" in prompt
    assert "A link or urgency alone is neutral" in prompt_source
    assert "visit_link=explicit browsing only if no more specific action" in prompt_source
    assert "VirusTotal" not in prompt
    assert "url_rep" not in prompt


def test_phi4_receives_only_the_presence_of_a_direct_ip_link():
    prompt_source = llm_context_analyzer.TASK_INSTRUCTIONS

    assert "VirusTotal" not in prompt_source
    assert "IP/geolocation" not in prompt_source

    prompt = build_fast_email_prompt({
        "from_": "sender@example.com",
        "to": "recipient@example.net",
        "subject": "Document",
        "body_clean": "Open the document here: http://192.0.2.10/login",
        "body_source": "text/plain",
        "auth_results": {},
        "arc_auth_results": {},
        "dkim_signature_present": True,
        "reply_to_mismatch": False,
        "return_path_domain_mismatch": False,
        "display_name_spoofing": None,
        "links": [{"url": "http://192.0.2.10/login", "host": "192.0.2.10", "is_ip": True}],
        "lookalike_alerts": [],
        "flags": [{"level": "HIGH", "field": "Link", "message": "URL with bare IP detected"}],
    })

    assert "META: links=1; attachments=0" in prompt
    assert "direct_ip_link" not in prompt
    assert "direct IP link" not in prompt
    assert "strong phishing infrastructure signal" not in prompt
    assert "URL with bare IP detected" not in prompt


def test_phi4_does_not_receive_auth_reputation_or_bert_context():
    prompt = build_fast_email_prompt({
        "subject": "Account review",
        "body_clean": "Please review your account using the supplied link.",
        "auth_results": {
            "SPF": {"status": "pass"},
            "DKIM": {"status": "none"},
            "DMARC": {"status": "fail"},
        },
        "links": [{"url": "https://bad.example", "host": "bad.example"}],
        "link_reputation": {
            "https://bad.example": {"status": "malicious", "malicious": 8},
        },
        "hop_reputation": {
            "198.51.100.8": {"status": "ok", "abuseConfidenceScore": 12},
        },
        "domain_reputation": {
            "sender.example": {"status": "ok", "abuseConfidenceScore": 81},
        },
        "attachments": [{
            "file_reputation": {"status": "suspicious", "suspicious": 2},
        }],
        "bert_ai_result": "phishing",
    })

    for hidden_check in (
        "CHECKS:", "SPF=", "DKIM=", "DMARC=", "url_rep", "file_rep",
        "domain_rep", "hop_rep", "BERT=",
    ):
        assert hidden_check not in prompt
    assert "technical checks" in llm_context_analyzer.TASK_INSTRUCTIONS


def test_phi4_maps_response_to_claimed_account_alert_to_verification_action():
    prompt_source = llm_context_analyzer.TASK_INSTRUCTIONS

    assert "verify_account=confirm/deny/report account activity" in prompt_source


def test_extracted_footer_link_is_metadata_not_a_standalone_body_action():
    prompt = build_fast_email_prompt({
        "subject": "Monthly update",
        "body_for_ai": "Here is the monthly service update.",
        "links": [{
            "url": "https://newsletter.example/unsubscribe",
            "host": "newsletter.example",
            "source": "html_href",
        }],
        "attachments": [],
    })

    assert "META: links=1" in prompt
    assert "[URL LINK]" not in prompt


def test_test4_signature_link_cannot_turn_bert_false_positive_into_phishing():
    fixture = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "raw"
        / "custom_legitimate"
        / "test4.eml"
    )
    soc = EmlSOCAnalyzer().analyze(str(fixture))
    soc["bert_ai_result"] = "phishing"
    soc["link_reputation"] = {
        link["url"]: {"status": "clean"}
        for link in soc["links"]
    }

    prompt = build_fast_email_prompt(soc)
    analysis = apply_email_risk_policy(
        soc,
        {
            "action": "visit_link",
            "channel": "link",
            "evidence": "www.cefla.com",
        },
    )

    assert soc["body_for_intent"] == "Test email per check SPF, DKIM ecc"
    assert soc["body_for_ai"] == "Test email per check SPF, DKIM ecc"
    assert len(soc["links"]) == 1
    assert soc["links"][0]["role"] == "signature"
    assert soc["links"][0]["actionable"] is False
    assert "META: links=0" in prompt
    assert "www.cefla.com" not in prompt
    assert analysis["requested_action"] == "informational"
    assert analysis["action_channel"] == "none"
    assert analysis["content_risk"] == "benign"
    assert analysis["technical_risk"] == "clean"
    assert analysis["final_verdict"] == "legitimate"


def test_test4_inline_logo_cannot_become_an_open_attachment_request():
    fixture = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "raw"
        / "custom_legitimate"
        / "test4.eml"
    )
    soc = EmlSOCAnalyzer().analyze(str(fixture))
    soc["bert_ai_result"] = "phishing"
    soc["link_reputation"] = {
        link["url"]: {"status": "clean"}
        for link in soc["links"]
    }

    prompt = build_fast_email_prompt(soc)
    analysis = apply_email_risk_policy(
        soc,
        {
            "action": "open_attachment",
            "channel": "attachment",
            "evidence": "[cid:d4754c29-aa04-4888-93c7-4b8ff339f14e]",
        },
    )

    assert len(soc["attachments"]) == 1
    assert soc["attachments"][0]["mime_role"] == "inline_resource"
    assert soc["attachments"][0]["actionable"] is False
    assert soc["attachments"][0]["content_disposition"] == "inline"
    assert soc["attachments"][0]["content_id"]
    assert "META: links=0; attachments=0; types=none" in prompt
    assert analysis["requested_action"] == "informational"
    assert analysis["action_channel"] == "none"
    assert analysis["content_risk"] == "benign"
    assert analysis["technical_risk"] == "clean"
    assert analysis["final_verdict"] == "legitimate"


def test_explicit_request_to_open_real_attachment_remains_actionable():
    soc = {
        "subject": "Document attached",
        "body_for_intent": "Please open the attached document.",
        "links": [],
        "attachments": [{
            "filename": "document.pdf",
            "mime_role": "attachment",
            "actionable": True,
        }],
        "auth_results": {},
        "bert_ai_result": "phishing",
    }

    analysis = apply_email_risk_policy(
        soc,
        {
            "action": "open_attachment",
            "channel": "attachment",
            "evidence": "open the attached document",
        },
    )

    assert analysis["requested_action"] == "open_attachment"
    assert analysis["action_channel"] == "supplied_attachment"
    assert analysis["final_verdict"] == "phishing"


def test_explicit_body_call_to_action_remains_actionable():
    url = "https://example.test/review"
    soc = {
        "subject": "Review",
        "body_for_intent": "Please click here to review the account.",
        "links": [{
            "url": url,
            "host": "example.test",
            "display_text": "click here",
            "role": "body_action",
            "actionable": True,
        }],
        "link_reputation": {url: {"status": "clean"}},
        "auth_results": {},
        "bert_ai_result": "phishing",
    }

    analysis = apply_email_risk_policy(
        soc,
        {
            "action": "visit_link",
            "channel": "link",
            "evidence": "click here",
        },
    )

    assert analysis["requested_action"] == "visit_link"
    assert analysis["action_channel"] == "supplied_link"
    assert analysis["final_verdict"] == "phishing"
