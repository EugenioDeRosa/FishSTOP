import inspect

from src.analyzer import llm_context_analyzer
from src.analyzer.llm_context_analyzer import build_fast_email_prompt, stream_phi4_email_analysis


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
    assert "VirusTotal" not in prompt
    assert "url_rep=clean:1" in prompt


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
    assert "direct_ip_link=true" in prompt
    assert "direct IP link" not in prompt
    assert "strong phishing infrastructure signal" not in prompt
    assert "URL with bare IP detected" not in prompt


def test_phi4_receives_normalized_reputation_context_with_malicious_as_strong():
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

    assert "CHECKS: SPF=pass; DKIM=none; DMARC=fail" in prompt
    assert "url_rep=malicious:1" in prompt
    assert "file_rep=suspicious:1" in prompt
    assert "domain_rep=malicious:1" in prompt
    assert "hop_rep=low_risk:1" in prompt
    assert "BERT=phishing" in prompt
    assert "malicious URL/domain/file=strong phishing evidence" in (
        llm_context_analyzer.TASK_INSTRUCTIONS
    )
