import inspect

from src.analyzer import llm_context_analyzer
from src.analyzer.llm_context_analyzer import apply_email_risk_policy, build_fast_email_prompt


def test_bert_and_auth_medium_without_body_risk_must_not_be_suspicious():
    soc = {
        "from_": "sales@example.com",
        "to": "customer@example.net",
        "subject": "Servizio post-vendita",
        "body_ai": (
            "Gentile Chiara, vorrei condividere come alcuni clienti stanno migliorando "
            "il servizio post-vendita. Resto disponibile per un confronto in uno slot comodo."
        ),
        "body_source": "text/plain",
        "auth_results": {
            "SPF": {"status": "pass"},
            "DKIM": {"status": "none"},
            "DMARC": {"status": "none"},
        },
        "arc_auth_results": {},
        "dkim_signature_present": False,
        "reply_to_mismatch": False,
        "return_path_domain_mismatch": False,
        "display_name_spoofing": None,
        "bert_ai_result": "phishing",
        "links": [],
        "link_reputation": {},
        "lookalike_alerts": [],
        "flags": [
            {"level": "MEDIUM", "field": "DKIM", "message": "DKIM NONE - signature validation should be reviewed"},
            {"level": "MEDIUM", "field": "DMARC", "message": "DMARC NONE"},
        ],
        "attachments": [],
    }

    prompt = build_fast_email_prompt(soc)
    prompt_source = inspect.getsource(llm_context_analyzer.stream_phi4_email_analysis)

    assert "Identity anomaly summary: none" in prompt
    assert "BERT result: available to FishSTOP UI only; not provided as verdict evidence to Phi-4" in prompt
    assert "never use weak-only evidence for a suspicious verdict" in prompt_source
    analysis = apply_email_risk_policy(soc, {
        "requested_action": "informational",
        "action_channel": "none",
        "content_risk": "benign",
        "confidence": 0.9,
        "reason": "Ordinary business communication",
    })
    assert analysis["final_verdict"] == "legitimate"
    assert analysis["identity_risk"] == "uncertain"


def test_business_finance_terms_are_not_payment_request_without_explicit_ask():
    prompt_source = inspect.getsource(llm_context_analyzer.stream_phi4_email_analysis)

    assert "pay/settle/transfer money" in prompt_source
    assert "business-process discussion" in llm_context_analyzer.SYSTEM_MESSAGE
    assert "unless it includes a risky action above" in prompt_source
