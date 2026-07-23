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
    prompt_source = llm_context_analyzer.TASK_INSTRUCTIONS

    assert "Identity anomaly summary" not in prompt
    assert "BERT result" not in prompt
    assert "A link or urgency alone is neutral" in prompt_source
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
    prompt_source = llm_context_analyzer.TASK_INSTRUCTIONS

    assert "payment or transfer" in prompt_source
    assert "business" in prompt_source
    assert "info unless it explicitly requests action" in prompt_source
