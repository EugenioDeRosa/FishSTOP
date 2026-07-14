import inspect

from src.analyzer import llm_context_analyzer
from src.analyzer.llm_context_analyzer import build_fast_email_prompt


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
    assert "If there is no risky requested action and no strong support, you MUST classify as not suspicious" in prompt_source
    assert "never use weak-only evidence for a suspicious verdict" in prompt_source


def test_business_finance_terms_are_not_payment_request_without_explicit_ask():
    prompt_source = inspect.getsource(llm_context_analyzer.stream_phi4_email_analysis)

    assert "pay/settle/transfer money" in prompt_source
    assert "business-process discussion" in prompt_source
    assert "unless it includes a risky action above" in prompt_source
