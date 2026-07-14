import inspect

from src.analyzer import llm_context_analyzer
from src.analyzer.llm_context_analyzer import build_fast_email_prompt


def test_bert_is_presented_as_supporting_pressure_signal_not_verdict():
    soc = {
        "from_": "sales@example.com",
        "to": "customer@example.net",
        "subject": "Service improvement discussion",
        "body_clean": "Gentile Chiara, vorrei condividere come alcuni clienti stanno migliorando il servizio post-vendita. Resto disponibile per un confronto in uno slot comodo.",
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

    assert "BERT result: available to FishSTOP UI only; not provided as verdict evidence to Phi-4" in prompt
    assert "Semantic analysis (BERT): phishing" not in prompt


def test_phi4_instructions_forbid_using_bert_as_phishing_reason():
    prompt_source = inspect.getsource(llm_context_analyzer.stream_phi4_email_analysis)

    assert "do not mention BERT" in prompt_source
    assert "BERT" in prompt_source
    assert "Weak only" in prompt_source
    assert "never use weak-only evidence for a suspicious verdict" in prompt_source
