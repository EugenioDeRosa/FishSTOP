import inspect

from src.analyzer import llm_context_analyzer
from src.analyzer.llm_context_analyzer import build_fast_email_prompt


def test_sales_followup_with_bert_and_auth_medium_flags_is_not_bec_by_itself():
    soc = {
        "from_": "sales@example.com",
        "to": "customer@example.net",
        "subject": "Servizio post-vendita",
        "body_ai": (
            "Gentile Chiara,\n\n"
            "torno a contattarla un'ultima volta per assicurarmi che i miei precedenti messaggi non si siano persi.\n"
            "Vorrei condividere come altri clienti stanno trasformando il servizio post-vendita e riducendo i tempi di intervento.\n\n"
            "Se questi temi sono rilevanti, resto disponibile per un confronto in uno slot comodo."
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

    assert "torno a contattarla un'ultima volta" in prompt
    assert "BERT result: available to FishSTOP UI only; not provided as verdict evidence to Phi-4" in prompt
    assert "ordinary marketing, sales follow-up" in llm_context_analyzer.SYSTEM_MESSAGE
    assert "business-process discussion" in llm_context_analyzer.SYSTEM_MESSAGE
    assert "unless it includes a risky action above" in prompt_source
