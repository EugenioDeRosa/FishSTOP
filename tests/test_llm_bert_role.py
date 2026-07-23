import inspect

from src.analyzer import llm_context_analyzer
from src.analyzer.llm_context_analyzer import (
    apply_email_risk_policy,
    build_fast_email_prompt,
    format_email_risk_analysis,
)


def test_bert_is_passed_as_compact_supporting_context_not_as_the_verdict():
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

    assert "BERT=phishing" in prompt
    assert "SUBJECT: Service improvement discussion" in prompt
    assert "Semantic analysis (BERT): phishing" not in prompt


def test_phi4_instructions_keep_bert_supporting_and_out_of_content_summary():
    prompt_source = llm_context_analyzer.TASK_INSTRUCTIONS

    assert "BERT=support only" in prompt_source
    assert "content only; no verdict/checks" in prompt_source
    assert "link or urgency alone is neutral" in prompt_source


def test_bert_is_reported_after_intent_analysis_as_support_or_contrary_evidence():
    phishing = apply_email_risk_policy(
        {"auth_results": {}, "bert_ai_result": "phishing"},
        {
            "action": "provide_credentials",
            "channel": "reply",
            "signals": ["credentials"],
            "summary": "The email body requests credentials by reply, a strong phishing pattern.",
        },
    )
    legitimate = apply_email_risk_policy(
        {"auth_results": {}, "bert_ai_result": "phishing"},
        {
            "action": "info",
            "channel": "none",
            "signals": [],
            "summary": "The subject and body provide routine information without requesting risky action.",
        },
    )

    assert "BERT classified the content as phishing" in phishing["corroboration"]["details"]
    assert "BERT classified the content as phishing" in legitimate["corroboration"]["caveats"]
    assert "BERT classified the content as phishing" in format_email_risk_analysis(phishing)
    assert legitimate["final_verdict"] == "legitimate"
