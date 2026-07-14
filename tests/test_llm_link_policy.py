import inspect

from src.analyzer import llm_context_analyzer
from src.analyzer.llm_context_analyzer import build_fast_email_prompt


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
    prompt_source = inspect.getsource(llm_context_analyzer.stream_phi4_email_analysis)

    assert "Here is our monthly update" in prompt
    assert "A link by itself is not suspicious" in prompt_source
    assert "clean/unknown/tracking/generic links" in prompt_source
