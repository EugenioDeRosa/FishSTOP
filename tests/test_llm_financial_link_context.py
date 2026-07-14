import inspect

from src.analyzer import llm_context_analyzer
from src.analyzer.llm_context_analyzer import build_fast_email_prompt


def test_phi4_prompt_uses_semantic_invoice_payment_instruction_without_local_heuristic():
    soc = {
        "from_": "billing@example.com",
        "to": "accounting@example.net",
        "subject": "Invoice payment required",
        "body_clean": "Please pay the attached invoice by bank transfer. Open the payment details here: https://example-host.test/payment",
        "body_source": "text/plain",
        "auth_results": {
            "SPF": {"status": "pass"},
            "DKIM": {"status": "pass"},
            "DMARC": {"status": "pass"},
        },
        "arc_auth_results": {},
        "dkim_signature_present": True,
        "reply_to_mismatch": False,
        "return_path_domain_mismatch": False,
        "display_name_spoofing": None,
        "bert_ai_result": "legitimate",
        "links": [
            {
                "url": "https://example-host.test/payment",
                "host": "example-host.test",
                "is_ip": False,
                "display_mismatch": False,
            }
        ],
        "link_reputation": {
            "https://example-host.test/payment": {
                "status": "clean",
                "detection_ratio": "0 / 90",
                "crowdsourced_context_summary": "",
            }
        },
        "lookalike_alerts": [],
        "flags": [],
        "attachments": [],
    }

    prompt = build_fast_email_prompt(soc)

    assert "Financial link request" not in prompt
    assert "Please pay the attached invoice by bank transfer" in prompt

def test_phi4_prompt_forbids_clean_technical_checks_overriding_payment_request():
    soc = {
        "from_": "billing@example.com",
        "to": "accounting@example.net",
        "subject": "Payment request",
        "body_clean": "Please make a bank transfer for this invoice using the payment instructions in the linked document.",
        "body_source": "text/plain",
        "auth_results": {
            "SPF": {"status": "pass"},
            "DKIM": {"status": "pass"},
            "DMARC": {"status": "pass"},
        },
        "arc_auth_results": {},
        "dkim_signature_present": True,
        "reply_to_mismatch": False,
        "return_path_domain_mismatch": False,
        "display_name_spoofing": None,
        "bert_ai_result": "legitimate",
        "links": [],
        "link_reputation": {},
        "lookalike_alerts": [],
        "flags": [],
        "attachments": [],
    }

    _ = build_fast_email_prompt(soc)
    prompt_source = inspect.getsource(llm_context_analyzer.stream_phi4_email_analysis)

    assert "pay/settle/transfer money" in prompt_source
    assert "unless it includes a risky action above" in prompt_source

def test_phi4_prompt_forbids_not_suspicious_payment_clean_vt_reasoning():
    prompt_source = inspect.getsource(llm_context_analyzer.stream_phi4_email_analysis)

    assert "Start exactly with 'The email provided is suspicious because'" in prompt_source
    assert "pay/settle/transfer money" in prompt_source
    assert "If there is no risky requested action and no strong support" in prompt_source
    assert "use technical facts only as support" in prompt_source
