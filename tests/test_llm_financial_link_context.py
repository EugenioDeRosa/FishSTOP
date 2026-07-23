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
    prompt_source = llm_context_analyzer.TASK_INSTRUCTIONS

    assert "payment or transfer" in prompt_source
    assert "Marketing, scheduling, sales and business are benign without these" in prompt_source

def test_phi4_prompt_forbids_not_suspicious_payment_clean_vt_reasoning():
    prompt_source = llm_context_analyzer.TASK_INSTRUCTIONS

    assert "JSON only" in prompt_source
    assert "payment or transfer" in prompt_source
    assert "content only; no verdict/checks" in prompt_source
    assert '"signals"' in prompt_source
    assert '"asks_to_click_link"' not in prompt_source
    assert '"content_risk"' not in prompt_source


def test_phi4_gets_extracted_link_for_invoice_payment_request_without_vt():
    prompt_source = llm_context_analyzer.TASK_INSTRUCTIONS

    assert "META link/file supplies an invoice/payment channel" in prompt_source
    assert "DMARC" not in prompt_source
    assert "VirusTotal" not in prompt_source

    soc = {
        "from_": "billing@example.com",
        "to": "accounting@example.net",
        "subject": "Faktura 26839907",
        "body_clean": (
            "Szanowny Panie, Oczekujemy na przelew bankowy pozosta?ej kwoty z faktury numer 26839907 "
            "w wysoko?ci 8027,69 USD. W za??czeniu przesy?amy faktur? oczekuj?c? na p?atno??. "
            "https://cdn.discordapp.com/attachments/1496187481362792538/1514522506084745246/Invoice63784.jse"
        ),
        "body_source": "text/plain",
        "auth_results": {"SPF": {"status": "pass"}, "DKIM": {"status": "pass"}, "DMARC": {"status": "unknown"}},
        "arc_auth_results": {},
        "dkim_signature_present": True,
        "reply_to_mismatch": False,
        "return_path_domain_mismatch": False,
        "display_name_spoofing": None,
        "bert_ai_result": "legitimate",
        "links": [{
            "url": "https://cdn.discordapp.com/attachments/1496187481362792538/1514522506084745246/Invoice63784.jse",
            "host": "cdn.discordapp.com",
            "source": "plain_text",
            "is_ip": False,
            "display_mismatch": False,
        }],
        "link_reputation": {},
        "lookalike_alerts": [],
        "flags": [],
        "attachments": [],
    }

    prompt = build_fast_email_prompt(soc)

    assert "przelew bankowy" in prompt
    assert "faktury numer 26839907" in prompt
    assert "[URL LINK]" in prompt
    assert "META: links=1; attachments=0" in prompt
    assert "generic extracted link" not in prompt
    assert "source=plain_text" not in prompt
