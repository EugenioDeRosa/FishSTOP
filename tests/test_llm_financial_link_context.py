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

    assert "Return this JSON schema exactly" in prompt_source
    assert "pay/settle/transfer money" in prompt_source
    assert "use technical facts only as support" in prompt_source


def test_phi4_gets_extracted_link_for_invoice_payment_request_without_vt():
    prompt_source = inspect.getsource(llm_context_analyzer.stream_phi4_email_analysis)

    assert "invoice/payment/bank-transfer request" in prompt_source
    assert "combined with an extracted link or attachment" in prompt_source
    assert "DMARC is unknown" in prompt_source
    assert "VirusTotal is clean/unavailable" in prompt_source

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
    assert "[URL]" in prompt
    assert "Link action signals" in prompt
    assert "generic extracted link" in prompt
    assert "source=plain_text" in prompt
