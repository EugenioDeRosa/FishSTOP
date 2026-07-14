from src.analyzer.llm_context_analyzer import build_fast_email_prompt


def test_phi4_prompt_includes_internal_pdf_indicators():
    soc = {
        "from_": "sender@example.com",
        "to": "analyst@example.net",
        "subject": "Invoice",
        "body_clean": "Please see the attached invoice.",
        "body_source": "text/plain",
        "auth_results": {},
        "arc_auth_results": {},
        "dkim_signature_present": False,
        "reply_to_mismatch": False,
        "return_path_domain_mismatch": False,
        "display_name_spoofing": None,
        "links": [],
        "lookalike_alerts": [],
        "flags": [],
        "attachments": [
            {
                "filename": "invoice.pdf",
                "extension_from_filename": "pdf",
                "content_type": "application/pdf",
                "magic_detected_format": "pdf",
                "anomaly": None,
                "pdf_security": {
                    "is_pdf": True,
                    "risk_level": "high",
                    "suspicious": True,
                    "score": 80,
                    "summary": "embedded JavaScript x1; automatic action on document open x1",
                    "indicators": [
                        {"label": "embedded JavaScript", "severity": "high", "count": 1},
                        {"label": "automatic action on document open", "severity": "high", "count": 1},
                    ],
                },
            }
        ],
    }

    prompt = build_fast_email_prompt(soc)

    assert "IMPORTANT phishing indicator: PDF contains risky active/internal features" in prompt
    assert "PDF internal indicators" in prompt
    assert "embedded JavaScript severity=high count=1" in prompt
    assert "automatic action on document open severity=high count=1" in prompt
