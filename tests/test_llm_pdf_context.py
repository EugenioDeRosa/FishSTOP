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
                    "summary": "embedded JavaScript x1; automatic action on document open x1",
                    "behaviors": [
                        {"key": "javascript", "label": "contains executable JavaScript", "severity": "high", "count": 1},
                        {"key": "open_action", "label": "runs an action when the document opens", "severity": "high", "count": 1},
                    ],
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
    assert "PDF malicious behaviors" in prompt
    assert "PDF internal indicators" in prompt
    assert "embedded JavaScript severity=high count=1" in prompt
    assert "automatic action on document open severity=high count=1" in prompt

def test_phi4_prompt_does_not_duplicate_pdf_flags_in_soc_flags():
    soc = {
        "from_": "sender@example.com",
        "to": "analyst@example.net",
        "subject": "Invoice",
        "body_clean": "Please see the attached invoice.",
        "body_source": "text/plain",
        "auth_results": {},
        "arc_auth_results": {},
        "dkim_signature_present": True,
        "reply_to_mismatch": False,
        "return_path_domain_mismatch": False,
        "display_name_spoofing": None,
        "links": [],
        "lookalike_alerts": [],
        "flags": [
            {"level": "HIGH", "field": "PDF Attachment", "message": "risky PDF features detected - external URI action x9"},
            {"level": "MEDIUM", "field": "PDF Content", "message": "internal PDF indicator - external URI action x9"},
            {"level": "HIGH", "field": "Attachment", "message": "Content-Type mismatch"},
        ],
        "attachments": [
            {
                "filename": "invoice.pdf",
                "extension_from_filename": "pdf",
                "content_type": "application/pdf",
                "magic_detected_format": "pdf",
                "anomaly": "PDF risk CRITICAL: external URI action x9",
                "pdf_security": {
                    "is_pdf": True,
                    "risk_level": "critical",
                    "suspicious": True,
                    "summary": "external URI action x9",
                    "behaviors": [],
                    "indicators": [
                        {"label": "external URI action", "severity": "medium", "count": 9},
                    ],
                },
            }
        ],
    }

    prompt = build_fast_email_prompt(soc)

    assert "PDF malicious behaviors: none" in prompt
    assert "PDF internal indicators: external URI action severity=medium count=9" in prompt
    assert "anomaly=none pdf_risk=critical" in prompt
    assert "- HIGH PDF Attachment" not in prompt
    assert "- MEDIUM PDF Content" not in prompt
    assert "- HIGH Attachment: Content-Type mismatch" in prompt

