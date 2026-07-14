from src.analyzer.soc_analyzer import EmlSOCAnalyzer
from src.views.analyzer import _main_alert_flags


def test_pdf_internal_indicators_are_soc_flags():
    report = {
        "auth_results": {},
        "arc_auth_results": {},
        "dkim_signature_present": True,
        "reply_to_mismatch": False,
        "reply_to_mismatch_legitimate": False,
        "return_path_domain_mismatch": False,
        "return_path": None,
        "html_strip_applied": False,
        "display_name_spoofing": None,
        "injection_server": {},
        "attachments": [
            {
                "filename": "invoice.pdf",
                "anomaly": None,
                "magic_bytes_hex": None,
                "pdf_security": {
                    "risk_level": "high",
                    "suspicious": True,
                    "summary": "embedded JavaScript x1; automatic action on document open x1",
                    "behaviors": [
                        {"label": "contains executable JavaScript", "severity": "high", "count": 1},
                        {"label": "runs an action when the document opens", "severity": "high", "count": 1},
                    ],
                    "indicators": [
                        {"label": "embedded JavaScript", "severity": "high", "count": 1},
                        {"label": "automatic action on document open", "severity": "high", "count": 1},
                    ],
                },
            }
        ],
        "links": [],
        "lookalike_alerts": [],
    }

    flags = EmlSOCAnalyzer._build_flags(report)
    pdf_content_flags = [flag for flag in flags if flag["field"] == "PDF Content"]

    assert len(pdf_content_flags) == 2
    assert all(flag["level"] == "HIGH" for flag in pdf_content_flags)
    assert any("contains executable JavaScript" in flag["message"] for flag in pdf_content_flags)
    assert any("runs an action when the document opens" in flag["message"] for flag in pdf_content_flags)


def test_main_alerts_prioritize_pdf_content_flags():
    flags = [
        {"level": "INFO", "field": "Attachment", "message": "magic bytes"},
        {"level": "MEDIUM", "field": "DMARC", "message": "DMARC FAIL"},
        {"level": "HIGH", "field": "PDF Content", "message": "embedded JavaScript"},
        {"level": "MEDIUM", "field": "SPF", "message": "SPF SOFTFAIL"},
    ]

    main = _main_alert_flags(flags)

    assert main[0]["field"] == "PDF Content"
    assert main[0]["message"] == "embedded JavaScript"

def test_pdf_risk_anomaly_does_not_create_duplicate_attachment_flag():
    report = {
        "auth_results": {},
        "arc_auth_results": {},
        "dkim_signature_present": True,
        "reply_to_mismatch": False,
        "reply_to_mismatch_legitimate": False,
        "return_path_domain_mismatch": False,
        "return_path": None,
        "html_strip_applied": False,
        "display_name_spoofing": None,
        "injection_server": {},
        "attachments": [
            {
                "filename": "invoice.pdf",
                "anomaly": "PDF risk CRITICAL: external URI action x9",
                "magic_bytes_hex": None,
                "pdf_security": {
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
        "links": [],
        "lookalike_alerts": [],
    }

    flags = EmlSOCAnalyzer._build_flags(report)

    assert not [flag for flag in flags if flag["field"] == "Attachment"]
    assert [flag for flag in flags if flag["field"] == "PDF Attachment"]
    assert not [flag for flag in flags if flag["field"] == "PDF Content"]

