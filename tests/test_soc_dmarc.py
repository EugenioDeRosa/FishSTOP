from src.analyzer.soc_analyzer import EmlSOCAnalyzer


def test_missing_dmarc_policy_is_medium_flag():
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
        "attachments": [],
        "links": [],
        "lookalike_alerts": [],
    }

    flags = EmlSOCAnalyzer._build_flags(report)

    assert {
        "level": "MEDIUM",
        "field": "DMARC",
        "message": "No DMARC policy detected in headers",
    } in flags


def test_failed_dmarc_policy_is_medium_flag():
    report = {
        "auth_results": {"DMARC": {"status": "fail"}},
        "arc_auth_results": {},
        "dkim_signature_present": True,
        "reply_to_mismatch": False,
        "reply_to_mismatch_legitimate": False,
        "return_path_domain_mismatch": False,
        "return_path": None,
        "html_strip_applied": False,
        "display_name_spoofing": None,
        "injection_server": {},
        "attachments": [],
        "links": [],
        "lookalike_alerts": [],
    }

    flags = EmlSOCAnalyzer._build_flags(report)

    assert {
        "level": "MEDIUM",
        "field": "DMARC",
        "message": "DMARC FAIL",
    } in flags
    assert not any(flag["field"] == "DMARC" and flag["level"] == "HIGH" for flag in flags)
