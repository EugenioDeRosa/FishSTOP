from src.analyzer.soc_analyzer import EmlSOCAnalyzer


def test_missing_spf_result_is_medium_flag():
    report = {
        "auth_results": {"DMARC": {"status": "pass"}},
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
        "field": "SPF",
        "message": "No SPF result found in headers",
    } in flags
