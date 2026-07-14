from src.analyzer.soc_analyzer import EmlSOCAnalyzer


def _base_report(dkim_status=None, signature_present=True):
    auth_results = {}
    if dkim_status is not None:
        auth_results["DKIM"] = {"status": dkim_status}
    return {
        "auth_results": auth_results,
        "arc_auth_results": {},
        "dkim_signature_present": signature_present,
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


def test_any_non_pass_dkim_result_is_medium():
    for status in ["fail", "permerror", "neutral", "none", "temperror"]:
        flags = EmlSOCAnalyzer._build_flags(_base_report(status))

        assert {
            "level": "MEDIUM",
            "field": "DKIM",
            "message": f"DKIM {status.upper()} - signature validation should be reviewed",
        } in flags
        assert not any(flag["field"] == "DKIM" and flag["level"] == "HIGH" for flag in flags)


def test_missing_dkim_signature_is_medium():
    flags = EmlSOCAnalyzer._build_flags(_base_report(None, signature_present=False))

    assert {
        "level": "MEDIUM",
        "field": "DKIM",
        "message": "DKIM signature missing from headers",
    } in flags
