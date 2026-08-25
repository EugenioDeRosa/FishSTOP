from src.analyzer.soc_analyzer import EmlSOCAnalyzer


def test_visible_text_mismatch_does_not_raise_link_alert():
    report = {
        "auth_results": {"dmarc": {"result": "pass"}},
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
        "links": [
            {
                "url": "https://or5.mailsap.com/click",
                "host": "or5.mailsap.com",
                "display_host": "filippo.zerbini",
                "display_mismatch": True,
                "is_ip": False,
            }
        ],
        "lookalike_alerts": [],
    }

    flags = EmlSOCAnalyzer._build_flags(report)

    assert not any(
        flag["field"] == "Link" and "possible masked link" in flag["message"]
        for flag in flags
    )
