from src.analyzer.lookalike import check_lookalike_domains, decode_punycode_domain


def _link(url: str, host: str) -> dict:
    return {
        "url": url,
        "host": host,
        "source": "test",
        "scheme": "https",
        "display_text": "",
        "is_ip": False,
    }


def test_decode_punycode_domain():
    decoded = decode_punycode_domain("xn--pple-43d.com")

    assert decoded.endswith("pple.com")
    assert decoded != "xn--pple-43d.com"


def test_punycode_idna_alert_is_general():
    alerts = check_lookalike_domains([
        _link("https://xn--pple-43d.com/login", "xn--pple-43d.com")
    ])

    assert any(
        alert["technique"] == "punycode_idna"
        and alert["matched_brand"] == "-"
        for alert in alerts
    )


def test_unicode_homoglyph_alert_is_general():
    alerts = check_lookalike_domains([
        _link("https://docs.gооgle.com/doc", "docs.gооgle.com")
    ])

    assert any(
        alert["technique"] == "unicode_homoglyph"
        and alert["matched_brand"] == "-"
        for alert in alerts
    )


def test_exact_brand_does_not_alert():
    alerts = check_lookalike_domains([
        _link("https://apple.com", "apple.com")
    ])

    assert alerts == []
