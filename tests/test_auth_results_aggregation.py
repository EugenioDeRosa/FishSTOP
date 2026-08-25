from src.analyzer.received_parser import (
    merge_auth_results,
    parse_auth_results,
    parse_received_spf_results,
)


def test_parse_auth_results_keeps_worst_status_and_all_evidence():
    parsed = parse_auth_results(
        "spf=pass smtp.mailfrom=example.com; dmarc=pass header.from=example.com\n"
        "spf=softfail smtp.mailfrom=example.com; dmarc=fail header.from=example.com"
    )

    assert parsed["SPF"]["status"] == "softfail"
    assert parsed["DMARC"]["status"] == "fail"
    assert [item["status"] for item in parsed["SPF"]["all_results"]] == ["pass", "softfail"]


def test_merge_auth_results_does_not_hide_arc_failures_behind_passes():
    direct = parse_auth_results(
        "spf=pass smtp.mailfrom=cefla.it; dkim=none header.d=none; "
        "dmarc=pass header.from=cefla.it"
    )
    arc = parse_auth_results(
        "i=3; spf=pass smtp.mailfrom=cefla.it; dmarc=pass header.from=cefla.it; dkim=none\n"
        "i=2; spf=softfail smtp.mailfrom=cefla.it; dmarc=fail header.from=cefla.it\n"
        "i=1; spf=softfail smtp.mailfrom=cefla.it; dmarc=fail header.from=cefla.it"
    )

    effective = merge_auth_results(
        ("Authentication-Results", direct),
        ("ARC-Authentication-Results", arc),
    )

    assert effective["SPF"]["status"] == "softfail"
    assert effective["SPF"]["source"] == "ARC-Authentication-Results"
    assert effective["DKIM"]["status"] == "none"
    assert effective["DMARC"]["status"] == "fail"
    assert len(effective["DMARC"]["all_results"]) == 4


def test_multiple_received_spf_headers_keep_softfail_over_pass():
    parsed = parse_received_spf_results([
        "Pass (receiver: sender permitted) client-ip=209.85.128.69",
        "softfail (receiver: sender not permitted) client-ip=119.2.43.115",
    ])

    assert parsed["SPF"]["status"] == "softfail"
    assert len(parsed["SPF"]["all_results"]) == 2
