from email.message import EmailMessage

from src.analyzer.soc_analyzer import EmlSOCAnalyzer


def _write_eml(tmp_path, *, from_value, reply_to=None, headers=None):
    msg = EmailMessage()
    msg["From"] = from_value
    msg["To"] = "analyst@example.net"
    msg["Subject"] = "Reply-To test"
    if reply_to:
        msg["Reply-To"] = reply_to
    for name, value in (headers or {}).items():
        msg[name] = value
    msg.set_content("Hello")

    path = tmp_path / "message.eml"
    path.write_bytes(msg.as_bytes())
    return path


def _flag_levels(report, field):
    return [flag["level"] for flag in report["flags"] if flag["field"] == field]


def test_reply_to_same_registered_domain_is_info_not_high(tmp_path):
    path = _write_eml(
        tmp_path,
        from_value="No Reply <no-reply@example.com>",
        reply_to="Support <support@example.com>",
    )

    report = EmlSOCAnalyzer().analyze(str(path))

    assert report["reply_to_mismatch"] is False
    assert report["reply_to_mismatch_legitimate"] is True
    assert "HIGH" not in _flag_levels(report, "Reply-To")
    assert "INFO" in _flag_levels(report, "Reply-To")


def test_reply_to_bulk_no_reply_to_support_is_info_not_high(tmp_path):
    path = _write_eml(
        tmp_path,
        from_value="CRM <no-reply@mail.crm-platform.test>",
        reply_to="Support <support@example.com>",
        headers={"List-Unsubscribe": "<mailto:unsubscribe@example.com>"},
    )

    report = EmlSOCAnalyzer().analyze(str(path))

    assert report["reply_to_mismatch"] is False
    assert report["reply_to_mismatch_legitimate"] is True
    assert "HIGH" not in _flag_levels(report, "Reply-To")


def test_reply_to_unrelated_personal_address_stays_high(tmp_path):
    path = _write_eml(
        tmp_path,
        from_value="Billing <billing@example.com>",
        reply_to="Private <collect.credentials@unrelated.test>",
    )

    report = EmlSOCAnalyzer().analyze(str(path))

    assert report["reply_to_mismatch"] is True
    assert report["reply_to_mismatch_legitimate"] is False
    assert "HIGH" in _flag_levels(report, "Reply-To")
