from src.views.analyzer import (
    _attachment_anomaly_without_pdf_risk,
    _pdf_indicator_lines,
    _pdf_status_text,
)


def test_pdf_risk_is_not_repeated_as_generic_attachment_anomaly():
    anomaly = "PDF risk CRITICAL: external URI action x20; Content-Type mismatch"

    assert _attachment_anomaly_without_pdf_risk(anomaly) == "Content-Type mismatch"


def test_pdf_status_and_indicator_lines_are_compact():
    pdf_security = {
        "risk_level": "critical",
        "summary": "external URI action x20; compressed object/xref stream x2",
        "indicators": [
            {"label": "external URI action", "count": 20},
            {"label": "compressed object/xref stream", "count": 2},
        ],
    }

    assert _pdf_status_text(pdf_security) == "PDF risk: CRITICAL"
    assert _pdf_indicator_lines(pdf_security) == [
        "external URI action x20",
        "compressed object/xref stream x2",
    ]
