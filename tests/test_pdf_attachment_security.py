from src.analyzer.attachment import analyze_attachment, analyze_pdf_security


def test_pdf_static_scan_flags_javascript_open_action():
    raw = (
        b"%PDF-1.7\n"
        b"1 0 obj << /Type /Catalog /OpenAction 2 0 R /Names << /JavaScript 3 0 R >> >> endobj\n"
        b"2 0 obj << /S /JavaScript /JS (app.alert('x')) >> endobj\n"
        b"%%EOF"
    )

    result = analyze_pdf_security(raw)

    assert result["is_pdf"] is True
    assert result["suspicious"] is True
    assert result["risk_level"] in {"high", "critical"}
    assert "embedded JavaScript" in result["summary"]
    assert "automatic action on document open" in result["summary"]


def test_attachment_pdf_security_is_added_to_anomaly():
    raw = b"%PDF-1.7\n1 0 obj << /Launch << /F (cmd.exe) >> >> endobj\n%%EOF"

    result = analyze_attachment("invoice.pdf", "application/pdf", "base64", raw)

    assert result["magic_detected_format"] == "pdf"
    assert result["pdf_security"]["suspicious"] is True
    assert result["pdf_security"]["risk_level"] == "critical"
    assert "PDF risk" in result["anomaly"]


def test_clean_pdf_reports_no_active_features():
    raw = b"%PDF-1.4\n1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n%%EOF"

    result = analyze_attachment("report.pdf", "application/pdf", "base64", raw)

    assert result["pdf_security"]["risk_level"] == "clean"
    assert result["pdf_security"]["suspicious"] is False
    assert result["anomaly"] is None


def test_pdf_static_scan_decodes_obfuscated_names():
    raw = b"%PDF-1.7\n1 0 obj << /J#61vaScript 2 0 R /Open#41ction 3 0 R >> endobj\n%%EOF"

    result = analyze_pdf_security(raw)

    assert result["suspicious"] is True
    assert "embedded JavaScript" in result["summary"]
    assert "automatic action on document open" in result["summary"]
    assert result["suspicious_name_escapes"] >= 2


def test_pdf_encrypted_without_active_content_is_medium_not_suspicious():
    raw = b"%PDF-1.7\n1 0 obj << /Type /Catalog >> endobj\n2 0 obj << /Encrypt 3 0 R >> endobj\n%%EOF"

    result = analyze_pdf_security(raw)

    assert result["risk_level"] == "medium"
    assert result["suspicious"] is False
    assert "encrypted PDF" in result["summary"]
