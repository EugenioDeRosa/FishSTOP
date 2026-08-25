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
    assert "score" not in result
    assert any(item["key"] == "javascript" for item in result["behaviors"])
    assert any(item["key"] == "open_action" for item in result["behaviors"])


def test_attachment_pdf_security_is_added_to_anomaly():
    raw = b"%PDF-1.7\n1 0 obj << /Launch << /F (cmd.exe) >> >> endobj\n%%EOF"

    result = analyze_attachment("invoice.pdf", "application/pdf", "base64", raw)

    assert result["magic_detected_format"] == "pdf"
    assert result["pdf_security"]["suspicious"] is True
    assert result["pdf_security"]["risk_level"] == "critical"
    assert any(item["key"] == "launch_action" for item in result["pdf_security"]["behaviors"])
    assert "PDF risk" in result["anomaly"]


def test_clean_pdf_reports_no_active_features():
    raw = b"%PDF-1.4\n1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n%%EOF"

    result = analyze_attachment("report.pdf", "application/pdf", "base64", raw)

    assert result["pdf_security"]["risk_level"] == "clean"
    assert result["pdf_security"]["suspicious"] is False
    assert result["pdf_security"]["behaviors"] == []
    assert "score" not in result["pdf_security"]
    assert result["anomaly"] is None


def test_pdf_static_scan_decodes_obfuscated_names():
    raw = b"%PDF-1.7\n1 0 obj << /J#61vaScript 2 0 R /Open#41ction 3 0 R >> endobj\n%%EOF"

    result = analyze_pdf_security(raw)

    assert result["suspicious"] is True
    assert any(item["key"] == "javascript" for item in result["behaviors"])
    assert "embedded JavaScript" in result["summary"]
    assert "automatic action on document open" in result["summary"]
    assert result["suspicious_name_escapes"] >= 2


def test_pdf_encrypted_without_active_content_is_medium_not_suspicious():
    raw = b"%PDF-1.7\n1 0 obj << /Type /Catalog >> endobj\n2 0 obj << /Encrypt 3 0 R >> endobj\n%%EOF"

    result = analyze_pdf_security(raw)

    assert result["risk_level"] == "medium"
    assert result["suspicious"] is False
    assert "encrypted PDF" in result["summary"]


def test_pdf_uri_and_structural_noise_are_not_malicious_by_themselves():
    raw = (
        b"%PDF-1.7\n"
        b"1 0 obj << /Type /Catalog /URI (https://example.com) /ObjStm 2 0 R /XRefStm 3 0 R >> endobj\n"
        b"2 0 obj << /URI (https://example.org) /AcroForm 4 0 R >> endobj\n"
        b"3 0 obj << /URI (mailto:test@example.com) >> endobj\n"
        b"/N#61me /An#6ftherName\n"
        b"%%EOF\n%%EOF"
    )

    result = analyze_pdf_security(raw)

    assert result["suspicious"] is False
    assert result["risk_level"] in {"low", "medium"}
    assert "external URI action" in result["summary"]
    assert "compressed object/xref stream" in result["summary"]
    assert result["behaviors"] == []


def test_pdf_open_action_without_active_payload_is_context_only():
    raw = b"%PDF-1.7\n1 0 obj << /Type /Catalog /OpenAction 2 0 R >> endobj\n2 0 obj << /S /GoTo /D [3 0 R /Fit] >> endobj\n%%EOF"

    result = analyze_pdf_security(raw)

    assert result["suspicious"] is False
    assert result["risk_level"] in {"low", "medium"}
    assert "automatic action on document open" in result["summary"]


def test_pdf_uri_action_with_tracked_nested_redirect_is_malicious_behavior():
    raw = (
        b"%PDF-1.7\n"
        b"40 0 obj << /Type /Action /S /URI "
        b"/URI (https://stat.chatify.dev/analytics/v1/count?"
        b"__url=https%3A%2F%2Fsites.google.com%2Fview%2F10931222245678%2Fhome%2F"
        b"&__analytics-id=abc&__uid=5155) >> endobj\n"
        b"42 0 obj << /Type /Action /S /URI "
        b"/URI (https://sites.google.com/view/1099812345/home) >> endobj\n"
        b"%%EOF"
    )

    result = analyze_pdf_security(raw)

    assert result["suspicious"] is True
    assert result["risk_level"] == "high"
    assert any(item["key"] == "uri_nested_redirect" for item in result["behaviors"])
    assert any(item["key"] == "uri_tracked_redirect" for item in result["behaviors"])
    assert any(item["key"] == "public_site_landing" for item in result["behaviors"])
    assert result["uri_evidence"]["nested_redirect_count"] == 1
    assert result["uri_evidence"]["tracked_redirect_count"] == 1
    assert result["uri_evidence"]["public_site_landing_count"] == 2
    assert "object 40" in result["summary"]
    assert "stat.chatify.dev" in result["summary"]
    assert "sites.google.com" in result["summary"]


def test_pdf_plain_uri_action_remains_context_only():
    raw = (
        b"%PDF-1.7\n"
        b"10 0 obj << /Type /Action /S /URI /URI (https://example.com/invoice-info) >> endobj\n"
        b"%%EOF"
    )

    result = analyze_pdf_security(raw)

    assert result["suspicious"] is False
    assert result["risk_level"] in {"low", "medium"}
    assert result["behaviors"] == []
    assert result["uri_evidence"]["nested_redirect_count"] == 0
    assert result["uri_evidence"]["tracked_redirect_count"] == 0
    assert result["uri_evidence"]["public_site_landing_count"] == 0





def test_pdf_structural_scan_analyzes_uri_values_hidden_from_raw_text(monkeypatch):
    class FakeReader:
        is_encrypted = False
        pages = [object()]
        attachments = {}
        open_destination = None
        page_mode = None
        root_object = {
            "/Type": "/Catalog",
            "/A": {
                "/Type": "/Action",
                "/S": "/URI",
                "/URI": "https://stat.chatify.dev/analytics/v1/count?__url=https%3A%2F%2Fsites.google.com%2Fview%2F10931222245678%2Fhome%2F&__uid=5155",
            },
        }

        def __init__(self, *args, **kwargs):
            pass

        def get_fields(self):
            return {}

    import pypdf

    monkeypatch.setattr(pypdf, "PdfReader", FakeReader)
    raw = b"%PDF-1.7\n1 0 obj << /Type /ObjStm /Filter /FlateDecode >> stream x\x9c\x03\x00 endstream\n%%EOF"

    result = analyze_pdf_security(raw)

    assert result["suspicious"] is True
    assert any(item["key"] == "uri_nested_redirect" for item in result["behaviors"])
    assert any(item["key"] == "uri_tracked_redirect" for item in result["behaviors"])
    assert any(item["key"] == "public_site_landing" for item in result["behaviors"])
    assert result["uri_evidence"]["nested_redirect_count"] == 1
    assert result["uri_evidence"]["tracked_redirect_count"] == 1
    assert result["uri_evidence"]["public_site_landing_count"] == 1
