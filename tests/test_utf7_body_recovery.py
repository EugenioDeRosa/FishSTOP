from src.analyzer.html_utils import recover_mislabelled_utf7_html
from src.analyzer.soc_analyzer import EmlSOCAnalyzer
from src.parser import EmailParserPipeline


UTF7_HTML = (
    "+ADw-html+AD4APA-body+AD4APA-p+AD4-Claim your account reward now.+ADw-/p+AD4-"
    "+ADw-a href+AD0AIg-https://evil.example/claim+ACIAPg-Open claim dashboard+ADw-/a+AD4-"
    "+ADw-/body+AD4APA-/html+AD4-"
)


def _write_mislabelled_eml(tmp_path):
    eml = (
        "From: sender@example.com\r\n"
        "To: recipient@example.net\r\n"
        "Subject: Account reward\r\n"
        "MIME-Version: 1.0\r\n"
        "Content-Type: text/html; charset=utf-8\r\n"
        "Content-Transfer-Encoding: 7bit\r\n"
        "\r\n"
        f"{UTF7_HTML}\r\n"
    )
    path = tmp_path / "mislabelled-utf7.eml"
    path.write_bytes(eml.encode("ascii"))
    return path


def test_utf7_recovery_requires_html_structure():
    recovered = recover_mislabelled_utf7_html(UTF7_HTML)

    assert recovered.startswith("<html><body>")
    assert '<a href="https://evil.example/claim">' in recovered
    assert recover_mislabelled_utf7_html("Reference +ADw- only") == "Reference +ADw- only"


def test_soc_analyzer_recovers_visible_body_and_links(tmp_path):
    report = EmlSOCAnalyzer().analyze(str(_write_mislabelled_eml(tmp_path)))

    assert "Claim your account reward now." in report["body_for_ai"]
    assert "Open claim dashboard" in report["body_for_ai"]
    assert "+ADw-" not in report["body_for_ai"]
    assert any(link["host"] == "evil.example" for link in report["links"])


def test_general_parser_recovers_mislabelled_utf7_html(tmp_path):
    parsed = EmailParserPipeline().parse_single_eml(str(_write_mislabelled_eml(tmp_path)))

    assert parsed["body"].startswith("<html><body>")
    assert "https://evil.example/claim" in parsed["body"]
    assert "+ADw-" not in parsed["body"]
