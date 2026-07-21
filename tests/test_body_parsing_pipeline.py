from email.message import EmailMessage

from src.analyzer.html_utils import strip_html
from src.analyzer.soc_analyzer import EmlSOCAnalyzer


def _write_message(tmp_path, message: EmailMessage, name: str = "message.eml"):
    path = tmp_path / name
    path.write_bytes(message.as_bytes())
    return path


def test_html_stripping_preserves_inline_words_blocks_alt_and_removes_hidden_text():
    html = """
    <html><body>
      <p>Pa<span>y</span>Pal</p>
      <div>Hello<br>World</div>
      <span hidden>HIDDEN ATTRIBUTE</span>
      <div hidden><span>NESTED HIDDEN TEXT</span></div>
      <span aria-hidden="true">ARIA HIDDEN</span>
      <span style="display:none">CSS HIDDEN</span>
      <img alt="Urgent account verification">
    </body></html>
    """

    text = strip_html(html)

    assert "PayPal" in text
    assert "Hello\nWorld" in text
    assert "Urgent account verification" in text
    assert "HIDDEN ATTRIBUTE" not in text
    assert "NESTED HIDDEN TEXT" not in text
    assert "ARIA HIDDEN" not in text
    assert "CSS HIDDEN" not in text


def test_multipart_alternative_uses_one_canonical_body_and_does_not_restore_thread(tmp_path):
    message = EmailMessage()
    message["From"] = "sender@example.com"
    message["To"] = "recipient@example.com"
    message["Subject"] = "Current request"
    message.set_content(
        "Current message only.\n\n"
        "On Monday, Old Sender wrote:\n"
        "OLD QUOTED PLAIN MESSAGE"
    )
    message.add_alternative(
        "<html><body><p>Current message only.</p>"
        "<p>On Monday, Old Sender wrote:</p>"
        "<p>OLD QUOTED HTML MESSAGE</p></body></html>",
        subtype="html",
    )

    report = EmlSOCAnalyzer().analyze(str(_write_message(tmp_path, message)))

    assert report["body_source"] == "text/plain"
    assert report["body_for_ai"] == "Current message only."
    assert "OLD QUOTED" not in report["body_for_ai"]
    assert "HTML-derived visible text" not in report["body_for_ai"]


def test_attached_rfc822_body_is_not_merged_into_outer_body(tmp_path):
    attached = EmailMessage()
    attached["From"] = "attached@example.com"
    attached["To"] = "recipient@example.com"
    attached["Subject"] = "Attached message"
    attached.set_content("ATTACHED BODY MUST NOT ENTER BERT")

    outer = EmailMessage()
    outer["From"] = "sender@example.com"
    outer["To"] = "recipient@example.com"
    outer["Subject"] = "Outer message"
    outer.set_content("OUTER BODY FOR BERT")
    outer.add_attachment(attached)

    report = EmlSOCAnalyzer().analyze(str(_write_message(tmp_path, outer)))

    assert "OUTER BODY FOR BERT" in report["body_for_ai"]
    assert "ATTACHED BODY MUST NOT ENTER BERT" not in report["body_for_ai"]


def test_large_plaintext_poison_block_is_removed_but_real_content_and_isolated_token_remain(tmp_path):
    poison_lines = [
        ("Ab9xK2mP7qR4sT8vW3yZ6cD1fG5hJ0kL" * 8) + str(index)
        for index in range(20)
    ]
    isolated_token = "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8S9t0U1v2W3x4Y5z6ABCD"
    message = EmailMessage()
    message["From"] = "sender@example.com"
    message["To"] = "recipient@example.com"
    message["Subject"] = "Casino reward"
    message.set_content(
        "\n".join(poison_lines)
        + "\n\nLolajack 400% bonus and 400 free spins.\n"
        + isolated_token
    )
    message.add_alternative(
        '<html><body><div style="display:none">'
        + "\n".join(poison_lines)
        + "</div><p>Lolajack 400% bonus and 400 free spins.</p></body></html>",
        subtype="html",
    )

    report = EmlSOCAnalyzer().analyze(str(_write_message(tmp_path, message)))

    assert report["body_plain_noise_removed_lines"] == len(poison_lines)
    assert report["body_plain_noise_removed_chars"] > 4096
    assert poison_lines[0] not in report["body_for_ai"]
    assert "Lolajack 400% bonus and 400 free spins." in report["body_for_ai"]
    assert isolated_token in report["body_for_ai"]
