from src.analyzer.body_context import select_body_for_ai


def test_reply_body_for_ai_removes_signature_disclaimer_and_thread_history():
    body = """Gentile Chiara,

torno a contattarla un'ultima volta per assicurarmi che i miei precedenti messaggi non si siano persi.
Nel confronto che Le propongo avrei piacere di condividere come altri clienti stanno trasformando il servizio post-vendita.

Se questi temi sono rilevanti, resto disponibile per un confronto in uno slot comodo.

Cordiali saluti,

Filippo Zerbini
Sales Development Executive
T +34697160047<https://or5.mailsap.com/api/mailings/click/example>
filippo.zerbini@example.com<https://or5.mailsap.com/api/mailings/click/example>

Please consider the impact on the environment before printing this e-mail.
This e-mail may contain trade secrets or privileged information.
If you no longer wish to receive communications, you can unsubscribe here.

On Fri, Jun 26, 2026 at 1:00 am, Filippo Zerbini wrote:

Gentile Chiara,
vecchio messaggio da non analizzare.
"""

    result = select_body_for_ai(body)

    assert result["body_context"] == "reply"
    assert "torno a contattarla" in result["body_ai"]
    assert "Cordiali saluti" not in result["body_ai"]
    assert "Sales Development Executive" not in result["body_ai"]
    assert "unsubscribe" not in result["body_ai"].lower()
    assert "vecchio messaggio" not in result["body_ai"]
    assert result["body_ai_removed_tail_lines"] > 0


def test_normal_body_for_ai_removes_legal_footer_but_keeps_message():
    body = """Ciao,

puoi confermarmi lo slot per domani?

This e-mail may contain confidential information.
Please delete it if received in error.
"""

    result = select_body_for_ai(body)

    assert result["body_context"] == "normal"
    assert result["body_ai"] == "Ciao,\n\npuoi confermarmi lo slot per domani?"
    assert result["body_ai_removed_tail_lines"] == 2


def test_underscore_separator_removes_previous_thread_and_good_luck_signature():
    body = """Dear Eugenio,

Please reply within a day if you agree.

Good luck,
Journal Operations Executive
______________________________________________
Monday, May 30, 2026, at 10:45:18 AM
To: recipient@example.com
Subject: Previous invitation

Old message with repeated pressure and deadlines.
"""

    result = select_body_for_ai(body)

    assert result["body_context"] == "reply"
    assert result["body_ai"] == "Dear Eugenio,\n\nPlease reply within a day if you agree."
    assert "Old message" not in result["body_ai"]
    assert "Journal Operations Executive" not in result["body_ai"]
    assert result["body_ai_removed_header_lines"] > 0
    assert result["body_ai_removed_tail_lines"] > 0
