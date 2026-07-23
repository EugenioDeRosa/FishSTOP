from src.ai_input import compact_ai_body


def test_compact_ai_body_replaces_noise_with_informative_placeholders():
    body = (
        "Apri https://example.test/reset?id=123 e scrivi a mario@example.test.\n"
        "Chiama +39 333 123 4567 oppure usa IP 192.0.2.10."
    )

    assert compact_ai_body(body) == (
        "Apri [URL LINK] e scrivi a [EMAIL ADDRESS].\n"
        "Chiama [PHONE NUMBER] oppure usa IP [IP ADDRESS]."
    )


def test_compact_ai_body_marks_hidden_html_link_without_losing_label():
    body = '<html><body><p>Apri <a href="https://evil.test/login">il portale</a>.</p></body></html>'

    compact = compact_ai_body(body)

    assert "Apri il portale [URL LINK]." in compact
    assert "evil.test" not in compact


def test_compact_ai_body_adds_marker_for_extracted_html_only_link():
    assert compact_ai_body("Apri il portale.", has_extracted_links=True) == (
        "Apri il portale.\n[URL LINK]"
    )


def test_compact_ai_body_collapses_long_high_entropy_poison_token():
    poison = ("9GACUTzE1LuLzYpPS2lNicIZM2MvWdoNd5CdZ3BhxoLmREBYpXd238Lxf0MbWr2K" * 5)
    body = f"{poison}@potQHLe8t7kbPiHLd89TwvziBnRGpnlULP0YC7gp1haKA\n\nClaim your bonus."

    compact = compact_ai_body(body)

    assert compact == "[OBFUSCATED DATA]\n\nClaim your bonus."
    assert poison not in compact
