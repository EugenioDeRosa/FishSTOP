from src.analyzer.link_extractor import extract_links


def test_html_anchor_text_domain_mismatch_is_detected():
    html = '<a href="https://evil.example/login">https://login.microsoft.com</a>'

    links = extract_links("", html)

    href_link = next(link for link in links if link["source"] == "html_href")
    assert href_link["host"] == "evil.example"
    assert href_link["display_host"] == "login.microsoft.com"
    assert href_link["display_mismatch"] is True


def test_same_registered_domain_visible_text_is_not_mismatch():
    html = '<a href="https://accounts.google.com/login">https://google.com</a>'

    links = extract_links("", html)

    href_link = next(link for link in links if link["source"] == "html_href")
    assert href_link["display_host"] == "google.com"
    assert href_link["display_mismatch"] is False


def test_possible_shortener_is_generic_not_whitelist_based():
    links = extract_links("https://abc.de/A7xY9", "")

    assert links[0]["is_possible_shortener"] is True
    assert links[0]["shortener_reason"]


def test_bracketed_placeholder_userinfo_recovers_real_destination_without_crashing():
    html = '<a href="https://[an_21]@bit.ly/48Q8bWj#campaign">Claim reward</a>'

    links = extract_links("", html)

    assert len(links) == 1
    assert links[0]["url"] == "https://bit.ly/48Q8bWj#campaign"
    assert links[0]["host"] == "bit.ly"


def test_unrecoverable_bracketed_host_is_ignored_without_crashing():
    html = '<a href="https://[not-an-ip]/claim">Claim reward</a>'

    assert extract_links("", html) == []


def test_outlook_signature_link_is_retained_but_not_actionable():
    html = """
    <div>Routine message.</div>
    <div id="Signature">
      <a href="http://www.example.com/">www.example.com</a>
    </div>
    """

    links = extract_links("Routine message.\nwww.example.com<http://www.example.com/>", html)

    assert len(links) == 1
    assert links[0]["role"] == "signature"
    assert links[0]["actionable"] is False
