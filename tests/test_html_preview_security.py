from src.analyzer.html_utils import sanitize_html_for_js_preview, sanitize_html_for_preview


def test_js_preview_removes_email_javascript_and_active_content():
    raw = """
    <html><body>
      <script>alert('x')</script>
      <img src="https://tracker.example/pixel" onerror="alert(1)">
      <a href="javascript:alert(2)" onclick="steal()" ping="https://tracker.example/p">Pay invoice</a>
      <iframe srcdoc="<script>alert(3)</script>"></iframe>
      <form action="https://evil.example"><input name="pw"></form>
      <div style="background-image:url(javascript:alert(4))">Hello</div>
    </body></html>
    """

    safe = sanitize_html_for_js_preview(raw).lower()

    assert "alert(" not in safe
    assert "onerror" not in safe
    assert "onclick" not in safe
    assert "javascript:" not in safe
    assert "srcdoc" not in safe
    assert "<iframe" not in safe
    assert "<form" not in safe
    assert "<input" not in safe
    assert "https://tracker.example" not in safe
    assert "content-security-policy" in safe
    assert "script-src &#x27;nonce-fishstop-preview-guard&#x27;" in safe


def test_plain_preview_removes_extended_url_attributes():
    raw = '<a href="https://evil.example" ping="https://evil.example/p">Open</a><img srcset="https://evil.example/x 1x">'

    safe = sanitize_html_for_preview(raw).lower()

    assert "https://evil.example" not in safe
    assert "href=" not in safe
    assert "ping=" not in safe
    assert "srcset=" not in safe
