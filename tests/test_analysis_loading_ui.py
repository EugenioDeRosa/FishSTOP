from src.views.analyzer import _bert_loading_html, _phi4_loading_html


def test_bert_loading_splash_identifies_the_pending_result():
    html = _bert_loading_html()

    assert 'role="status"' in html
    assert 'aria-live="polite"' in html
    assert "DistilBERT" in html
    assert "Content classification in progress" in html
    assert "fs-analysis-splash__skeleton" in html


def test_phi4_loading_splash_identifies_the_pending_result():
    html = _phi4_loading_html()

    assert "Phi-4 mini" in html
    assert "Semantic analysis in progress" in html
    assert "Loading Phi-4 mini result" in html


def test_loading_splash_escapes_progress_detail():
    html = _phi4_loading_html("<script>alert('x')</script>")

    assert "<script>" not in html
    assert "&lt;script&gt;" in html
