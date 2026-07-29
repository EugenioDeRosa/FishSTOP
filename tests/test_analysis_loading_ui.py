from pathlib import Path

from src.views.analyzer import (
    _background_jobs_complete,
    _render_url_intelligence_card,
    _verdict_loading_html,
)


def test_unified_loading_splash_waits_for_the_final_verdict():
    html = _verdict_loading_html()

    assert 'role="status"' in html
    assert 'aria-live="polite"' in html
    assert "FishSTOP" in html
    assert "Building the final verdict" in html
    assert "DistilBERT" in html
    assert "Phi-4 mini" in html
    assert "fs-analysis-splash__skeleton" in html


def test_unified_loading_finishes_only_when_every_job_is_terminal():
    assert _background_jobs_complete(("done", "error", "overloaded"))
    assert _background_jobs_complete(("done", "cancelled"))
    assert not _background_jobs_complete(("done", "running"))
    assert not _background_jobs_complete(("queued", "done"))


def test_individual_progress_messages_are_not_rendered_anymore():
    analyzer_source = (
        Path(__file__).resolve().parents[1] / "src" / "views" / "analyzer.py"
    ).read_text(encoding="utf-8")

    assert "All analysis jobs completed:" not in analyzer_source
    assert "Results appear independently as soon as they are ready:" not in analyzer_source
    assert "_bert_loading_html" not in analyzer_source
    assert "_phi4_loading_html" not in analyzer_source


def test_unknown_url_can_be_copied_and_opened_in_virustotal_web_ui():
    analyzer_source = (
        Path(__file__).resolve().parents[1] / "src" / "views" / "analyzer.py"
    ).read_text(encoding="utf-8")

    assert "Copy URL &amp; scan on VirusTotal" in analyzer_source
    assert 'href="https://www.virustotal.com/gui/home/url"' in analyzer_source
    assert "_render_url_intelligence_card" in analyzer_source
    assert "_safe_vt_url_lookup" in analyzer_source


def test_existing_report_card_is_compact_and_has_no_copy_action(monkeypatch):
    rendered = {}
    monkeypatch.setattr(
        "src.views.analyzer.st.iframe",
        lambda html, **kwargs: rendered.update(html=html, **kwargs),
    )

    _render_url_intelligence_card(
        {"url": "https://bad.example/path", "host": "bad.example"},
        {
            "status": "malicious",
            "detection_ratio": "2 / 92",
            "last_analysis": "2026-07-29 20:00 UTC",
            "permalink": "https://www.virustotal.com/gui/url/report",
        },
        1,
    )

    assert "MALICIOUS" in rendered["html"]
    assert "2 / 92 engines" in rendered["html"]
    assert "VirusTotal URL report" in rendered["html"]
    assert "Copy URL &amp; scan" not in rendered["html"]
    assert "Malicious 1" not in rendered["html"]
    assert "Harmless" not in rendered["html"]
