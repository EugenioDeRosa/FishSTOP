from pathlib import Path

from src.views import analyzer as analyzer_view


class _RecordingAnalyzer:
    def __init__(self):
        self.paths: list[Path] = []

    def analyze(self, eml_path: str) -> dict:
        path = Path(eml_path)
        assert path.exists()
        self.paths.append(path)
        return {"subject": "safe"}


def test_uploaded_email_uses_unique_deleted_temporary_file(monkeypatch):
    recording_analyzer = _RecordingAnalyzer()
    monkeypatch.setattr(
        analyzer_view,
        "get_core_backend",
        lambda: (object(), recording_analyzer),
    )

    first = analyzer_view._analyze_eml_bytes(b"Subject: first\r\n\r\nBody")
    second = analyzer_view._analyze_eml_bytes(b"Subject: second\r\n\r\nBody")

    assert first == {"subject": "safe"}
    assert second == {"subject": "safe"}
    assert recording_analyzer.paths[0] != recording_analyzer.paths[1]
    assert all(not path.exists() for path in recording_analyzer.paths)


def test_final_summary_severity_follows_all_phi4_policy_verdicts():
    counts = {"HIGH": 1, "MEDIUM": 2, "LOW": 0, "INFO": 0}

    assert analyzer_view._severity(
        counts,
        {"final_verdict": "phishing"},
    ) == ("CRITICAL", "Final combined verdict: phishing")
    assert analyzer_view._severity(
        counts,
        {"final_verdict": "review"},
    ) == ("SUSPICIOUS", "Final combined verdict: manual review required")
    assert analyzer_view._severity(
        counts,
        {"final_verdict": "legitimate"},
    ) == ("LOW", "Final combined verdict: likely legitimate")


def test_static_severity_is_used_until_phi4_policy_finishes():
    counts = {"HIGH": 0, "MEDIUM": 2, "LOW": 0, "INFO": 0}

    assert analyzer_view._severity(counts) == (
        "SUSPICIOUS",
        "Indicators require manual validation",
    )


class _RecordingJobs:
    def __init__(self):
        self.submissions = []

    def get_or_submit(self, pool, key, function, *args, **kwargs):
        self.submissions.append((pool, key, function, args, kwargs))


class _BackgroundValidator:
    def check_url_reputation(self, url, api_key=None):
        return {"status": "clean"}

    def check_domain_reputation(self, domain, api_key=None):
        return {"status": "ok"}

    def check_ip_reputation(self, ip, api_key=None):
        return {"status": "ok"}

    def geolocate_ip(self, ip):
        return {"status": "ok"}

    def check_file_hash(self, sha256, api_key=None):
        return {"status": "clean"}


def test_background_lookup_plan_captures_session_api_keys(monkeypatch):
    jobs = _RecordingJobs()
    monkeypatch.setattr(
        analyzer_view,
        "get_secret",
        lambda name, default="": {
            "VIRUSTOTAL_API_KEY": "session-vt",
            "ABUSEIPDB_API_KEY": "session-abuse",
        }.get(name, default),
    )

    plan = analyzer_view._schedule_background_lookups(
        jobs,
        _BackgroundValidator(),
        {
            "from_": "Sender <sender@example.test>",
            "links": [{"url": "https://example.test/action"}],
            "received_hops": [{
                "sender_ip": "8.8.8.8",
                "all_ips": ["8.8.8.8"],
            }],
            "attachments": [{"hash_sha256": "abc123"}],
        },
    )

    assert set(plan) == {
        "urls", "domains", "ip_reputation", "geolocation", "files",
    }
    submitted_kwargs = [submission[4] for submission in jobs.submissions]
    assert any(kwargs.get("api_key") == "session-vt" for kwargs in submitted_kwargs)
    assert any(
        kwargs.get("api_key") == "session-abuse"
        for kwargs in submitted_kwargs
    )


def test_bert_is_scheduled_on_its_dedicated_background_pool(monkeypatch):
    jobs = _RecordingJobs()
    monkeypatch.setattr(
        analyzer_view,
        "get_secret",
        lambda name, default="": (
            "session-hf" if name == "HF_TOKEN" else default
        ),
    )

    reference = analyzer_view._schedule_bert_background(
        jobs,
        "Subject: test\nBody text",
        "email-digest",
    )

    assert reference is not None
    assert reference[0] == "bert"
    assert len(jobs.submissions) == 1
    pool, key, function, args, kwargs = jobs.submissions[0]
    assert pool == "bert"
    assert key[0] == "bert"
    assert function is analyzer_view._run_bert_background
    assert args == (
        "Subject: test\nBody text",
        "session-hf",
    )
    assert kwargs == {}


def test_completed_bert_result_is_applied_without_model_rerun():
    soc = {}
    analyzer_view._apply_bert_result_to_soc(
        soc,
        {
            "classification": "phishing",
            "probability_malicious": 82.5,
            "probability_legitimate": 17.5,
            "chunk_count": 3,
            "calibration": {"source": "huggingface"},
        },
    )

    assert soc["bert_ai_result"] == "phishing"
    assert soc["bert_malicious_probability"] == 82.5
    assert soc["bert_legitimate_probability"] == 17.5
    assert soc["bert_chunk_count"] == 3
    assert soc["bert_probability_calibrated"] is True


def test_first_fragment_poll_refreshes_values_completed_after_render():
    assert analyzer_view._background_refresh_required(
        rendered_states=("running", "running"),
        current_states=("done", "running"),
        previous_states=None,
    ) is True


def test_fragment_does_not_rerun_when_rendered_values_are_current():
    assert analyzer_view._background_refresh_required(
        rendered_states=("done", "running"),
        current_states=("done", "running"),
        previous_states=("done", "running"),
    ) is False
