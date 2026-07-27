from pathlib import Path
import threading
import time

import torch

from src.background_jobs import JobSnapshot
from src.views import analyzer as analyzer_view


class _RecordingAnalyzer:
    def __init__(self):
        self.paths: list[Path] = []

    def analyze(self, eml_path: str) -> dict:
        path = Path(eml_path)
        assert path.exists()
        self.paths.append(path)
        return {"subject": "safe"}


def test_clear_email_state_preserves_client_settings(monkeypatch):
    state = {
        "fishstop_eml_uploader": object(),
        "raw_eml_text": "raw email",
        "current_eml_name": "message.eml",
        "current_eml_hash": "digest",
        "soc_analysis_digest": {"subject": "cached"},
        "phi4_analysis_digest_result": {"final_verdict": "review"},
        "background_lookup_signature_digest": ("done",),
        "fishstop_user_api_keys": {"GITHUB_MODELS_TOKEN": "secret"},
        "page": "analyze",
    }
    monkeypatch.setattr(analyzer_view.st, "session_state", state)

    analyzer_view._clear_email_analysis_state()

    assert state == {
        "fishstop_user_api_keys": {"GITHUB_MODELS_TOKEN": "secret"},
        "page": "analyze",
    }


def test_clear_email_cancels_old_jobs_and_rotates_analysis_session(monkeypatch):
    class Jobs:
        cancelled_session = ""

        def cancel_session(self, session_id):
            self.cancelled_session = session_id
            return 1

    jobs = Jobs()
    state = {
        "fishstop_analysis_session_id": "client-a",
        "current_eml_hash": "digest",
        "page": "analyze",
    }
    monkeypatch.setattr(analyzer_view.st, "session_state", state)
    monkeypatch.setattr(
        analyzer_view,
        "_get_background_job_manager",
        lambda: jobs,
    )

    analyzer_view._clear_email_analysis_state()

    assert jobs.cancelled_session == "client-a"
    assert state == {"page": "analyze"}


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


def test_background_job_keys_are_isolated_between_client_sessions(monkeypatch):
    jobs = _RecordingJobs()
    monkeypatch.setattr(
        analyzer_view,
        "get_secret",
        lambda name, default="": "token" if "API_KEY" in name else default,
    )
    soc = {
        "from_": "Sender <sender@example.test>",
        "links": [{"url": "https://example.test/action"}],
        "received_hops": [],
        "attachments": [],
    }

    analyzer_view._schedule_background_lookups(
        jobs, _BackgroundValidator(), soc, "client-a"
    )
    analyzer_view._schedule_background_lookups(
        jobs, _BackgroundValidator(), soc, "client-b"
    )

    url_keys = [
        key
        for pool, key, *_rest in jobs.submissions
        if pool == "virustotal" and key[0] == "url"
    ]
    assert url_keys[0][1] == "client-a"
    assert url_keys[1][1] == "client-b"
    assert url_keys[0] != url_keys[1]


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


def test_production_bert_uses_server_token_not_session_token(monkeypatch):
    jobs = _RecordingJobs()
    monkeypatch.setattr(analyzer_view, "is_production_mode", lambda: True)
    monkeypatch.setattr(
        analyzer_view,
        "get_secret",
        lambda name, default="": "client-hf",
    )
    monkeypatch.setattr(
        analyzer_view,
        "get_server_secret",
        lambda name, default="": "server-hf",
    )

    analyzer_view._schedule_bert_background(
        jobs,
        "Body text",
        "email-digest",
        "client-a",
    )

    _pool, key, _function, args, _kwargs = jobs.submissions[0]
    assert key[1] == "client-a"
    assert args == ("Body text", "server-hf")


def test_shared_bert_resource_serializes_concurrent_inference(monkeypatch):
    class _Config:
        fishstop_dataset_sha256 = ""

    class _Model:
        config = _Config()

    active = 0
    maximum_active = 0
    counter_lock = threading.Lock()

    def fake_predict(*_args, **_kwargs):
        nonlocal active, maximum_active
        with counter_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.03)
        with counter_lock:
            active -= 1
        return torch.tensor([[1.0, 2.0]]), 1

    monkeypatch.setattr(
        analyzer_view,
        "init_content_model",
        lambda _token: (object(), _Model(), "test"),
    )
    monkeypatch.setattr(
        analyzer_view,
        "init_calibration",
        lambda _token: {
            "temperature": 1.0,
            "threshold": 0.5,
            "band": 0.3,
            "positive_label_id": 1,
            "dataset_sha256": "",
        },
    )
    monkeypatch.setattr(analyzer_view, "predict_email_logits", fake_predict)

    first = threading.Thread(
        target=analyzer_view._run_bert_background,
        args=("first", ""),
    )
    second = threading.Thread(
        target=analyzer_view._run_bert_background,
        args=("second", ""),
    )
    first.start()
    second.start()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert maximum_active == 1


def test_hosted_phi4_is_scheduled_without_per_email_consent(monkeypatch):
    jobs = _RecordingJobs()
    monkeypatch.setattr(
        analyzer_view,
        "active_llm_backend",
        lambda: "github models (Phi-4-mini-instruct)",
    )
    monkeypatch.setattr(
        analyzer_view,
        "get_secret",
        lambda name, default="": (
            "session-github" if name == "GITHUB_MODELS_TOKEN" else default
        ),
    )

    reference = analyzer_view._schedule_phi4_background(
        jobs,
        {"subject": "Test", "body_for_ai": "Analyze this email."},
        "analysis-key",
    )

    assert reference is not None
    assert reference[0] == "llm"
    assert len(jobs.submissions) == 1
    pool, key, function, args, kwargs = jobs.submissions[0]
    assert pool == "llm"
    assert key[0] == "phi4"
    assert function is analyzer_view._run_phi4_background
    assert args[1] == "session-github"
    assert callable(args[2])
    assert callable(args[3])
    assert kwargs == {}


def test_phi4_worker_publishes_section_progress():
    progress = []

    def fake_stream(*_args, **_kwargs):
        yield {
            "status": "progress",
            "stage": "content",
            "current": 2,
            "total": 4,
        }
        yield {
            "status": "ok",
            "analysis": {"final_verdict": "review"},
        }

    original = analyzer_view.stream_phi4_email_analysis
    analyzer_view.stream_phi4_email_analysis = fake_stream
    try:
        result = analyzer_view._run_phi4_background(
            {},
            progress_callback=progress.append,
            cancellation_requested=lambda: False,
        )
    finally:
        analyzer_view.stream_phi4_email_analysis = original

    assert progress == [{
        "status": "progress",
        "stage": "content",
        "current": 2,
        "total": 4,
    }]
    assert result["status"] == "ok"


def test_phi4_worker_stops_when_cancellation_is_requested():
    consumed = []

    def fake_stream(*_args, **_kwargs):
        consumed.append("first")
        yield {"status": "stream", "delta": "token"}
        consumed.append("second")
        yield {"status": "ok", "analysis": {}}

    original = analyzer_view.stream_phi4_email_analysis
    analyzer_view.stream_phi4_email_analysis = fake_stream
    checks = iter([False, True])
    try:
        result = analyzer_view._run_phi4_background(
            {},
            cancellation_requested=lambda: next(checks),
        )
    finally:
        analyzer_view.stream_phi4_email_analysis = original

    assert result == {"status": "cancelled"}
    assert consumed == ["first"]


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


class _SnapshotJobs:
    def __init__(self, snapshot):
        self._snapshot = snapshot

    def snapshot(self, pool, key):
        assert pool == "llm"
        assert key == ("phi4", "result")
        return self._snapshot


def test_completed_phi4_result_is_available_during_same_app_render():
    expected = {"final_verdict": "review", "content_summary": "Review needed"}
    result, error, state = analyzer_view._phi4_background_outcome(
        _SnapshotJobs(JobSnapshot(
            "done",
            result={"status": "ok", "analysis": expected},
        )),
        ("llm", ("phi4", "result")),
    )

    assert result is expected
    assert error == ""
    assert state == "done"


def test_invalid_completed_phi4_result_becomes_an_error_without_rerun():
    result, error, state = analyzer_view._phi4_background_outcome(
        _SnapshotJobs(JobSnapshot("done", result={"status": "ok"})),
        ("llm", ("phi4", "result")),
    )

    assert result is None
    assert error == "Structured Phi-4 analysis is missing"
    assert state == "done"


def test_running_phi4_snapshot_remains_non_blocking():
    result, error, state = analyzer_view._phi4_background_outcome(
        _SnapshotJobs(JobSnapshot("running")),
        ("llm", ("phi4", "result")),
    )

    assert result is None
    assert error == ""
    assert state == "running"


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
