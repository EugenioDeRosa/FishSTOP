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
