from src.analyzer.llm_context_analyzer import format_email_risk_analysis
from src.views.analyzer import _show_phi4_result


class _ResultTarget:
    def __init__(self):
        self.kind = ""
        self.message = ""

    def _capture(self, kind, message):
        self.kind = kind
        self.message = message

    def error(self, message):
        self._capture("error", message)

    def warning(self, message):
        self._capture("warning", message)

    def success(self, message):
        self._capture("success", message)

    def info(self, message):
        self._capture("info", message)


def test_phi4_result_shows_only_the_fluent_explanation():
    result = {
        "final_verdict": "review",
        "content_summary": (
            "The subject and body request operational information and contain "
            "identity claims that require verification"
        ),
        "corroboration": {
            "supports_decision": False,
            "details": [],
            "caveats": [],
        },
        "intent_evidence": (
            "Me puedes verificar las fechas de salida de los equipos pendiente"
        ),
        "intent_signals": ["impersonation", "threat"],
        "signal_evidence": "Our company will be closed for Christmas holidays",
        "claimed_brand": "Medical Equipment",
    }
    target = _ResultTarget()

    _show_phi4_result(target, result)

    assert target.kind == "warning"
    assert target.message == format_email_risk_analysis(result)
    for field_label in (
        "Intent evidence:",
        "Context signals:",
        "Context evidence:",
        "Claimed identity:",
    ):
        assert field_label not in target.message
