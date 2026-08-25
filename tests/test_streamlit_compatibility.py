from pathlib import Path

from src.views.analyzer import _report_table_rows


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_structured_report_values_have_one_arrow_compatible_type():
    rows = _report_table_rows(
        [
            ("Sender", "alice@example.test"),
            ("Mismatch", True),
            ("Score", 42),
            ("Metadata", {"source": "header"}),
        ]
    )

    assert rows == [
        {"Field": "Sender", "Value": "alice@example.test"},
        {"Field": "Mismatch", "Value": "True"},
        {"Field": "Score", "Value": "42"},
        {"Field": "Metadata", "Value": '{"source": "header"}'},
    ]
    assert {type(row["Value"]) for row in rows} == {str}


def test_runtime_code_no_longer_uses_deprecated_components_html():
    runtime_files = [
        PROJECT_ROOT / "src" / "views" / "analyzer.py",
        PROJECT_ROOT / "src" / "components" / "email_globe.py",
    ]

    for path in runtime_files:
        source = path.read_text(encoding="utf-8")
        assert "streamlit.components.v1" not in source
        assert "components.html(" not in source
