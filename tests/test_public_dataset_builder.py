from pathlib import Path

import pandas as pd

from src.public_dataset_builder import (
    MAX_TEXT_CHARS,
    _assign_splits,
    _clean_dataset_frame,
    _dedupe_templates,
    _parse_binary_label,
    template_hash,
    text_hash,
)


def _text(name: str) -> str:
    return f"Subject {name} this is a sufficiently long email body for dataset quality testing."


def test_binary_labels_are_strict_and_source_values_are_supported():
    assert _parse_binary_label(0) == 0
    assert _parse_binary_label("Safe Email") == 0
    assert _parse_binary_label("Phishing Email") == 1
    assert _parse_binary_label("spam") == 1
    assert _parse_binary_label(2) is None
    assert _parse_binary_label("unknown") is None


def test_cleaning_rejects_placeholders_bad_labels_corruption_and_conflicts():
    conflict = _text("same-content-conflicting-labels")
    duplicate = _text("same-label-duplicate")
    rows = [
        {"text": _text("legit"), "label": 0},
        {"text": _text("phish"), "label": 1},
        {"text": duplicate, "label": 0},
        {"text": duplicate, "label": 0},
        {"text": conflict, "label": 0},
        {"text": conflict, "label": 1},
        {"text": "empty", "label": 0},
        {"text": _text("bad-label"), "label": "mystery"},
        {"text": "x" * (MAX_TEXT_CHARS + 1), "label": 1},
    ]

    cleaned, stats = _clean_dataset_frame(pd.DataFrame(rows))

    assert len(cleaned) == 3
    assert stats["exact_duplicates"] == 1
    assert stats["exact_label_conflicts"] == 2
    assert stats["invalid_text"] == 1
    assert stats["invalid_label"] == 1
    assert stats["too_long"] == 1
    assert text_hash(conflict.lower()) not in set(cleaned["text_hash"])


def test_template_dedup_removes_variants_and_preserves_unicode():
    rows = pd.DataFrame(
        [
            {"text": _text("invoice 123 https://one.example/a"), "label": 1},
            {"text": _text("invoice 999 https://two.example/b"), "label": 1},
            {"text": "Avviso sicurezza account italiano con caratteri validi e testo abbastanza lungo.", "label": 0},
            {"text": "Messaggio aziendale greco alpha beta con contenuto differente e abbastanza lungo.", "label": 0},
        ]
    )
    deduped, stats = _dedupe_templates(rows)
    assert len(deduped) == 3
    assert stats["template_duplicates"] == 1
    assert template_hash("Email valida in lingua italiana con caratteri accentati e testo lungo")


def test_synthetic_rows_are_train_only_when_real_test_data_exists():
    rows = []
    for label in (0, 1):
        for index in range(20):
            text = _text(f"real-{label}-{index}")
            rows.append({"text": text, "label": label, "source": "real", "source_file": str(index), "text_hash": text_hash(text)})
        for index in range(5):
            text = _text(f"synthetic-{label}-{index}")
            rows.append({"text": text, "label": label, "source": "kaggle_phishing_and_legitimate_emails", "source_file": str(index), "text_hash": text_hash(text)})

    split = _assign_splits(pd.DataFrame(rows))
    synthetic = split[split["source"] == "kaggle_phishing_and_legitimate_emails"]
    assert set(synthetic["split"]) == {"train"}
    assert set(split[split["source"] == "real"]["split"]) == {"train", "validation", "test"}