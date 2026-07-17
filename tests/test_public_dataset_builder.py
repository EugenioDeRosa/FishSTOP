from pathlib import Path

import pandas as pd

from src.public_dataset_builder import (
    MAX_TEXT_CHARS,
    _assign_splits,
    _clean_dataset_frame,
    _dedupe_templates,
    _parse_binary_label,
    combine_public_and_synthetic_datasets,
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
    topics = [
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet",
        "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango",
    ]
    for label in (0, 1):
        for index in range(20):
            topic = topics[index]
            unit = f"{topic}{'safe' if label == 0 else 'danger'}"
            text = f"Subject {unit} " + " ".join([unit] * 50)
            rows.append({"text": text, "label": label, "source": "real", "source_file": str(index), "text_hash": text_hash(text)})
        for index in range(5):
            text = _text(f"synthetic-{label}-{index}")
            source = "synthetic_modern_v2" if index == 0 else "kaggle_phishing_and_legitimate_emails"
            rows.append({"text": text, "label": label, "source": source, "source_file": str(index), "text_hash": text_hash(text)})

    split = _assign_splits(pd.DataFrame(rows))
    synthetic = split[split["source"] != "real"]
    assert set(synthetic["split"]) == {"train"}
    assert set(split[split["source"] == "real"]["split"]) == {"train", "validation", "test"}


def test_near_duplicate_campaign_variants_stay_in_the_same_split():
    rows = []
    for label in (0, 1):
        for index in range(20):
            text = _text(f"independent alphabetic campaign {chr(97 + index)} labelword {label}")
            rows.append(
                {
                    "text": text,
                    "label": label,
                    "source": "real",
                    "source_file": str(index),
                    "text_hash": text_hash(text),
                }
            )
    base = (
        "Security notification: your mailbox storage requires review. "
        "Open the account portal and confirm the information shown in the dashboard. "
    )
    for suffix in ("Reference alpha one", "Reference alpha two"):
        text = base + suffix
        rows.append(
            {
                "text": text,
                "label": 1,
                "source": "real",
                "source_file": suffix,
                "text_hash": text_hash(text),
            }
        )

    split = _assign_splits(pd.DataFrame(rows))
    campaign = split[split["text"].str.startswith("Security notification")]
    assert campaign["campaign_id"].nunique() == 1
    assert campaign["split"].nunique() == 1


def test_complete_dataset_keeps_synthetic_train_only_and_balanced(tmp_path: Path):
    public_rows = []
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    for label in (0, 1):
        for index in range(20):
            split = "train" if index < 12 else "validation" if index < 16 else "test"
            token = alphabet[index] + alphabet[label + 20]
            text = _text(f"public {token} classword {'safe' if label == 0 else 'danger'}")
            public_rows.append(
                {
                    "text": text,
                    "label": label,
                    "source": "public_real",
                    "source_file": token,
                    "text_hash": text_hash(text),
                    "split": split,
                }
            )

    synthetic_rows = []
    for label in (0, 1):
        for token in ("omega", "sigma"):
            text = _text(f"synthetic {token} {'benign' if label == 0 else 'attack'}")
            synthetic_rows.append(
                {
                    "text": text,
                    "label": label,
                    "source": "synthetic_modern_v2",
                    "source_file": token,
                }
            )

    public_csv = tmp_path / "public.csv"
    synthetic_csv = tmp_path / "synthetic.csv"
    output_csv = tmp_path / "complete.csv"
    pd.DataFrame(public_rows).to_csv(public_csv, index=False)
    pd.DataFrame(synthetic_rows).to_csv(synthetic_csv, index=False)

    result = combine_public_and_synthetic_datasets(
        public_csv=public_csv,
        synthetic_csv=synthetic_csv,
        output_csv=output_csv,
        max_synthetic_train_fraction=0.50,
    )

    assert result["status"] == "ok"
    complete = pd.read_csv(output_csv)
    assert "campaign_id" in complete.columns
    assert complete["label"].value_counts().to_dict() == {0: 22, 1: 22}
    assert set(complete.loc[complete["source"] == "synthetic_modern_v2", "split"]) == {"train"}
    assert not complete.loc[complete["split"].isin(["validation", "test"]), "source"].str.startswith("synthetic_").any()
