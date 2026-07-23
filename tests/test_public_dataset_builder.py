from pathlib import Path
import zipfile

import pandas as pd
import src.public_dataset_builder as dataset_builder

from src.public_dataset_builder import (
    BuildResult,
    FORCED_SOURCE_SPLITS,
    MAX_TEXT_CHARS,
    _assign_source_holdout_splits,
    _assign_splits,
    _clean_dataset_frame,
    _dedupe_templates,
    _parse_binary_label,
    _rows_from_phishing_pot_zip,
    _rows_from_spaphish_frame,
    _rows_from_zenodo_validation_frame,
    build_balanced_public_dataset,
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


def test_source_holdout_keeps_real_sources_out_of_other_splits():
    rows = []
    source_words = {
        0: ("calendar", "project", "meeting"),
        1: ("credential", "invoice", "airdrop"),
    }
    for label in (0, 1):
        for source_index, source_word in enumerate(source_words[label]):
            source = f"source-{label}-{source_word}"
            for row_index in range(4):
                text = _text(
                    f"{source_word} {'routine collaboration notes' if label == 0 else 'urgent security action'} "
                    f"variant {chr(97 + row_index)}"
                )
                rows.append({
                    "text": text,
                    "label": label,
                    "source": source,
                    "source_file": str(row_index),
                    "text_hash": text_hash(text),
                })

    split = _assign_source_holdout_splits(pd.DataFrame(rows))

    assert not split.groupby("source")["split"].nunique().gt(1).any()
    assert set(split["split"]) == {"train", "validation", "test"}
    assert all(set(group["label"]) == {0, 1} for _, group in split.groupby("split"))
    assert _parse_binary_label(2) is None
    assert _parse_binary_label("unknown") is None


def test_source_holdout_uses_two_training_sources_when_four_are_available():
    rows = []
    source_phrases = {
        0: ("quarterly calendar", "engineering roadmap", "cafeteria menu", "travel itinerary"),
        1: ("credential theft", "fraudulent invoice", "wallet airdrop", "account takeover"),
    }
    for label in (0, 1):
        for source_index, phrase in enumerate(source_phrases[label]):
            source = f"source-{label}-{source_index}"
            for row_index in range(4):
                text = _text(f"{phrase} independent corpus variant {row_index}")
                rows.append(
                    {
                        "text": text,
                        "label": label,
                        "source": source,
                        "source_file": str(row_index),
                        "text_hash": text_hash(text),
                    }
                )

    split = _assign_source_holdout_splits(pd.DataFrame(rows))
    train = split[split["split"] == "train"]

    assert train[train["label"] == 0]["source"].nunique() >= 2
    assert train[train["label"] == 1]["source"].nunique() >= 2


def test_forced_sources_keep_declared_validation_and_test_splits():
    rows = []
    unique_tokens = (
        "albatross",
        "birchwood",
        "cinnamon",
        "dragonfly",
        "evergreen",
        "firebrick",
        "goldfinch",
        "honeycomb",
        "ironstone",
        "jellyfish",
        "kingfisher",
        "limestone",
    )
    for source_index, source in enumerate(FORCED_SOURCE_SPLITS):
        for label in (0, 1):
            token = unique_tokens[source_index * 2 + label]
            text = " ".join([token] * 30)
            rows.append(
                {
                    "text": text,
                    "label": label,
                    "source": source,
                    "source_file": f"{source}-{label}",
                    "text_hash": text_hash(text),
                }
            )

    split = _assign_source_holdout_splits(pd.DataFrame(rows))

    assert not split.groupby("source")["split"].nunique().gt(1).any()
    for source, expected_split in FORCED_SOURCE_SPLITS.items():
        assert set(split.loc[split["source"] == source, "split"]) == {expected_split}


def test_spaphish_rows_keep_one_source_without_filtering_email_dates():
    frame = pd.DataFrame(
        [
            {"hash": "old", "subject": "Old", "body": _text("old body"), "date": "01/01/2021", "Label": 0},
            {"hash": "new-safe", "subject": "Seguro", "body": _text("modern safe"), "date": "01/06/2022", "Label": 0},
            {"hash": "new-phish", "subject": "Alerta", "body": _text("modern phish"), "date": "15/10/2025", "Label": 1},
            {"hash": "missing", "subject": "Unknown", "body": _text("unknown date"), "date": None, "Label": 1},
        ]
    )

    rows = _rows_from_spaphish_frame(frame)

    assert len(rows) == 4
    assert {row["source"] for row in rows} == {"spaphish"}
    assert {row["label"] for row in rows} == {0, 1}


def test_split_drops_near_duplicate_campaigns_with_conflicting_labels():
    ambiguous_safe = _text("account verification portal immediately alpha")
    ambiguous_phish = _text("account verification portal immediately beta")
    rows = [
        {
            "text": ambiguous_safe,
            "label": 0,
            "source": "safe-source",
            "source_file": "safe-ambiguous",
            "text_hash": text_hash(ambiguous_safe),
        },
        {
            "text": ambiguous_phish,
            "label": 1,
            "source": "phish-source",
            "source_file": "phish-ambiguous",
            "text_hash": text_hash(ambiguous_phish),
        },
    ]
    for label, token in ((0, "legitimate correspondence"), (1, "malicious credential theft")):
        for index in range(4):
            text = _text(f"{token} independent example {index}")
            rows.append(
                {
                    "text": text,
                    "label": label,
                    "source": f"source-{label}",
                    "source_file": str(index),
                    "text_hash": text_hash(text),
                }
            )

    split = _assign_splits(pd.DataFrame(rows))

    assert text_hash(ambiguous_safe) not in set(split["text_hash"])
    assert text_hash(ambiguous_phish) not in set(split["text_hash"])


def test_phishing_pot_rows_do_not_filter_untrusted_date_headers(tmp_path: Path):
    archive_path = tmp_path / "phishing-pot.zip"
    messages = {
        "repo/email/modern.eml": (
            b"Date: Tue, 10 Jun 2025 12:00:00 +0000\r\n"
            b"Subject: Modern alert\r\n\r\n"
            b"Verify your account using the secure portal immediately."
        ),
        "repo/email/old.eml": (
            b"Date: Tue, 10 Jun 2020 12:00:00 +0000\r\n"
            b"Subject: Old alert\r\n\r\n"
            b"This historical message must not enter the modern dataset."
        ),
        "repo/email/future.eml": (
            b"Date: Tue, 10 Jun 2033 12:00:00 +0000\r\n"
            b"Subject: Future alert\r\n\r\n"
            b"This implausible future message must not enter the dataset."
        ),
        "repo/email/missing.eml": (
            b"Subject: Missing date\r\n\r\n"
            b"This undated message cannot satisfy the temporal policy."
        ),
    }
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name, content in messages.items():
            archive.writestr(name, content)

    rows = _rows_from_phishing_pot_zip(archive_path)

    assert len(rows) == 4
    assert {row["source_file"] for row in rows} == set(messages)


def test_zenodo_rows_map_safe_and_phishing_labels():
    frame = pd.DataFrame(
        {
            "Email Text": [_text("safe zenodo"), _text("phishing zenodo")],
            "Email Type": ["Safe Email", "Phishing Email"],
        }
    )

    rows = list(_rows_from_zenodo_validation_frame(frame))

    assert [row["label"] for row in rows] == [0, 1]
    assert {row["source"] for row in rows} == {"zenodo_2024"}


def test_historical_sources_are_accepted(monkeypatch, tmp_path: Path):
    def fake_result(source: str):
        def add_source(**kwargs):
            return BuildResult(source, 1, 1, 0, 0, "ok")
        return add_source

    monkeypatch.setattr(dataset_builder, "add_enron_sample", fake_result("enron"))
    monkeypatch.setattr(dataset_builder, "add_spamassassin", fake_result("spamassassin"))
    monkeypatch.setattr(dataset_builder, "add_github_phishing_pot", fake_result("github_phishing_pot"))
    monkeypatch.setattr(
        dataset_builder,
        "dataset_stats",
        lambda path: {"legitimate": 2, "phishing": 1},
    )
    monkeypatch.setattr(
        dataset_builder,
        "balance_dataset",
        lambda *args, **kwargs: {
            "rows": 3,
            "per_class": None,
            "output": str(kwargs["output_csv"]),
            "template_duplicates": 0,
            "label_conflicts": 0,
        },
    )

    result = build_balanced_public_dataset(
        ["enron", "spamassassin", "github_phishing_pot"],
        output_csv=tmp_path / "public.csv",
        staging_csv=tmp_path / "staging.csv",
    )

    assert result["status"] == "ok"
    assert {item.source for item in result["results"]} == {
        "enron",
        "spamassassin",
        "github_phishing_pot",
    }


def test_campaign_split_mixes_each_source_and_label_across_splits():
    rows = []
    for source in ("corpus-alpha", "corpus-beta"):
        for label in (0, 1):
            for index in range(30):
                class_word = "legitimate" if label == 0 else "malicious"
                token = f"{source}-{class_word}-{chr(97 + index // 26)}{chr(97 + index % 26)}"
                text = " ".join([token] * 20)
                rows.append(
                    {
                        "text": text,
                        "label": label,
                        "source": source,
                        "source_file": token,
                        "text_hash": text_hash(text),
                    }
                )

    split = _assign_splits(pd.DataFrame(rows))

    for _, stratum in split.groupby(["source", "label"]):
        assert stratum["split"].value_counts().to_dict() == {
            "train": 21,
            "test": 6,
            "validation": 3,
        }
    assert not split.groupby("campaign_id")["split"].nunique().gt(1).any()


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
    danger_topics = [
        "amber", "bronze", "crimson", "denim", "emerald", "fuchsia", "ginger", "hazel", "ivory", "jade",
        "khaki", "lilac", "magenta", "navy", "ochre", "pearl", "quartz", "ruby", "scarlet", "teal",
    ]
    for label in (0, 1):
        for index in range(20):
            topic = (topics if label == 0 else danger_topics)[index]
            unit = f"{topic}{'safe' if label == 0 else 'danger'}"
            class_context = "routine project collaboration" if label == 0 else "urgent credential theft warning"
            text = f"Subject {unit} {class_context} " + " ".join([unit] * 50)
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
            class_context = "routine team scheduling" if label == 0 else "credential harvesting request"
            text = _text(f"independent alphabetic campaign {chr(97 + index)} {class_context}")
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


def test_complete_dataset_downsamples_synthetic_rows_to_configured_cap(tmp_path: Path):
    public_rows = []
    for label in (0, 1):
        for index in range(50):
            split = "train" if index < 40 else "validation" if index < 45 else "test"
            token = chr(97 + index // 26) + chr(97 + index % 26)
            text = _text(
                f"public cap {token} {'routine collaboration' if label == 0 else 'credential theft'}"
            )
            public_rows.append(
                {
                    "text": text,
                    "label": label,
                    "source": f"public-cap-{split}-{label}",
                    "source_file": str(index),
                    "text_hash": text_hash(text),
                    "campaign_id": f"campaign-{label}-{index}",
                    "split": split,
                }
            )
    synthetic_rows = [
        {
            "text": _text(
                f"synthetic cap {chr(97 + index // 26)}{chr(97 + index % 26)} "
                f"{'benign notice' if label == 0 else 'malicious request'}"
            ),
            "label": label,
            "source": "synthetic_modern_v2",
            "source_file": str(index),
        }
        for label in (0, 1)
        for index in range(30)
    ]
    public_csv = tmp_path / "public-cap.csv"
    synthetic_csv = tmp_path / "synthetic-cap.csv"
    output_csv = tmp_path / "complete-cap.csv"
    pd.DataFrame(public_rows).to_csv(public_csv, index=False)
    pd.DataFrame(synthetic_rows).to_csv(synthetic_csv, index=False)

    result = combine_public_and_synthetic_datasets(
        public_csv=public_csv,
        synthetic_csv=synthetic_csv,
        output_csv=output_csv,
        max_synthetic_train_fraction=0.10,
    )

    assert result["status"] == "ok"
    assert result["synthetic_rows"] < result["synthetic_rows_available"]
    assert result["synthetic_train_fraction"] <= 0.10
